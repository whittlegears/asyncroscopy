"""Verified "return to this spot", and the operations it depends on.

The stage coordinate saved with a site is a starting guess. A return visit:

1. moves to the predicted position (saved coordinate through the current mount
   transform),
2. takes a lower-magnification *survey* and finds the stored reference in it
   (rotation search when the specimen may have been remounted),
3. moves to centre what it found, using the calibrated pixel-to-stage mapping,
4. acquires at the site's native settings and registers against the original
   image again, iterating until the residual is within tolerance,
5. refuses to call it a match unless every check passes (see
   :func:`asyncroscopy.atlas.registration.judge`), and
6. records the visit, match or not, in the atlas with its image and evidence.

Everything here talks to the instrument through a small :class:`Microscope`
protocol and runs each instrument operation through a *step* callback. The MCP
server passes the action queue's step, so every move and acquisition shows up
as a child action that the operator can watch, pause between, or abort.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from asyncroscopy.atlas.registration import (
    Location,
    MatchCriteria,
    center_crop,
    judge,
    locate,
    offset_from_center,
    rotation_range,
)
from asyncroscopy.atlas.stage_mapping import StageMapping, calibrate, fit_rigid
from asyncroscopy.atlas.store import Atlas


class Microscope(Protocol):
    """What the atlas needs from an instrument."""

    def get_stage(self) -> list[float]: ...

    def move_stage(self, position: list[float]) -> None: ...

    def acquire(self, fov_m: float, imsize: int) -> np.ndarray: ...

    def settings(self) -> dict: ...


# step(name, args, fn) runs fn and returns its result. The default just runs it.
Step = Callable[[str, dict, Callable[[], Any]], Any]


def direct_step(name: str, args: dict, fn: Callable[[], Any]) -> Any:
    return fn()


@dataclass
class RevisitParams:
    survey_factor: float = 4.0          # survey FOV / site FOV
    survey_imsize: int | None = None    # None: enough pixels that the reference shrinks only 2x
    search_tiles: int = 1               # survey tiles per side (1 = predicted position only; 3 = 3x3)
    max_rotation_deg: float | None = None   # None: 0 on the original mount, 10 after a remount
    rotation_step_deg: float = 1.0
    tolerance_px: float = 2.0           # accept when the residual is within this many native pixels
    max_iterations: int = 4
    verify_crop: float = 0.6            # central fraction of the reference used to verify
    criteria: MatchCriteria = field(default_factory=MatchCriteria)

    def to_json(self) -> dict:
        d = asdict(self)
        d["criteria"] = self.criteria.to_json()
        return d

    @classmethod
    def from_json(cls, data: dict | None) -> RevisitParams:
        data = dict(data or {})
        crit = MatchCriteria(**data.pop("criteria", {}) or {})
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(criteria=crit, **known)


def _stage_with(position: list[float], xy: tuple[float, float]) -> list[float]:
    out = [float(v) for v in position]
    out = (out + [0.0] * 5)[:5]
    out[0], out[1] = float(xy[0]), float(xy[1])
    return out


def _rotations(atlas: Atlas, params: RevisitParams) -> list[float]:
    if params.max_rotation_deg is not None:
        limit = params.max_rotation_deg
    elif atlas.mount_id == "original":
        limit = 0.0
    else:
        rotation_known = atlas.mount.fitted_at and len(atlas.mount.landmarks) >= 2
        limit = abs(atlas.mount.rotation_deg) + 2.0 if rotation_known else 10.0
    return rotation_range(limit, params.rotation_step_deg)


def _survey_tiles(n: int, spacing_m: float) -> list[tuple[float, float]]:
    """Offsets for an n x n survey grid, nearest the centre first."""
    half = (n - 1) / 2.0
    tiles = [((i - half) * spacing_m, (j - half) * spacing_m) for i in range(n) for j in range(n)]
    return sorted(tiles, key=lambda t: math.hypot(*t))


def find_site(atlas: Atlas, site_id: str, mic: Microscope, step: Step = direct_step,
              params: RevisitParams | None = None, start: list[float] | None = None) -> dict:
    """Locate a site from its predicted position and centre it, verified.

    Returns a result dict whose ``status`` is ``match`` only if the final
    native-magnification registration passed every check and converged within
    tolerance. Any other status means "not confirmed": the caller must not
    treat the current position as the site.
    """
    params = params or RevisitParams()
    site = atlas.get(site_id)
    reference = atlas.reference(site_id)
    mapping: StageMapping = atlas.mapping
    rotations = _rotations(atlas, params)
    native_px = site.pixel_size_m
    imsize = site.imsize
    predicted = start or atlas.predicted_stage(site_id)
    result: dict[str, Any] = {
        "site": site_id, "site_name": site.name, "predicted_stage": predicted,
        "params": params.to_json(), "mapping_calibrated": mapping.calibrated,
        "mount_id": atlas.mount_id, "rotations_searched_deg": [min(rotations), max(rotations)],
        "warnings": [],
    }
    if not mapping.calibrated:
        result["warnings"].append("stage-to-image mapping is not calibrated; using identity axes, "
                                  "which can make corrections diverge")

    # --- 1-2. Survey around the prediction --------------------------------------
    survey_fov = site.fov_m * params.survey_factor
    survey_size = int(params.survey_imsize or max(imsize, round(imsize * params.survey_factor / 2)))
    survey_px = survey_fov / survey_size
    scale = native_px / survey_px
    found: Location | None = None
    survey_image = None
    survey_stage = None
    tried = []
    for ox, oy in _survey_tiles(max(1, params.search_tiles), survey_fov * 0.8):
        pos = _stage_with(predicted, (predicted[0] + ox, predicted[1] + oy))
        step("move_stage", {"position": pos, "purpose": "survey"}, lambda p=pos: mic.move_stage(p))
        image = step("acquire_survey", {"fov_m": survey_fov, "imsize": survey_size},
                     lambda: mic.acquire(survey_fov, survey_size))
        loc = locate(reference, image, rotations_deg=rotations, scale=scale)
        verdict = judge(loc, params.criteria)
        tried.append({"stage": pos, "location": loc.to_json(), "verdict": verdict.to_json()})
        if verdict.ok:
            found, survey_image, survey_stage = loc, image, pos
            break
        if survey_image is None or loc.ncc >= max(t["location"]["ncc"] for t in tried):
            survey_image = image          # keep the most similar tile for the record
    result["survey"] = {"fov_m": survey_fov, "tiles": tried}
    if found is None:
        best = max(tried, key=lambda t: t["location"]["ncc"])
        result.update(status=best["verdict"]["status"], reasons=["survey: " + r for r in best["verdict"]["reasons"]]
                      or ["survey: site not found"], final_stage=mic.get_stage(), image=survey_image,
                      image_kind="survey")
        return result

    # --- 3. Centre what the survey found ----------------------------------------
    off_px = offset_from_center(found, survey_image.shape)
    dx, dy = mapping.stage_delta_for_image_offset((off_px[0] * survey_px, off_px[1] * survey_px))
    position = _stage_with(survey_stage, (survey_stage[0] + dx, survey_stage[1] + dy))
    step("move_stage", {"position": position, "purpose": "centre survey match"},
         lambda p=position: mic.move_stage(p))

    # --- 4. Verify at native settings, iterate ------------------------------------
    template = center_crop(reference, params.verify_crop)
    fine_rot = [found.rotation_deg] if rotations == [0.0] else \
        [found.rotation_deg + d for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    iterations = []
    image = None
    loc = None
    verdict = None
    for it in range(params.max_iterations):
        image = step("acquire", {"fov_m": site.fov_m, "imsize": imsize}, lambda: mic.acquire(site.fov_m, imsize))
        loc = locate(template, image, rotations_deg=fine_rot)
        verdict = judge(loc, params.criteria)
        res_px = offset_from_center(loc, image.shape)
        residual_px = math.hypot(*res_px)
        iterations.append({"stage": position, "residual_px": residual_px, "location": loc.to_json(),
                           "verdict": verdict.to_json()})
        if not verdict.ok or residual_px <= params.tolerance_px or it == params.max_iterations - 1:
            break                       # never end on an unverified move
        dx, dy = mapping.stage_delta_for_image_offset((res_px[0] * native_px, res_px[1] * native_px))
        position = _stage_with(position, (position[0] + dx, position[1] + dy))
        step("move_stage", {"position": position, "purpose": f"correction {it + 1}"},
             lambda p=position: mic.move_stage(p))
        fine_rot = [loc.rotation_deg] if len(fine_rot) == 1 else [loc.rotation_deg + d for d in (-0.5, 0.0, 0.5)]

    last = iterations[-1]
    residual_px = last["residual_px"]
    status = verdict.status
    reasons = list(verdict.reasons)
    if verdict.ok and residual_px > params.tolerance_px:
        status = "not_converged"
        reasons.append(f"residual {residual_px:.1f} px still above tolerance {params.tolerance_px:.1f} px "
                       f"after {len(iterations)} iterations")
    final_stage = position
    result.update(
        status=status, reasons=reasons, final_stage=final_stage,
        correction_from_prediction_nm=1e9 * math.hypot(final_stage[0] - predicted[0], final_stage[1] - predicted[1]),
        residual_px=residual_px, residual_nm=residual_px * native_px * 1e9,
        sigma_px=loc.sigma_px, sigma_nm=loc.sigma_px * native_px * 1e9, ncc=loc.ncc,
        ambiguity=loc.ambiguity, consistency=loc.consistency, rotation_deg=loc.rotation_deg,
        iterations=iterations, image=image, image_kind="native",
    )
    return result


def return_to_site(atlas: Atlas, site_id: str, mic: Microscope, step: Step = direct_step,
                   params: RevisitParams | None = None, provenance: dict | None = None) -> dict:
    """Return to a saved site, verify it, and record the visit (match or not) in the atlas."""
    started = time.time()
    result = find_site(atlas, site_id, mic, step, params)
    image = result.pop("image", None)
    result["settings"] = mic.settings()
    result["duration_s"] = time.time() - started
    result.update(provenance or {})
    visit = atlas.record_visit(site_id, result, image)
    return visit


def save_site(atlas: Atlas, mic: Microscope, name: str, fov_m: float, imsize: int, step: Step = direct_step,
              landmark: bool = False, notes: str = "", provenance: dict | None = None):
    """Image the current position at native settings and save it as a site."""
    image = step("acquire", {"fov_m": fov_m, "imsize": imsize}, lambda: mic.acquire(fov_m, imsize))
    stage = mic.get_stage()
    return atlas.add_site(name, image, stage, fov_m, landmark=landmark, notes=notes,
                          settings=mic.settings(), provenance=provenance)


def calibrate_mapping(atlas: Atlas, mic: Microscope, fov_m: float, imsize: int, step: Step = direct_step,
                      step_fraction: float = 0.15, criteria: MatchCriteria | None = None) -> StageMapping:
    """Measure the stage-to-image Jacobian at the current position and store it in the atlas.

    Each axis is stepped both ways from a freshly imaged start, so stage move
    noise and a constant offset cancel to first order: column = (s(+d) - s(-d)) / 2d.
    """
    criteria = criteria or MatchCriteria()
    start = [float(v) for v in mic.get_stage()]
    px = fov_m / imsize

    def shift_after(dx: float, dy: float) -> np.ndarray:
        step("move_stage", {"position": start, "purpose": "calibration start"}, lambda: mic.move_stage(start))
        base = step("acquire", {"fov_m": fov_m, "imsize": imsize, "purpose": "calibration start"},
                    lambda: mic.acquire(fov_m, imsize))
        pos = _stage_with(start, (start[0] + dx, start[1] + dy))
        step("move_stage", {"position": pos, "purpose": "calibration step"}, lambda: mic.move_stage(pos))
        moved = step("acquire", {"fov_m": fov_m, "imsize": imsize, "purpose": "calibration step"},
                     lambda: mic.acquire(fov_m, imsize))
        loc = locate(center_crop(base, max(0.3, 0.9 - 2 * step_fraction)), moved)
        verdict = judge(loc, criteria)
        if not verdict.ok:
            raise RuntimeError(f"calibration step ({dx:.3g}, {dy:.3g}) m could not be registered: "
                               f"{verdict.status}: {'; '.join(verdict.reasons)}")
        ox, oy = offset_from_center(loc, moved.shape)
        return np.array([ox * px, oy * px])

    def measure(dx: float, dy: float) -> tuple[float, float]:
        s = (shift_after(dx, dy) - shift_after(-dx, -dy)) / 2.0
        return float(s[0]), float(s[1])

    try:
        mapping = calibrate(measure, step_fraction * fov_m)
    finally:
        step("move_stage", {"position": start, "purpose": "calibration return"}, lambda: mic.move_stage(start))
    atlas.set_mapping(mapping)
    return mapping


def reacquire_landmarks(atlas: Atlas, mic: Microscope, step: Step = direct_step,
                        params: RevisitParams | None = None, provenance: dict | None = None) -> dict:
    """After a remount: find every landmark, then fit how the specimen moved.

    Landmarks are searched with a wide survey grid and a rotation search. Only
    landmarks that verify as ``match`` enter the fit; with fewer than two, the
    rotation is unknown and the result says so.
    """
    params = params or RevisitParams(survey_factor=6.0, search_tiles=3, max_rotation_deg=10.0)
    landmarks = atlas.landmarks()
    if not landmarks:
        raise RuntimeError("the atlas has no landmark sites; mark at least two well-separated sites as landmarks")
    found_old, found_new, names, results = [], [], [], []
    for lm in landmarks:
        visit = return_to_site(atlas, lm.id, mic, step, params, provenance)
        results.append({"site": lm.id, "name": lm.name, "status": visit["status"],
                        "residual_nm": visit.get("residual_nm"), "visit": visit["id"]})
        if visit["status"] == "match":
            found_old.append(lm.stage[:2])
            found_new.append(visit["final_stage"][:2])
            names.append(lm.id)
            # Provisional fit, so the next landmark's prediction already includes this one.
            atlas.set_mount_transform(fit_rigid(found_old, found_new, names))
    out: dict[str, Any] = {"landmarks": results, "verified": len(names)}
    if not names:
        out["status"] = "no_landmarks_verified"
        return out
    transform = fit_rigid(found_old, found_new, names)
    atlas.set_mount_transform(transform)
    out["transform"] = transform.to_json()
    out["status"] = "fitted" if len(names) >= 2 else "translation_only"
    if len(names) < 2:
        out["warning"] = "only one landmark verified: translation fitted, rotation unknown"
    return out
