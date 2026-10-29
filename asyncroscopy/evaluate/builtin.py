"""Built-in evaluators: information, resolution, dose and time.

Each one reads only the trace and the data; none of them calls an instrument. Where the
twin's ground truth is given (``RunData.truth``), information and resolution use it, and say
so in their reasons. That truth is the twin's own model, so a score against it tells how well
a run did *on the twin*, not how it would do on a specimen (docs/Operation/evaluators.md).
"""

from __future__ import annotations

import math

import numpy as np

from asyncroscopy.evaluate.core import Evaluator, Score, register_evaluator
from asyncroscopy.evaluate.run import RunData, event_result, match_particles


def _proxy_scorer():
    """pyTEMlib's baseline field-of-view scorer (``pyTEMlib.scoring``): the stand-in where there is no ground truth."""
    from pyTEMlib.scoring import available_scorers

    return available_scorers()["baseline"]


def _sharp_features(image) -> tuple[int, float, str]:
    """(features with sharp edges, score 0-1, reason) from pyTEMlib's baseline scorer."""
    from pyTEMlib.scoring.baseline import EDGE_WIDTH_MID_PX

    scorer = _proxy_scorer()
    measured = scorer.measure(image.array)
    sharp = sum(f["edge_width_px"] < EDGE_WIDTH_MID_PX for f in measured["features"])
    value, reason = scorer.score(image.array, dict(image.attrs))
    return int(sharp), float(value), reason


IN_FOCUS_M = 20e-9  # |C1, A1| below this counts as in focus: about the depth of field of a 30 mrad probe near 1 nm resolution
IN_FOCUS_SCORE = 0.5  # without the twin's focus truth: a pyTEMlib baseline score at or above this counts as in focus


def in_focus(data: RunData, image, tolerance_m: float = IN_FOCUS_M, sharp_score: float = IN_FOCUS_SCORE) -> dict:
    """Whether ``image`` was taken in focus: from the focus error the bench recorded (twin), else pyTEMlib's baseline score.

    Returns {"ok": bool, "error": m} or {"ok": bool, "sharpness": 0-1, "reason": ...}."""
    truth = data.focus_truth(image)
    if truth is not None and truth.get("c1_m") is not None:
        error = math.hypot(float(truth.get("c1_m") or 0.0), float(truth.get("a1_abs_m") or 0.0))
        return {"error": error, "ok": error <= float(tolerance_m)}
    _sharp, value, reason = _sharp_features(image)
    return {"sharpness": value, "ok": value >= float(sharp_score), "reason": reason}


# ----------------------------------------------------------------------
@register_evaluator
class Information(Evaluator):
    """Features found, per unit of dose and per unit of time.

    With the twin's ground truth: particles a detected feature lands on (``match_particles``),
    counted once per run however often they were imaged. Without it: features with sharp
    edges by pyTEMlib's baseline scorer, summed over images (a proxy: the same particle imaged
    twice counts twice, and nothing checks it is real).

    ``features_in_focus`` counts only features found in frames taken in focus (``in_focus``):
    finding a particle and resolving it are different objectives, and they disagree about how
    much tuning pays (docs/Operation/evaluators.md).
    """

    name = "information"
    description = "Ground-truth features found (twin) or sharp-edged features (pyTEMlib proxy), per e/A^2 and per second"
    objectives = {
        "features_found": ("features", "max"),
        "features_in_focus": ("features", "max"),
        "features_per_s": ("features/s", "max"),
        "features_per_dose": ("features per e/A^2", "max"),
        "recall": ("fraction", "max"),
        "images": ("images", "max"),
    }
    option_names = ("k_sigma", "duration_s", "in_focus_m", "in_focus_score")
    k_sigma = 5.0  # detection threshold, in noise standard deviations
    in_focus_m = IN_FOCUS_M
    in_focus_score = IN_FOCUS_SCORE
    duration_s = None  # time to divide by; default: the run's instrument-clock span (tuning time included)

    def _image_scores(self, data: RunData, image) -> tuple[dict, str]:
        matched = match_particles(data, image, k=float(self.k_sigma))
        if matched is not None:
            return matched, "twin ground truth"
        sharp, _value, _reason = _sharp_features(image)
        return {"found_proxy": sharp}, "pyTEMlib baseline proxy"

    def score_event(self, event, data):
        images = data.images(event)
        if not images:
            return {}
        found = in_field = 0
        dose = seconds = 0.0
        source = ""
        for image in images:
            scores, source = self._image_scores(data, image)
            found += len(scores["found"]) if "found" in scores else scores["found_proxy"]
            in_field += len(scores.get("in_field", []))
            dose += image.frame_dose_e_per_a2() or 0.0
            seconds += image.frame_time_s() or 0.0
        out = {
            "features_found": Score(found, "features", f"{found} found in {len(images)} image(s) ({source})"),
            "features_per_dose": Score(found / dose if dose else None, "features per e/A^2", f"frame dose {dose:.4g} e/A^2"),
            "features_per_s": Score(found / seconds if seconds else None, "features/s", f"frame time {seconds:.4g} s"),
        }
        if in_field:
            out["recall"] = Score(found / in_field, "fraction", f"{found} of {in_field} ground-truth particles in the field")
        return out

    def score_run(self, data):
        pairs = data.all_images()
        unique: set[int] = set()
        in_field: set[int] = set()
        focused_unique: set[int] = set()
        proxy = proxy_focused = 0
        dose = 0.0
        truth_used = False
        for _event, image in pairs:
            scores, _source = self._image_scores(data, image)
            focused = in_focus(data, image, float(self.in_focus_m), float(self.in_focus_score))["ok"]
            if "found" in scores:
                truth_used = True
                unique.update(scores["found"])
                in_field.update(scores["in_field"])
                if focused:
                    focused_unique.update(scores["found"])
            else:
                proxy += scores["found_proxy"]
                proxy_focused += scores["found_proxy"] if focused else 0
            dose += image.frame_dose_e_per_a2() or 0.0
        found = len(unique) + proxy
        duration = float(self.duration_s) if self.duration_s else data.clock_span_s()
        how = "unique ground-truth particles found (twin)" if truth_used else "sharp-edged features, pyTEMlib baseline proxy (no ground truth)"
        if truth_used and proxy:
            how += "; plus a proxy count for images the truth could not place"
        missing = f"; {len(data.missing)} data key(s) unreadable" if data.missing else ""
        out = {
            "features_found": Score(found, "features", f"{how}, {len(pairs)} image(s){missing}"),
            "features_in_focus": Score(
                len(focused_unique) + proxy_focused, "features",
                f"as features_found, from frames in focus only (error <= {float(self.in_focus_m):g} m; baseline score >= {float(self.in_focus_score):g} without focus truth)",
            ),
            "features_per_s": Score(found / duration if duration else None, "features/s", f"over {duration:.4g} s of instrument clock" if duration else "no instrument clock span in the trace"),
            "features_per_dose": Score(found / dose if dose else None, "features per e/A^2", f"over {dose:.4g} e/A^2 summed over frames"),
            "images": Score(len(pairs), "images", "images the trace references and that could be read"),
        }
        if in_field:
            out["recall"] = Score(len(unique) / len(in_field), "fraction", f"{len(unique)} of {len(in_field)} ground-truth particles in the imaged fields")
        return out


# ----------------------------------------------------------------------
@register_evaluator
class Resolution(Evaluator):
    """Image quality: the focus error behind each image (twin, when recorded) or pyTEMlib's baseline score."""

    name = "resolution"
    description = "Defocus/astigmatism error at each frame (twin, when the bench recorded it), else pyTEMlib baseline sharpness"
    objectives = {
        "focus_error_m": ("m", "min"),
        "in_focus_fraction": ("fraction", "max"),
        "sharpness": ("score 0-1", "max"),
    }
    option_names = ("tolerance_m", "sharp_score")
    tolerance_m = IN_FOCUS_M
    sharp_score = IN_FOCUS_SCORE

    def _one(self, data, image) -> dict:
        return in_focus(data, image, float(self.tolerance_m), float(self.sharp_score))

    def score_event(self, event, data):
        images = data.images(event)
        if not images:
            return {}
        one = self._one(data, images[0])
        if "error" in one:
            return {"focus_error_m": Score(one["error"], "m", "hypot(C1, |A1|) at this frame (twin ground truth)")}
        return {"sharpness": Score(one["sharpness"], "score 0-1", one["reason"])}

    def score_run(self, data):
        results = [self._one(data, image) for _event, image in data.all_images()]
        if not results:
            return {"in_focus_fraction": Score(None, "fraction", "no images in this run")}
        ok = sum(r["ok"] for r in results) / len(results)
        errors = [r["error"] for r in results if "error" in r]
        sharp = [r["sharpness"] for r in results if "sharpness" in r]
        out = {"in_focus_fraction": Score(ok, "fraction", f"{len(results)} image(s); in focus: error <= {self.tolerance_m:g} m (twin) or baseline score >= {self.sharp_score:g}")}
        if errors:
            out["focus_error_m"] = Score(float(np.median(errors)), "m", f"median hypot(C1, |A1|) over {len(errors)} frame(s), twin ground truth; worst {max(errors):.3g} m")
        if sharp:
            out["sharpness"] = Score(float(np.mean(sharp)), "score 0-1", f"mean pyTEMlib baseline score over {len(sharp)} image(s)")
        return out


# ----------------------------------------------------------------------
@register_evaluator
class Dose(Evaluator):
    """Cumulative dose per area: frames laid on a grid at their field, plus the dose jobs report (e.g. the Tuner's)."""

    name = "dose"
    description = "Cumulative e/A^2 on the imaged area (max, mean, fraction over a threshold) and dose reported by jobs"
    objectives = {
        "max_dose_e_per_a2": ("e/A^2", "min"),
        "mean_dose_e_per_a2": ("e/A^2", "min"),
        "over_threshold_fraction": ("fraction", "min"),
        "exposed_area_m2": ("m^2", "max"),
        "job_dose_e_per_a2": ("e/A^2", "min"),
    }
    option_names = ("threshold_e_per_a2", "cell_m")
    threshold_e_per_a2 = 1000.0  # a beam-sensitive specimen's budget; set per specimen
    cell_m = None  # grid cell; default: a quarter of the smallest field of view

    def score_event(self, event, data):
        doses = [d for d in (image.frame_dose_e_per_a2() for image in data.images(event)) if d is not None]
        reported = event_result(event).get("dose_e_per_a2")
        if doses:
            return {"frame_dose_e_per_a2": Score(float(sum(doses)), "e/A^2", "current x dwell / pixel area, from the data's attributes")}
        if isinstance(reported, (int, float)):
            return {"job_dose_e_per_a2": Score(float(reported), "e/A^2", f"reported by {event.get('action')}")}
        return {}

    def score_run(self, data):
        frames = []
        for _event, image in data.all_images():
            dose, centre, fov = image.frame_dose_e_per_a2(), data.centre_m(image), image.fov_m
            if dose is not None and centre is not None and fov is not None:
                frames.append((dose, centre, fov))
        # Jobs that report their own dose (the Tuner): counted, not placed (their events carry no field position).
        job = 0.0
        for event in data.events:
            if str(event.get("action", "")).endswith(".step") and event.get("ok"):
                value = event_result(event).get("dose_e_per_a2")
                if isinstance(value, (int, float)):
                    job += float(value)
        out = {"job_dose_e_per_a2": Score(job, "e/A^2", "summed dose_e_per_a2 of job steps (not placed on the grid: no position)")}
        if not frames:
            out["max_dose_e_per_a2"] = Score(None, "e/A^2", "no image with a dose and a position")
            return out
        cell = float(self.cell_m) if self.cell_m else min(min(f[2]) for f in frames) / 4.0
        grid: dict[tuple[int, int], float] = {}
        for dose, (cx, cy), (fx, fy) in frames:
            i0, i1 = math.floor((cx - fx / 2) / cell + 1e-9), math.ceil((cx + fx / 2) / cell - 1e-9)
            j0, j1 = math.floor((cy - fy / 2) / cell + 1e-9), math.ceil((cy + fy / 2) / cell - 1e-9)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    grid[(i, j)] = grid.get((i, j), 0.0) + dose
        values = np.fromiter(grid.values(), dtype=np.float64)
        threshold = float(self.threshold_e_per_a2)
        out.update({
            "max_dose_e_per_a2": Score(float(values.max()), "e/A^2", f"{len(frames)} frame(s) on a {cell:.3g} m grid"),
            "mean_dose_e_per_a2": Score(float(values.mean()), "e/A^2", "mean over exposed cells"),
            "over_threshold_fraction": Score(float((values > threshold).mean()), "fraction", f"exposed area over {threshold:g} e/A^2"),
            "exposed_area_m2": Score(float(len(values) * cell * cell), "m^2", "area of cells any frame touched"),
        })
        if data.truth and isinstance(data.truth.get("dose"), dict):
            twin = data.truth["dose"].get("max_dose_e_per_a2")
            if isinstance(twin, (int, float)):
                out["twin_max_dose_e_per_a2"] = Score(float(twin), "e/A^2", "the twin's own dose map (every exposure, tuning included)")
        return out


# ----------------------------------------------------------------------
@register_evaluator
class Time(Evaluator):
    """Wall time, instrument-clock time, and the wall time in which nothing was running (idle)."""

    name = "time"
    description = "Wall and instrument-clock time of the run, busy time (union of event durations) and idle time"
    objectives = {
        "wall_s": ("s", "min"),
        "clock_s": ("s", "min"),
        "busy_s": ("s", "max"),
        "idle_s": ("s", "min"),
        "idle_fraction": ("fraction", "min"),
    }

    def score_event(self, event, data):
        duration = event.get("duration_s")
        return {"duration_s": Score(float(duration), "s", "wall time of this call or step")} if isinstance(duration, (int, float)) else {}

    def score_run(self, data):
        spans = []
        for event in data.events:
            start = event.get("ts")
            if not isinstance(start, (int, float)):
                continue
            duration = event.get("duration_s") if isinstance(event.get("duration_s"), (int, float)) else 0.0
            end = max(float(start) + float(duration), float(event.get("written", start)) if isinstance(event.get("written"), (int, float)) else float(start))
            spans.append((float(start), end, float(duration)))
        if not spans:
            return {"wall_s": Score(None, "s", "no timed events")}
        wall = max(s[1] for s in spans) - min(s[0] for s in spans)
        busy, current_start, current_end = 0.0, None, None
        for start, _end, duration in sorted((s for s in spans if s[2] > 0), key=lambda s: s[0]):
            end = start + duration
            if current_end is None or start > current_end:
                busy += (current_end - current_start) if current_end is not None else 0.0
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        if current_end is not None:
            busy += current_end - current_start
        busy = min(busy, wall)
        clock = data.clock_span_s()
        return {
            "wall_s": Score(wall, "s", f"first event start to last event end, {len(spans)} event(s)"),
            "clock_s": Score(clock, "s", "instrument clock span (simulated seconds on a twin)" if clock is not None else "no instrument clock in the trace"),
            "busy_s": Score(busy, "s", "union of tool-call and job-step durations"),
            "idle_s": Score(wall - busy, "s", "wall time with no call or step running (thinking, waiting, approvals)"),
            "idle_fraction": Score((wall - busy) / wall if wall > 0 else 0.0, "fraction", "idle / wall"),
        }
