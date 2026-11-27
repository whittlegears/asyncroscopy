"""Remount benchmark: save sites, unload and remount the specimen, go back to each.

It records, for each preselected site, whether the atlas claimed a match, how
close it actually got (ground truth from the simulator, which the atlas never
sees), and whether it ever **confidently picked the wrong spot**: a ``match``
whose true error is larger than a quarter of the field of view.

Synthetic backend (seconds, no Tango)::

    uv run python -m asyncroscopy.atlas.benchmark --sites 10 --rotation 4 --shift 3e-6 -2e-6

Digital twin over Tango (start the DigitalTwin stack first)::

    uv run python -m asyncroscopy.atlas.benchmark --backend twin --tango-host localhost --tango-port 9094 \\
        --fov 4e-9 --imsize 128 --spread 8e-9 --rotation 3 --shift 1e-9 -1e-9

Both write ``benchmark.json`` and ``benchmark.md`` under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from asyncroscopy.atlas import revisit
from asyncroscopy.atlas.store import Atlas


def _rot(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def pick_sites(rng: np.random.Generator, n: int, spread: float, min_sep: float,
               avoid: Callable[[np.ndarray], bool] | None = None) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _ in range(20000):
        if len(out) >= n:
            break
        p = rng.uniform(-spread / 2, spread / 2, size=2)
        if avoid and avoid(p):
            continue
        if all(np.linalg.norm(p - q) >= min_sep for q in out):
            out.append(p)
    if len(out) < n:
        raise RuntimeError(f"could only place {len(out)} of {n} sites {min_sep:g} m apart within {spread:g} m")
    return out


def run(mic, remount: Callable[[float, tuple[float, float]], None], truth: Callable[[list[float]], np.ndarray],
        out_dir: Path, *, n_sites: int, n_landmarks: int, fov: float, imsize: int, spread: float,
        rotation_deg: float, shift_m: tuple[float, float], seed: int,
        extra_sites: list[tuple[str, np.ndarray]] | None = None, log=print) -> dict:
    """Run the benchmark on any :class:`~asyncroscopy.atlas.revisit.Microscope`.

    *remount(rotation, shift)* changes the specimen mount; *truth(original_xy)*
    returns where a feature saved at *original_xy* now is in stage coordinates.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = Atlas(out_dir / f"atlas-{time.strftime('%Y%m%dT%H%M%S')}", specimen_id=f"benchmark-seed{seed}")
    t0 = time.time()

    mic.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    mapping = revisit.calibrate_mapping(atlas, mic, fov, imsize)
    log(f"calibrated stage->image: {np.round(mapping.stage_to_image, 3).tolist()}")

    # Landmarks far apart (corners), test sites spread in between.
    corners = [np.array([sx, sy]) * spread * 0.4 for sx, sy in ((-1, -1), (1, 1), (1, -1), (-1, 1))]
    landmarks = corners[:n_landmarks]
    sites = pick_sites(rng, n_sites, spread * 0.8, min_sep=fov * 1.5,
                       avoid=lambda p: any(np.linalg.norm(p - c) < fov * 2 for c in landmarks))
    plan = [("landmark", f"landmark-{i + 1}", p) for i, p in enumerate(landmarks)]
    plan += [("site", f"site-{i + 1:02d}", p) for i, p in enumerate(sites)]
    plan += [("extra", name, p) for name, p in (extra_sites or [])]
    saved = []
    for kind, name, p in plan:
        mic.move_stage([float(p[0]), float(p[1]), 0.0, 0.0, 0.0])
        site = revisit.save_site(atlas, mic, name, fov, imsize, landmark=(kind == "landmark"))
        saved.append((kind, site))
    log(f"saved {len(saved)} sites ({n_landmarks} landmarks)")

    remount(rotation_deg, shift_m)
    atlas.new_mount("remounted")
    log(f"remounted: rotation {rotation_deg} deg, shift {shift_m} m")

    lm = revisit.reacquire_landmarks(atlas, mic)
    log(f"landmarks: {lm['status']} ({lm['verified']}/{n_landmarks} verified); "
        f"fitted rotation {lm.get('transform', {}).get('rotation_deg')}")

    rows = []
    for kind, site in saved:
        if kind == "landmark":
            continue
        visit = revisit.return_to_site(atlas, site.id, mic)
        truth_xy = truth(site.stage[:2])
        err_nm = float(np.linalg.norm(np.asarray(mic.get_stage()[:2]) - truth_xy) * 1e9)
        wrong = visit["status"] == "match" and err_nm > 0.25 * fov * 1e9
        rows.append({"site": site.name, "kind": kind, "status": visit["status"], "true_error_nm": err_nm,
                     "reported_residual_nm": visit.get("residual_nm"), "reported_sigma_nm": visit.get("sigma_nm"),
                     "ncc": visit.get("ncc"), "ambiguity": visit.get("ambiguity"),
                     "confidently_wrong": wrong, "reasons": visit.get("reasons", []), "visit": visit["id"]})
        log(f"  {site.name:12s} {visit['status']:14s} true error {err_nm:8.1f} nm"
            + ("   <-- CONFIDENTLY WRONG" if wrong else ""))

    scored = [r for r in rows if r["kind"] == "site"]
    matches = [r for r in scored if r["status"] == "match"]
    errs = [r["true_error_nm"] for r in matches]
    summary = {
        "sites": len(scored), "found": len(matches),
        "refused": {s: sum(1 for r in scored if r["status"] == s) for s in ("ambiguous", "no_match", "not_converged")},
        "confidently_wrong": sum(r["confidently_wrong"] for r in rows),
        "median_error_nm": statistics.median(errs) if errs else None,
        "max_error_nm": max(errs) if errs else None,
        "fov_nm": fov * 1e9, "pixel_nm": fov / imsize * 1e9,
        "landmarks_verified": lm["verified"], "fitted_rotation_deg": lm.get("transform", {}).get("rotation_deg"),
        "true_rotation_deg": rotation_deg, "duration_s": time.time() - t0,
    }
    report = {"summary": summary, "rows": rows, "landmarks": lm, "atlas": str(atlas.root),
              "config": {"n_sites": n_sites, "n_landmarks": n_landmarks, "fov_m": fov, "imsize": imsize,
                         "spread_m": spread, "rotation_deg": rotation_deg, "shift_m": list(shift_m), "seed": seed}}
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "benchmark.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Return-to-site after remount",
        "",
        f"- Found (verified match): **{s['found']} / {s['sites']}**",
        f"- Confidently wrong (match, but true error > FOV/4): **{s['confidently_wrong']}**",
        f"- Refused: {', '.join(f'{k} {v}' for k, v in s['refused'].items())}",
        (f"- True error of matches: median {_n(s['median_error_nm'])} nm, max {_n(s['max_error_nm'])} nm "
         f"(FOV {s['fov_nm']:.0f} nm, pixel {s['pixel_nm']:.2f} nm)"),
        (f"- Landmarks verified: {s['landmarks_verified']}; fitted rotation {_n(s['fitted_rotation_deg'], 2)} deg "
         f"(true {s['true_rotation_deg']} deg)"),
        "",
        "| site | status | true error (nm) | reported residual (nm) | ±sigma (nm) | ncc | second peak |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["rows"]:
        flag = " ⚠" if r["confidently_wrong"] else ""
        lines.append(f"| {r['site']}{' (' + r['kind'] + ')' if r['kind'] != 'site' else ''} | {r['status']}{flag} | "
                     f"{r['true_error_nm']:.1f} | {_n(r['reported_residual_nm'])} | {_n(r['reported_sigma_nm'])} | "
                     f"{_n(r['ncc'], 2)} | {_n(r['ambiguity'], 2)} |")
    return "\n".join(lines) + "\n"


def _n(v: Any, d: int = 1) -> str:
    return "—" if v is None else f"{v:.{d}f}"


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

def synthetic_backend(seed: int, fov: float, imsize: int):
    from asyncroscopy.atlas.synthetic import SyntheticMicroscope

    mic = SyntheticMicroscope(seed=seed, fov_m=fov, imsize=imsize)
    return mic, mic.remount, lambda xy: mic.where_is(xy), mic


def twin_backend(host: str, port: int, instrument: str, scan: str, data: str, mount_attr: str = "specimen_mount"):
    import os

    import tango
    from tiled.client import from_uri

    from asyncroscopy.atlas.tango_backend import TangoMicroscope
    from asyncroscopy.mcp.mcp_server import MCPServer

    os.environ["TANGO_HOST"] = f"{host}:{port}"
    inst = tango.DeviceProxy(instrument)
    inst.set_timeout_millis(60_000)
    scan_proxy = tango.DeviceProxy(scan)
    uri = json.loads(tango.DeviceProxy(data).get_config())["uri"]

    def fetch(key: str):
        return MCPServer._find_first_2d_array(from_uri(uri)[key], prefer_key="image")

    mic = TangoMicroscope(inst, scan_proxy, fetch)
    state = {"rot": 0.0, "shift": np.zeros(2)}

    def remount(rotation_deg: float, shift):
        inst.write_attribute(mount_attr, [float(shift[0]), float(shift[1]), float(rotation_deg)])
        state["rot"], state["shift"] = float(rotation_deg), np.asarray(shift, dtype=float)

    def truth(xy):
        return _rot(state["rot"]) @ np.asarray(xy, dtype=float) + state["shift"]

    return mic, remount, truth, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("synthetic", "twin"), default="synthetic")
    ap.add_argument("--sites", type=int, default=10)
    ap.add_argument("--landmarks", type=int, default=3)
    ap.add_argument("--fov", type=float, default=2e-6, help="site field of view, m")
    ap.add_argument("--imsize", type=int, default=128)
    ap.add_argument("--spread", type=float, default=24e-6, help="width of the area sites are picked from, m")
    ap.add_argument("--rotation", type=float, default=4.0, help="remount rotation, deg")
    ap.add_argument("--shift", type=float, nargs=2, default=(3e-6, -2e-6), help="remount shift, m")
    ap.add_argument("--periodic-sites", type=int, default=2,
                    help="synthetic only: extra sites inside a perfectly periodic patch (should be refused)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    ap.add_argument("--tango-host", default="localhost")
    ap.add_argument("--tango-port", type=int, default=9094)
    ap.add_argument("--instrument", default="asyncroscopy/instrument/default")
    ap.add_argument("--scan", default="asyncroscopy/scan/default")
    ap.add_argument("--data", default="asyncroscopy/data/default")
    a = ap.parse_args(argv)

    extra = []
    if a.backend == "synthetic":
        mic, remount, truth, sim = synthetic_backend(a.seed, a.fov, a.imsize)
        if sim.periodic_patch is not None and a.periodic_sites:
            px, py, size = sim.periodic_patch
            rng = np.random.default_rng(a.seed + 1)
            for i in range(a.periodic_sites):
                extra.append((f"periodic-{i + 1}", np.array([px, py]) + rng.uniform(-size / 4, size / 4, 2)))
    else:
        mic, remount, truth, _ = twin_backend(a.tango_host, a.tango_port, a.instrument, a.scan, a.data)
    report = run(mic, remount, truth, a.out, n_sites=a.sites, n_landmarks=a.landmarks, fov=a.fov, imsize=a.imsize,
                 spread=a.spread, rotation_deg=a.rotation, shift_m=tuple(a.shift), seed=a.seed, extra_sites=extra)
    print()
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
