"""Stitching measured on the twin against its ground truth (docs/designs/multiscale_maps.md, phase P2).

Takes a serpentine grid of tiles with the twin from a configuration (stage error, drift, scan distortion,
detector drift, support texture: whatever the configuration turns on), stitches them with pyTEMlib exactly
as assemble_map does, and scores four things against what the twin knows really happened:

* placement: each tile's centre as drawn on the map against its true centre, in pixels, after removing the
  common offset (and, separately, after removing the best-fitting affine map, which shows how much of the
  error is the shared scan distortion bending the whole map rather than tiles disagreeing with each other);
* seams: the brightness step across the drawn tile edges compared with the step between neighbouring pixels
  inside tiles (1.0: no visible edge);
* brightness: the mean difference between two neighbouring tiles over the area they share, as drawn;
* unplaced: tiles with no usable match, which the map draws from the stage's guess.

It runs the twin in this process (no Tango database, no data server), so it needs nothing running:

    uv run --no-sync python benchmarks/mosaic_twin.py
    uv run --no-sync python benchmarks/mosaic_twin.py --grid 6 --imsize 256 --out ""    # a quick look

Writes benchmarks/results/mosaic_twin.md (and .json) and exits 1 when a threshold is missed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "asyncroscopy/digitaltwin/benchmark"
# The thresholds of docs/designs/multiscale_maps.md, goal G2.
MISPLACED_PX = 2.0  # a tile further than this from the shared affine fit is misplaced (design G3's "wrong placement")
THRESHOLDS = {
    "placement_median_px": 0.3,
    "placement_p95_px": 0.7,
    "placement_max_px": 2.0,
    "seam_ratio": 1.2,
    "neighbour_brightness_pct": 1.0,
    "unplaced_tiles": 0,
    "misplaced_tiles": 0,
}


def _run_servers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_servers_for_benchmark", ROOT / "startup_scripts" / "run_servers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def twin_properties(config_path: Path, save_dir: Path, sample_size_m: float | None = None, particles: int | None = None) -> dict[str, list[str]]:
    """The configuration's instrument properties for a stand-alone twin: no other devices, no waiting.

    ``sample_size_m`` makes the sample that wide, keeping the configuration's cluster density; ``particles`` sets the
    particle count (default: the configuration's density). The twin stops placing particles after 15 s, so a count
    this machine cannot place in time would make the sample depend on the machine: keep it to tens of thousands."""
    from asyncroscopy.testing import prop_path

    config = _run_servers().load_config(config_path)
    properties = {k: v for k, v in config.instrument.properties.items() if not k.endswith("_device_address")}
    if sample_size_m:
        area = (sample_size_m / float(properties["sample_size_xy"][0])) ** 2
        for name in ("sample_particle_count", "sample_cluster_count"):
            if name in properties:
                properties[name] = [str(int(round(int(properties[name][0]) * area)))]
        properties["sample_size_xy"] = [repr(float(sample_size_m))]
    if particles:
        properties["sample_particle_count"] = [str(int(particles))]
    properties["time_scale"] = ["0.0"]
    properties["acquisition_save_directory"] = [prop_path(save_dir)]
    return properties


def mapper_stage_sigma(config_path: Path) -> float:
    """The stage uncertainty the configuration's Mapper gives the stitcher."""
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for device in (raw.get("devices") or {}).values():
        if isinstance(device, dict) and device.get("class_name") == "Mapper":
            return float((device.get("properties") or {}).get("stage_sigma_m", 5e-8))
    raise SystemExit(f"{config_path} has no Mapper; pass --stage-sigma")


def serpentine(grid: int, step_m: float) -> list[tuple[int, int, float, float]]:
    """(row, col, x, y) for every tile, rows along +y, each row in the opposite direction to the one before."""
    half = (grid - 1) * step_m / 2.0
    out = []
    for row in range(grid):
        cols = range(grid) if row % 2 == 0 else range(grid - 1, -1, -1)
        out += [(row, col, -half + col * step_m, -half + row * step_m) for col in cols]
    return out


def acquire(device: Any, digital_twin_module: ModuleType, plan: list, fov_m: float, imsize: int, dwell_s: float) -> tuple[list, list[dict]]:
    """Take every tile; returns (images, the twin's acquisition records for them)."""
    saved: list = []

    def keep(_device: Any, _data_server: Any, kind: str, _detectors: list, arrays: list, **_kwargs: Any) -> str:
        saved.append(np.asarray(arrays[0]))
        return f"tile-{len(saved)}"

    digital_twin_module.save_acquisition = keep
    device._set_fov(float(fov_m))
    for _row, _col, x, y in plan:
        device._move_stage([x, y, 0.0, 0.0, 0.0])
        device._acquire_scanned_image(int(imsize), float(dwell_s), ["haadf"])
    truth = json.loads(device.export_ground_truth())
    records = [a for a in truth["acquisitions"] if a["kind"] == "image"][-len(plan):]
    return saved, records


def _affine_residual(estimate: np.ndarray, true: np.ndarray) -> np.ndarray:
    """|estimate - best affine map of true| per tile: what is left once a shared stretch, shear and turn are removed.

    The map is fitted to the tiles within MISPLACED_PX of it, refitted until that set stops changing, so a few
    misplaced tiles do not tilt the fit and inflate every other tile's error (they are counted separately)."""
    design = np.hstack([true, np.ones((len(true), 1))])
    keep = np.ones(len(true), bool)
    for _ in range(10):
        if keep.sum() < 3:  # too few tiles near any fit to refine it: keep the plain least-squares fit
            coeffs, *_ = np.linalg.lstsq(design, estimate, rcond=None)
            return np.hypot(*(estimate - design @ coeffs).T)
        coeffs, *_ = np.linalg.lstsq(design[keep], estimate[keep], rcond=None)
        residual = np.hypot(*(estimate - design @ coeffs).T)
        if np.array_equal(residual <= MISPLACED_PX, keep):
            break
        keep = residual <= MISPLACED_PX
    return residual


def placement_errors(positions_px: np.ndarray, records: list[dict], pixel_m: float, imsize: int) -> dict:
    """Tile centres on the canvas (top-left + half a tile) against the true centres, in pixels."""
    true = np.array([[r["center_true_m"][1], r["center_true_m"][0]] for r in records]) / pixel_m  # (row, col)
    drawn = positions_px + (imsize - 1) / 2.0
    diff = drawn - true
    diff -= np.median(diff, axis=0)
    offset_only = np.hypot(*diff.T)
    local = _affine_residual(drawn, true)
    placed = local <= MISPLACED_PX
    if not placed.any():  # stage positions alone: nothing is within 2 px of any fit, so report every tile
        placed = np.ones(len(local), bool)
    return {
        "median_px": float(np.median(offset_only)), "p95_px": float(np.percentile(offset_only, 95)), "max_px": float(offset_only.max()),
        "after_affine_median_px": float(np.median(local[placed])), "after_affine_p95_px": float(np.percentile(local[placed], 95)),
        "after_affine_max_px": float(local[placed].max()), "misplaced": [int(k) for k in np.nonzero(~placed)[0]],
    }


def seam_ratio(canvas: np.ndarray, tops: np.ndarray, imsize: int, margin: int = 8) -> dict:
    """Mean |step| between neighbouring pixels across drawn tile edges, divided by the same inside tiles."""
    h, w = canvas.shape
    edge_cols, edge_rows = np.zeros(w, bool), np.zeros(h, bool)
    for top, left in tops:
        for c in (left, left + imsize):
            if 0 < c < w:
                edge_cols[c] = True
        for r in (top, top + imsize):
            if 0 < r < h:
                edge_rows[r] = True
    covered = canvas > 0
    step_x = np.abs(np.diff(canvas, axis=1))  # step_x[:, c - 1] is between columns c - 1 and c
    step_y = np.abs(np.diff(canvas, axis=0))
    near_col = np.convolve(edge_cols, np.ones(2 * margin + 1), "same") > 0
    near_row = np.convolve(edge_rows, np.ones(2 * margin + 1), "same") > 0
    both_x = covered[:, 1:] & covered[:, :-1]
    both_y = covered[1:, :] & covered[:-1, :]
    at_x = both_x & edge_cols[1:][None, :]
    at_y = both_y & edge_rows[1:][:, None]
    inside_x = both_x & ~near_col[1:][None, :] & ~near_row[:, None]
    inside_y = both_y & ~near_row[1:][:, None] & ~near_col[None, :]
    across = np.concatenate([step_x[at_x], step_y[at_y]])
    inside = np.concatenate([step_x[inside_x], step_y[inside_y]])
    return {"ratio": float(across.mean() / inside.mean()), "edge_pixels": int(across.size)}


def neighbour_brightness(images: list[np.ndarray], tops: np.ndarray, gains: np.ndarray | None, offsets: np.ndarray | None, imsize: int) -> dict:
    """For each pair of tiles drawn overlapping, |mean A - mean B| / mean over the area they share, in percent."""
    values = []
    for a in range(len(images)):
        for b in range(a + 1, len(images)):
            r0, c0 = np.maximum(tops[a], tops[b])
            r1, c1 = np.minimum(tops[a], tops[b]) + imsize
            if r1 - r0 < 16 or c1 - c0 < 16:
                continue
            pa = images[a][r0 - tops[a][0] : r1 - tops[a][0], c0 - tops[a][1] : c1 - tops[a][1]].astype(np.float64)
            pb = images[b][r0 - tops[b][0] : r1 - tops[b][0], c0 - tops[b][1] : c1 - tops[b][1]].astype(np.float64)
            if gains is not None:
                pa, pb = pa * gains[a], pb * gains[b]
            if offsets is not None:
                pa, pb = pa + offsets[a], pb + offsets[b]
            mean = (pa.mean() + pb.mean()) / 2.0
            if mean > 0:
                values.append(100.0 * abs(pa.mean() - pb.mean()) / mean)
    values = np.array(values)
    return {"mean_pct": float(values.mean()), "p95_pct": float(np.percentile(values, 95)), "pairs": int(values.size)}


def run(args: argparse.Namespace) -> dict:
    import tango

    from asyncroscopy.instruments.electron_microscope import digital_twin as digital_twin_module
    from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
    from asyncroscopy.testing import instrument_context
    from pyTEMlib.image_tools.image_mosaic import Tile, register_tiles, stitch_to_arrays

    step = args.fov * (1.0 - args.overlap)
    plan = serpentine(args.grid, step)
    pixel = args.fov / args.imsize
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as save_dir:
        span = (args.grid - 1) * step + args.fov
        properties = twin_properties(args.config, Path(save_dir), args.sample_size or None, args.particles or None)
        for setting in args.set:
            name, _, value = setting.partition("=")
            properties[name.strip()] = [value.strip()]
        if span > float(properties["sample_size_xy"][0]):
            raise SystemExit(f"the grid spans {span * 1e6:.1f} um, more than the {float(properties['sample_size_xy'][0]) * 1e6:.1f} um sample; pass --sample-size")
        with instrument_context([{"class": DigitalTwin, "devices": [{"name": DEVICE, "properties": properties}]}], startup_timeout_s=120.0) as ctx:
            tango.DeviceProxy(ctx.get_device_access(DEVICE)).ping()
            device = tango.Util.instance().get_device_by_name(DEVICE)
            device._blanked = False
            started = time.perf_counter()
            images, records = acquire(device, digital_twin_module, plan, args.fov, args.imsize, args.dwell)
            acquired_s = time.perf_counter() - started
    tiles = [
        Tile(image=images[k], stage_xy_m=(r["reported_m"][0], r["reported_m"][1]), pixel_size_m=pixel, index=k, acquired_s=r["t_start_s"])
        for k, r in enumerate(records)
    ]
    started = time.perf_counter()
    calibration = None
    if args.calibration:
        calibration = np.array(json.loads(Path(args.calibration).read_text(encoding="utf-8"))["scan_linear"]["estimate"])
    placement = register_tiles(tiles, stage_sigma_m=args.stage_sigma, max_drift_m_per_s=args.max_drift, scan_linear=calibration)
    levels, placement = stitch_to_arrays(tiles, placement)
    stitched_s = time.perf_counter() - started
    tops = np.round(placement.positions_px).astype(int)  # where tile edges fall, to the pixel, for the seam check
    fallback = placement.fallback if placement.fallback is not None else np.zeros(len(tiles), bool)
    detector = [r.get("detector", {}).get("gain") for r in records]
    return {
        "config": str(Path(args.config).relative_to(ROOT)) if Path(args.config).is_absolute() else str(args.config),
        "sample_size_m": float(properties["sample_size_xy"][0]), "particles": int(properties["sample_particle_count"][0]),
        "grid": args.grid, "imsize": args.imsize, "fov_m": args.fov, "overlap": args.overlap, "dwell_s": args.dwell,
        "stage_sigma_m": args.stage_sigma, "tiles": len(tiles), "pixel_m": pixel,
        "overrides": list(args.set), "calibration": args.calibration or None, "acquire_s": round(acquired_s, 1), "stitch_s": round(stitched_s, 1),
        "stage_only": placement_errors(placement.stage_positions_px, records, pixel, args.imsize),
        "stitched": placement_errors(placement.positions_px, records, pixel, args.imsize),
        "seams": seam_ratio(levels[0].astype(np.float64), tops, args.imsize),
        "brightness": neighbour_brightness(images, tops, placement.gains, placement.offsets, args.imsize),
        "scan_linear": {"applied": placement.linear.tolist(), "estimate": None if placement.linear_estimate is None else placement.linear_estimate.tolist(),
                        "stderr": None if placement.linear_stderr is None else placement.linear_stderr.tolist()},
        "gain_range": None if placement.gains is None else [float(placement.gains.min()), float(placement.gains.max())],
        "unplaced_tiles": int(fallback.sum()),
        "misplaced_tiles": [k for k in placement_errors(placement.positions_px, records, pixel, args.imsize)["misplaced"] if not fallback[k]],
        "detector_gain_range": [min(g for g in detector if g is not None), max(g for g in detector if g is not None)] if any(g is not None for g in detector) else None,
        "matches_used": int(sum(1 for m in placement.matches if m.used)),
    }


def verdict(result: dict) -> dict[str, bool]:
    drawn = result["stitched"]
    return {
        "placement_median_px": drawn["after_affine_median_px"] <= THRESHOLDS["placement_median_px"],
        "placement_p95_px": drawn["after_affine_p95_px"] <= THRESHOLDS["placement_p95_px"],
        "placement_max_px": drawn["after_affine_max_px"] <= THRESHOLDS["placement_max_px"],
        "misplaced_tiles": len(result["misplaced_tiles"]) <= THRESHOLDS["misplaced_tiles"],
        "seam_ratio": result["seams"]["ratio"] <= THRESHOLDS["seam_ratio"],
        "neighbour_brightness_pct": result["brightness"]["mean_pct"] <= THRESHOLDS["neighbour_brightness_pct"],
        "unplaced_tiles": result["unplaced_tiles"] <= THRESHOLDS["unplaced_tiles"],
    }


def report(result: dict, passed: dict[str, bool]) -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rows = []
    for name in ("stage_only", "stitched"):
        e = result[name]
        rows.append(f"| {name.replace('_', ' ')} | {e['median_px']:.2f} / {e['p95_px']:.2f} / {e['max_px']:.2f} | "
                    f"{e['after_affine_median_px']:.2f} / {e['after_affine_p95_px']:.2f} / {e['after_affine_max_px']:.2f} |")
    checks = "\n".join(f"| {k} | {THRESHOLDS[k]} | {'pass' if ok else 'FAIL'} |" for k, ok in passed.items())
    return f"""# Stitching on the twin

`benchmarks/mosaic_twin.py`, asyncroscopy {commit}, {time.strftime('%Y-%m-%d')}.

{result['tiles']} tiles ({result['grid']} x {result['grid']}, serpentine) of {result['imsize']} pixels, field {result['fov_m'] * 1e9:.0f} nm
({result['pixel_m'] * 1e9:.2f} nm pixels), {result['overlap'] * 100:.0f} % overlap, dwell {result['dwell_s'] * 1e6:.1f} us, from
`{result['config']}` (a {result['sample_size_m'] * 1e6:.0f} um sample of {result['particles']} particles; its film texture
carries registration between particles), twin settings changed: {result['overrides'] or 'none'}, scan calibration
from: {result['calibration'] or 'this map'}; stitched with stage_sigma_m {result['stage_sigma_m'] * 1e9:.0f} nm. Detector gain over the run:
{result['detector_gain_range']}. Taking the tiles took {result['acquire_s']} s, stitching {result['stitch_s']} s;
{result['matches_used']} overlap matches were used.

## Placement against the twin's truth (pixels: median / 95th percentile / worst)

| Positions | After removing the common offset (all tiles) | After removing the best affine map (tiles within 2 px of it) |
| :-- | :-- | :-- |
{chr(10).join(rows)}

"Stitched" is where the tiles are drawn on the map: pyTEMlib resamples each tile at its sub-pixel position. The
scan distortion the stitcher removed: {result['scan_linear']['applied']}
(measured {result['scan_linear']['estimate']}); tile gains it applied range over {result['gain_range']}.
The difference between the two columns is the shared scan distortion (rotation, shear, field-of-view scale)
bending the whole map; what is left after the affine fit is tiles disagreeing with each other. Placement is judged
on the second column (the shared distortion is a calibration, measured on a large map or given): tiles further than
2 px from the affine fit are misplaced, and counted, not averaged in. Misplaced or unplaced: {result['stitched']['misplaced']}.

## Seams and brightness

* Step across drawn tile edges / step inside tiles: {result['seams']['ratio']:.3f} ({result['seams']['edge_pixels']} edge pixels).
* Brightness difference between neighbouring tiles over their shared area: mean {result['brightness']['mean_pct']:.2f} %,
  95th percentile {result['brightness']['p95_pct']:.2f} % ({result['brightness']['pairs']} pairs).
* Tiles with no usable match (drawn from the stage's guess): {result['unplaced_tiles']}.
* Tiles placed by a match but more than 2 px from where they belong (wrong placements): {result['misplaced_tiles']}.

## Against the thresholds (design goal G2)

| Check | Threshold | Result |
| :-- | :-- | :-- |
{checks}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "digital_twin_large_map.yaml")
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--imsize", type=int, default=512)
    parser.add_argument("--fov", type=float, default=2.4e-6, help="field of view (m): the overlap must exceed the stage's error (3 sigma plus backlash)")
    parser.add_argument("--sample-size", type=float, default=36e-6, help="sample width (m); 0 keeps the configuration's")
    parser.add_argument("--particles", type=int, default=20000, help="particles on the sample; 0: the configuration's density")
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--dwell", type=float, default=1e-6)
    parser.add_argument("--stage-sigma", type=float, default=0.0, help="what the Mapper tells the stitcher (m); 0: the configuration's Mapper stage_sigma_m")
    parser.add_argument("--max-drift", type=float, default=5e-11, help="most the sample drifts (m/s)")
    parser.add_argument("--calibration", default="", help="a previous result's .json: use the scan distortion that (large) map measured")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a twin property, e.g. --set scan_flyback_compression=0")
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "mosaic_twin.md"), help='report path; "" prints only')
    args = parser.parse_args(argv)
    if not args.stage_sigma:
        args.stage_sigma = mapper_stage_sigma(args.config)
    result = run(args)
    passed = verdict(result)
    text = report(result, passed)
    print(text)
    if args.out:
        out = Path(args.out)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps({**result, "passed": passed}, indent=2), encoding="utf-8")
    return 0 if all(passed.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
