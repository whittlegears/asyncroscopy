"""Placing finer images on coarser ones, measured on the twin (docs/designs/multiscale_maps.md, phase P3).

One survey tile (tier 1) is taken; inside it, children at a finer tier (tier 2, 16x) are aimed by image shift
at points spread over it; inside each child, one grandchild at the finest tier (tier 3, 8x more, atomic
columns) is aimed the same way. Every image is placed on its parent by pyTEMlib's ``register_child``, from a
prediction made with the commanded image shift alone, and scored against where the twin knows it really
landed: the true centre of the child, located in the parent's own pixels through the parent frame's real
pixel-to-sample coordinates (which carry its scan distortion and drift).

Scores (design goal G3): placement error in parent pixels (95th percentile <= 0.5), the share placed (>= 90 %),
wrong placements (placed but more than 2 parent pixels off: zero), and where a grandchild's centre lands in the
tier-1 survey after both placements, against the truth, in survey pixels (no more than the two steps' allowances
added up: an image's absolute position is only as good as the coarsest placement in its chain).

It runs the twin in this process, like benchmarks/mosaic_twin.py:

    uv run --no-sync python benchmarks/multiscale_twin.py
    uv run --no-sync python benchmarks/multiscale_twin.py --children 10 --out ""     # a quick look
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mosaic_twin  # noqa: E402  (twin_properties, the in-process twin set-up)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "asyncroscopy/digitaltwin/multiscale"
THRESHOLDS = {"p95_parent_px": 0.5, "placed_share": 0.9, "wrong": 0}
WRONG_PX = 2.0


class FrameRecorder:
    """Keeps each frame's real pixel-to-sample coordinates, which the twin computes and then forgets."""

    def __init__(self, device: Any) -> None:
        self.frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        real = device._scan_frame

        def scan_frame(imsize: int, dwell_time: float, scan_region: Any) -> dict:
            frame = real(imsize, dwell_time, scan_region)
            self.frames[frame["index"]] = (frame["xs"].copy(), frame["ys"].copy())
            return frame

        device._scan_frame = scan_frame


def where_in(frame: tuple[np.ndarray, np.ndarray], point_xy: np.ndarray) -> np.ndarray:
    """(row, col) in a frame's pixels of a sample point, by inverting the frame's (locally affine) pixel map."""
    xs, ys = frame
    rows, cols = np.indices(xs.shape)
    design = np.stack([rows.ravel(), cols.ravel(), np.ones(rows.size)], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, np.stack([xs.ravel(), ys.ravel()], axis=1), rcond=None)
    linear, offset = coef[:2].T, coef[2]  # sample = linear @ (row, col) + offset
    return np.linalg.solve(linear, np.asarray(point_xy, dtype=np.float64) - offset)


def acquire(device: Any, saved: list, fov_m: float, imsize: int, dwell_s: float, shift_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    device._set_fov(float(fov_m))
    device._set_image_shift([float(shift_xy[0]), float(shift_xy[1])])
    device._acquire_scanned_image(int(imsize), float(dwell_s), ["haadf"])
    truth = json.loads(device.export_ground_truth())
    return np.asarray(saved[-1], dtype=np.float64), [a for a in truth["acquisitions"] if a["kind"] == "image"][-1]


def predicted(parent_record: dict, parent_shift: np.ndarray, child_shift: np.ndarray, parent_pixel: float, imsize: int) -> np.ndarray:
    """Where the child's centre should be in the parent's pixels, from the commanded image shifts alone."""
    d = (np.asarray(child_shift) - np.asarray(parent_shift)) / parent_pixel  # (x, y) in parent pixels
    return np.array([(imsize - 1) / 2.0 + d[1], (imsize - 1) / 2.0 + d[0]])


def run(args: argparse.Namespace) -> dict:
    import tango

    from asyncroscopy.instruments.electron_microscope import digital_twin as digital_twin_module
    from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
    from asyncroscopy.testing import instrument_context
    from pyTEMlib.image_tools.register import register_child

    rng = np.random.default_rng(args.seed)
    fovs = [args.fov, args.fov / args.step1, args.fov / args.step1 / args.step2]
    pixels = [f / args.imsize for f in fovs]
    saved: list = []

    def keep(_device: Any, _data_server: Any, kind: str, _detectors: list, arrays: list, **_kwargs: Any) -> str:
        saved.append(np.asarray(arrays[0]))
        return f"image-{len(saved)}"

    digital_twin_module.save_acquisition = keep
    results = {"tier2": [], "tier3": []}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as save_dir:
        properties = mosaic_twin.twin_properties(args.config, Path(save_dir), args.sample_size, args.particles)
        with instrument_context([{"class": DigitalTwin, "devices": [{"name": DEVICE, "properties": properties}]}], startup_timeout_s=120.0) as ctx:
            tango.DeviceProxy(ctx.get_device_access(DEVICE)).ping()
            device = tango.Util.instance().get_device_by_name(DEVICE)
            device._blanked = False
            recorder = FrameRecorder(device)
            zero = np.zeros(2)
            parent, parent_record = acquire(device, saved, fovs[0], args.imsize, args.dwell, zero)
            parent_frame = recorder.frames[parent_record["index"]]
            reach = 0.4 * fovs[0]  # aim children over the middle 80 % of the survey tile
            for _ in range(args.children):
                shift2 = rng.uniform(-reach, reach, 2)
                child, child_record = acquire(device, saved, fovs[1], args.imsize, args.dwell, shift2)
                true2 = where_in(parent_frame, np.array(child_record["center_true_m"]))
                placed2 = register_child(child, pixels[1], parent, pixels[0], predicted(parent_record, zero, shift2, pixels[0], args.imsize), args.window, smooth_px=args.smooth)
                entry2 = {"true": true2.tolist(), "status": placed2.status, "reason": placed2.reason, "score": placed2.score, "ambiguity": placed2.ambiguity}
                if placed2.status == "placed":
                    entry2["error_px"] = float(np.hypot(*(np.array(placed2.translation_px) - true2)))
                results["tier2"].append(entry2)
                # One grandchild inside this child, aimed at a point over its middle 60 %.
                shift3 = shift2 + rng.uniform(-0.3 * fovs[1], 0.3 * fovs[1], 2)
                grandchild, grand_record = acquire(device, saved, fovs[2], args.imsize, args.dwell, shift3)
                child_frame = recorder.frames[child_record["index"]]
                true3 = where_in(child_frame, np.array(grand_record["center_true_m"]))
                placed3 = register_child(grandchild, pixels[2], child, pixels[1], predicted(child_record, shift2, shift3, pixels[1], args.imsize), args.window, smooth_px=args.smooth)
                entry3 = {"true": true3.tolist(), "status": placed3.status, "reason": placed3.reason, "score": placed3.score, "ambiguity": placed3.ambiguity}
                if placed3.status == "placed":
                    entry3["error_px"] = float(np.hypot(*(np.array(placed3.translation_px) - true3)))
                    if placed2.status == "placed":
                        # Chain both placements: the grandchild's centre in the survey, against the truth there.
                        in_child = np.array(placed3.translation_px)
                        via = np.array(placed2.translation_px) + (in_child - (args.imsize - 1) / 2.0) * pixels[1] / pixels[0]
                        truth_survey = where_in(parent_frame, np.array(grand_record["center_true_m"]))
                        entry3["chained_error_survey_px"] = float(np.hypot(*(via - truth_survey)))
                results["tier3"].append(entry3)
    elapsed = time.perf_counter() - started

    def summary(entries: list[dict]) -> dict:
        placed = [e for e in entries if e["status"] == "placed"]
        errors = np.array([e["error_px"] for e in placed]) if placed else np.array([np.nan])
        return {
            "children": len(entries), "placed": len(placed), "placed_share": len(placed) / max(len(entries), 1),
            "median_px": float(np.nanmedian(errors)), "p95_px": float(np.nanpercentile(errors, 95)) if placed else float("nan"),
            "wrong": int(np.sum(errors > WRONG_PX)) if placed else 0,
            "unplaced_reasons": sorted({e["reason"].split(":")[0] for e in entries if e["status"] != "placed"}),
        }

    chained = np.array([e["chained_error_survey_px"] for e in results["tier3"] if "chained_error_survey_px" in e])
    return {
        "config": str(args.config), "imsize": args.imsize, "fovs_m": fovs, "pixels_m": pixels, "steps": [args.step1, args.step2],
        "window_px": args.window, "seconds": round(elapsed, 1), "tier2": summary(results["tier2"]), "tier3": summary(results["tier3"]),
        "chained_survey_px": {"median": float(np.median(chained)) if chained.size else None, "p95": float(np.percentile(chained, 95)) if chained.size else None, "n": int(chained.size)},
        "entries": results,
    }


def verdict(result: dict) -> dict[str, bool]:
    t2, t3, chained = result["tier2"], result["tier3"], result["chained_survey_px"]
    return {
        "tier2_p95_parent_px": t2["p95_px"] <= THRESHOLDS["p95_parent_px"],
        "tier3_p95_parent_px": t3["p95_px"] <= THRESHOLDS["p95_parent_px"],
        "tier2_placed_share": t2["placed_share"] >= THRESHOLDS["placed_share"],
        "tier3_placed_share": t3["placed_share"] >= THRESHOLDS["placed_share"],
        "wrong_placements": t2["wrong"] + t3["wrong"] <= THRESHOLDS["wrong"],
        # Each step may be off by p95_parent_px of its parent's pixels, so a chain by their sum, in survey pixels.
        "atomic_image_in_survey_after_two_placements": chained["median"] is not None
        and chained["p95"] <= THRESHOLDS["p95_parent_px"] * (1.0 + result["pixels_m"][1] / result["pixels_m"][0]),
    }


def report(result: dict, passed: dict[str, bool]) -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    def row(name: str, s: dict) -> str:
        return f"| {name} | {s['children']} | {s['placed']} ({s['placed_share'] * 100:.0f} %) | {s['median_px']:.3f} / {s['p95_px']:.3f} | {s['wrong']} | {'; '.join(s['unplaced_reasons']) or '—'} |"

    checks = "\n".join(f"| {k} | {'pass' if ok else 'FAIL'} |" for k, ok in passed.items())
    c = result["chained_survey_px"]
    return f"""# Multiscale placement on the twin

`benchmarks/multiscale_twin.py`, asyncroscopy {commit}, {time.strftime('%Y-%m-%d')}, from `{result['config']}`.

Tiers: {', '.join(f'{f * 1e9:.2f} nm field ({p * 1e9:.4f} nm pixels)' for f, p in zip(result['fovs_m'], result['pixels_m']))};
steps {result['steps'][0]}x then {result['steps'][1]}x; {result['imsize']}-pixel images; each child predicted from the commanded image
shift and searched within {result['window_px']} parent pixels. {result['seconds']} s in all.

| Step | Children | Placed | Error, parent pixels: median / 95th | Wrong (> {WRONG_PX} px) | Why some were not placed |
| :-- | :-- | :-- | :-- | :-- | :-- |
{row('survey -> tier 2', result['tier2'])}
{row('tier 2 -> tier 3 (atomic)', result['tier3'])}

A grandchild's centre placed in the survey through both placements, against the truth, in survey pixels:
median {c['median']}, 95th percentile {c['p95']} ({c['n']} chained). An image's absolute position is only as good as
the coarsest placement in its chain: 0.2 survey pixels here is about 1 nm, many atomic pixels.

| Check (design goal G3) | Result |
| :-- | :-- |
{checks}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "digital_twin_large_map.yaml")
    parser.add_argument("--children", type=int, default=50)
    parser.add_argument("--imsize", type=int, default=512)
    parser.add_argument("--fov", type=float, default=2.4e-6, help="survey field (m)")
    parser.add_argument("--step1", type=float, default=16.0)
    parser.add_argument("--step2", type=float, default=8.0)
    parser.add_argument("--window", type=float, default=8.0, help="search radius in parent pixels around the prediction")
    parser.add_argument("--dwell", type=float, default=2e-6)
    parser.add_argument("--smooth", type=float, default=1.0, help="smoothing (parent pixels) before matching")
    parser.add_argument("--sample-size", type=float, default=1.2e-5)
    parser.add_argument("--particles", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "multiscale_twin.md"), help='report path; "" prints only')
    args = parser.parse_args(argv)
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
