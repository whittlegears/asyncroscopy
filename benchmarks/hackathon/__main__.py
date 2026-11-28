"""Run controllers on the hackathon task, score them and draw what they saw.

    uv run python -m benchmarks.hackathon                                   # the three baselines on three samples
    uv run python -m benchmarks.hackathon --seeds 7 8 --only kalman
    uv run python -m benchmarks.hackathon --controller my_team.py:controller  # your own, beside the baselines

Prints a table of scores and writes, under ``outputs/hackathon/``, ``scores.json`` and one picture per controller and
sample: the survey as the microscope imaged it, the answered map and the true map.

A controller is a function ``controller(microscope) -> Answer`` (``task.py``). Give it as ``file.py:name``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from benchmarks.hackathon import task
from benchmarks.hackathon.baselines import BASELINES
from benchmarks.hackathon.score import score, truth_maps

OUT = task.ROOT / "outputs" / "hackathon"
SURVEY_SHRINK = 4  # survey shows images at a quarter size


def load_controller(spec: str) -> tuple[str, Callable]:
    """``file.py:name`` as (name, the function)."""
    path, _, name = spec.partition(":")
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return name or "controller", getattr(module, name or "controller")


def survey_picture(microscope: task.Microscope) -> np.ndarray:
    """Every image the controller took, drawn where the stage said it was. Places never imaged stay empty (NaN)."""
    pixel = task.FOV_M / task.IMSIZE * SURVEY_SHRINK
    size = int(round(task.AREA_M / pixel))
    tile = task.IMSIZE // SURVEY_SHRINK
    canvas = np.full((size, size), np.nan, dtype=np.float32)
    for image in microscope.images_taken:
        small = image.pixels.reshape(tile, SURVEY_SHRINK, tile, SURVEY_SHRINK).mean(axis=(1, 3))
        left = int(np.clip(round((image.x_m + task.AREA_M / 2) / pixel - tile / 2), 0, size - tile))
        top = int(np.clip(round((image.y_m + task.AREA_M / 2) / pixel - tile / 2), 0, size - tile))
        canvas[top : top + tile, left : left + tile] = small
    return canvas


def draw(path: Path, title: str, microscope: task.Microscope, answer: task.Answer, truth: dict) -> None:
    """One picture: the survey, the answered map and the true map (holes in black)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_share, true_hole = truth_maps(truth)
    half_um = task.AREA_M / 2 * 1e6
    extent = [-half_um, half_um, -half_um, half_um]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    survey = survey_picture(microscope)
    axes[0].imshow(survey, cmap="gray", origin="lower", extent=extent, vmin=np.nanpercentile(survey, 1), vmax=np.nanpercentile(survey, 99.7))
    axes[0].set_title(f"survey: {len(microscope.images_taken)} images, {microscope.autofocus_runs} autofocus runs, {microscope.used_s:.0f} s")
    axes[0].set_facecolor("#203040")  # never imaged
    colours = plt.get_cmap("viridis").copy()
    colours.set_bad("black")
    for axis, share, hole, name in ((axes[1], answer.share_map, answer.hole_map, f"answered: share of {answer.marker_element}"), (axes[2], true_share, true_hole, f"true: share of {truth['marker_element']}")):
        shown = axis.imshow(np.ma.masked_where(np.asarray(hole, bool), np.asarray(share, float)), cmap=colours, origin="lower", extent=extent, vmin=0, vmax=1)
        axis.set_title(name + " (holes black)")
    figure.colorbar(shown, ax=axes[1:], shrink=0.8, label="marker share of particles")
    for axis in axes:
        axis.set_xlabel("x (um)")
        axis.set_ylabel("y (um)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def run(controllers: dict[str, Callable], seeds: list[int], budget_s: float, pictures: bool = True) -> list[dict]:
    """Every controller on every seed's sample: one row of scores each."""
    rows = []
    with task.twin() as device:
        for seed in seeds:
            for name, controller in controllers.items():
                microscope = task.start(device, seed, budget_s)
                answer = controller(microscope)
                truth = task.ground_truth(device)
                row = {"controller": name, "seed": seed, **score(answer, truth), "images": len(microscope.images_taken), "autofocus_runs": microscope.autofocus_runs, "used_s": round(microscope.used_s, 1)}
                rows.append(row)
                print(f"{name:>10} seed {seed}: {row['score']:5.1f}  (marker {row['marker_points']:.0f}, rule {row['rule_points']:.1f}, map {row['map_points']:.1f}; {row['images']} images, {row['autofocus_runs']} autofocus)", flush=True)
                if pictures:
                    draw(OUT / f"{name}_seed{seed}.png", f"{name}, sample {seed}: {row['score']:.1f} points", microscope, answer, truth)
    return rows


def table(rows: list[dict]) -> str:
    lines = ["| Controller | Score | Marker | Rule | Map | Images | Autofocus runs |", "| :-- | --: | --: | --: | --: | --: | --: |"]
    for name in dict.fromkeys(row["controller"] for row in rows):
        mine = [row for row in rows if row["controller"] == name]
        mean = lambda key: np.mean([row[key] for row in mine])
        lines.append(f"| {name} | {mean('score'):.1f} | {mean('marker_points'):.1f} | {mean('rule_points'):.1f} | {mean('map_points'):.1f} | {mean('images'):.0f} | {mean('autofocus_runs'):.0f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103], help="One sample per seed.")
    parser.add_argument("--only", nargs="+", choices=sorted(BASELINES), help="Run only these baselines.")
    parser.add_argument("--controller", action="append", default=[], metavar="FILE.py:NAME", help="Your own controller; may be given more than once.")
    parser.add_argument("--budget", type=float, default=task.BUDGET_S, help="Simulated seconds each controller may spend.")
    parser.add_argument("--no-pictures", action="store_true")
    args = parser.parse_args()
    controllers = {name: BASELINES[name] for name in (args.only or BASELINES)}
    controllers.update(dict(load_controller(spec) for spec in args.controller))
    rows = run(controllers, args.seeds, args.budget, pictures=not args.no_pictures)
    print(f"\nMean over {len(args.seeds)} samples, budget {args.budget:.0f} s (100 is perfect):\n\n{table(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nscores and pictures are in {OUT.relative_to(task.ROOT)}")


if __name__ == "__main__":
    main()
