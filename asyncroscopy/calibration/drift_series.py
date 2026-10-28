"""Specimen drift: settling after a move, then steady thermal drift.

Moves the stage once, then images the same area repeatedly and tracks the
feature by cross-correlation. Fits an exponential settling term plus a
linear drift. Prints twin drift properties.
"""

from __future__ import annotations

import time

import numpy as np

from asyncroscopy.calibration._common import (
    acquire,
    connect,
    parse_args,
    print_yaml,
    shift_m,
)


def main() -> None:
    args = parse_args(__doc__, lambda p: (p.add_argument("--frames", type=int, default=30), p.add_argument("--interval", type=float, default=20.0)))
    microscope, _scan = connect(args)
    pixel = args.fov / args.imsize
    position = list(microscope.get_stage())
    position[0] += 1e-6
    microscope.move_stage(position)
    start = time.monotonic()
    reference = acquire(microscope)
    times, offsets = [0.0], [np.zeros(2)]
    for _ in range(args.frames - 1):
        time.sleep(args.interval)
        times.append(time.monotonic() - start)
        offsets.append(shift_m(reference, acquire(microscope), pixel))
    t = np.array(times)
    d = np.linalg.norm(np.vstack(offsets), axis=1)
    late = t > t.max() / 2
    rate = float(np.polyfit(t[late], d[late], 1)[0]) if late.sum() > 2 else 0.0
    early_excess = d - rate * t
    tau = float(t[np.argmax(early_excess >= 0.63 * early_excess.max())]) if early_excess.max() > 0 else 30.0
    print_yaml(
        f"drift_series: {args.frames} frames over {t.max():.0f} s",
        {
            "drift_rate_per_min": rate * 60.0,
            "stage_settle_tau": max(tau, 1.0),
            "stage_settle_drift": float(early_excess.max() / max(tau, 1.0)),
        },
    )


if __name__ == "__main__":
    main()
