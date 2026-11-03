"""Stage landing error, readback error and backlash.

Moves away from a reference point and back, approaching from both directions,
and measures where the image really landed by cross-correlation with the
reference image. Prints twin stage properties.
"""

from __future__ import annotations

import numpy as np
import tango

from asyncroscopy.calibration._common import (
    acquire,
    connect,
    parse_args,
    print_yaml,
    shift_m,
)


def measure(microscope: tango.DeviceProxy, fov: float, imsize: int, excursion: float, repeats: int) -> dict[str, float]:
    """Return twin stage properties measured on ``microscope`` around its current position."""
    pixel = fov / imsize
    home = np.array(microscope.get_stage(), dtype=float)
    reference = acquire(microscope)
    landings = {"+x": [], "-x": [], "+y": [], "-y": []}
    readback = []
    for _ in range(repeats):
        # "+x" returns home moving in +x (from the -x side), and so on.
        for label, axis, sign in (("+x", 0, -1), ("-x", 0, 1), ("+y", 1, -1), ("-y", 1, 1)):
            away = home.copy()
            away[axis] += sign * excursion
            microscope.move_stage(list(away))
            microscope.move_stage(list(home))
            readback.append(np.array(microscope.get_stage())[:2] - home[:2])
            # The image shift is how far the sample moved under the beam: the opposite of the stage error.
            landings[label].append(-shift_m(reference, acquire(microscope), pixel))
    means = {label: np.mean(values, axis=0) for label, values in landings.items()}
    all_landings = np.vstack([np.vstack(v) for v in landings.values()])
    centred = np.vstack([np.vstack(landings[k]) - means[k] for k in landings])
    return {
        # Errors are relative to the reference landing, so bias is measured between approaches, not absolutely.
        "stage_move_noise_std": float(centred.std(axis=0).mean() / np.sqrt(2.0)),
        "stage_backlash_x": float(abs(means["+x"][0] - means["-x"][0]) / 2.0),
        "stage_backlash_y": float(abs(means["+y"][1] - means["-y"][1]) / 2.0),
        "stage_reported_noise_std": float(np.std(np.vstack(readback))),
        "# returns_measured": len(all_landings),
    }


def main() -> None:
    args = parse_args(__doc__, lambda p: (p.add_argument("--excursion", type=float, default=2e-6), p.add_argument("--repeats", type=int, default=8)))
    microscope, _scan = connect(args)
    print_yaml(f"stage_repeatability: returns from {args.excursion:.1e} m", measure(microscope, args.fov, args.imsize, args.excursion, args.repeats))


if __name__ == "__main__":
    main()
