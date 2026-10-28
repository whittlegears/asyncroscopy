"""Contamination and damage under repeated exposure.

Run on a sacrificial area. Images the same field repeatedly and records how
the mean background and the brightest features change with accumulated dose.
Prints twin dose properties.
"""

from __future__ import annotations

import json

import numpy as np

from asyncroscopy.calibration._common import acquire, connect, parse_args, print_yaml

ELECTRON_CHARGE = 1.602176634e-19


def main() -> None:
    args = parse_args(__doc__, lambda p: p.add_argument("--frames", type=int, default=40))
    microscope, _scan = connect(args)
    current_pa = float(json.loads(microscope.get_parameters()).get("beam_current_pa", 50.0))
    dose_per_frame = current_pa * 1e-12 * args.dwell / ELECTRON_CHARGE / (args.fov / args.imsize / 1e-10) ** 2
    backgrounds, peaks = [], []
    for _ in range(args.frames):
        image = acquire(microscope)
        backgrounds.append(np.percentile(image, 10))
        peaks.append(np.percentile(image, 99.5))
    dose = dose_per_frame * np.arange(1, args.frames + 1)
    background = np.array(backgrounds) - backgrounds[0]
    peak = np.array(peaks) / max(peaks[0], 1e-12)
    half = background.max() * 0.63
    contamination_dose = float(dose[np.argmax(background >= half)]) if background.max() > 0 else 0.0
    fading = np.nonzero(peak < 0.9)[0]
    print_yaml(
        f"dose_series: {args.frames} frames, {dose_per_frame:.2e} e/A^2 per frame",
        {
            "dose_tracking": True,
            "contamination_dose": contamination_dose,
            "damage_critical_dose": float(dose[fading[0]]) if len(fading) else 0.0,
            "# background_rise_counts": float(background.max()),
        },
    )


if __name__ == "__main__":
    main()
