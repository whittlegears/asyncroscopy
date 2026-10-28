"""Measure an instrument so the digital twin can be tuned to match it.

Each script drives any asyncroscopy microscope (real or twin) through Tango
and prints a YAML block of twin properties. Run them on the instrument, paste
the output into the twin config's ``instrument.properties``.

    uv run python -m asyncroscopy.calibration.stage_repeatability --device asyncroscopy/instrument/default
    uv run python -m asyncroscopy.calibration.drift_series --device asyncroscopy/instrument/default
    uv run python -m asyncroscopy.calibration.dose_series --device asyncroscopy/instrument/default
"""
