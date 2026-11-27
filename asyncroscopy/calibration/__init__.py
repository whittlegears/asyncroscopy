"""Measure an instrument so the digital twin can be tuned to match it.

Each script drives any asyncroscopy microscope (real or twin) through Tango
and prints a YAML block of twin properties. Run them on the instrument with
``--profile configs/profiles/<instrument>.yaml`` and they also write what they
measured into that instrument profile (``profile.py``), which becomes a twin
config with ``python -m asyncroscopy.calibration.profile``. ``trace_logger.py``
records the slow changes no short script can see (hours and days).

    uv run python -m asyncroscopy.calibration.stage_repeatability --device asyncroscopy/instrument/default
    uv run python -m asyncroscopy.calibration.drift_series --device asyncroscopy/instrument/default
    uv run python -m asyncroscopy.calibration.dose_series --device asyncroscopy/instrument/default
"""
