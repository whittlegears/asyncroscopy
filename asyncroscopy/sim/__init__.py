"""Modality-free simulation primitives: a simulated clock, seeded per-event random streams, a stage with
commanded, true and reported positions, and a sparse dose map.

They import without any instrument code, so every simulator (the electron-microscope twins, YAML twins,
a team's own) can use them.
"""
