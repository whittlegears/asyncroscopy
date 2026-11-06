"""Dependency-free plumbing every other package may use: file locks and sequenced appends
(``fileio``), bind-address rules (``net``) and the shared names (``names``).

The layering rule: ``core`` imports nothing from asyncroscopy; ``contracts``, ``trace``, ``data``,
``safety`` and ``sim`` import only ``core`` and each other; ``instruments`` and ``jobs`` sit above
them, ``mcp`` above those, and ``agents`` on top. Nothing points the other way.
"""
