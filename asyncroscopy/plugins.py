"""Plug-ins: a team's own profiles, drift trackers and strategies, imported in every device process.

A class registered with ``@register_profile`` (or ``@register_tracker``, ``@register_strategy``) only
exists once its module has been imported. The launcher runs each device in its own process, and that
process imports only the device's own module, so a team's decorator never runs there and the Mapper
answers "unknown instrument profile". ``load_plugins`` closes the gap: it imports every module or file
named in ``ASYNCROSCOPY_PLUGINS``, and the registries call it before saying a name is unknown::

    ASYNCROSCOPY_PLUGINS=my_lab.profiles,C:/lab/afm_profile.py

Entries are separated by commas or ``os.pathsep`` (``;`` on Windows, ``:`` elsewhere). An entry ending in
``.py`` (or naming an existing file) is imported from that file; anything else is a module name the
servers' Python can import. The stack YAML's top-level ``plugins:`` list reaches every device through the
same variable (startup_scripts/run_servers.py). Installed packages can use entry points instead
(``asyncroscopy.map_profiles``, ``asyncroscopy.drift_trackers``, ``asyncroscopy.focus_strategies``).

Each entry is imported once per process. One that fails to import is remembered with its error, which
the registries add to their "unknown ..." message, so a typo in a plug-in reads as that typo.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

ENV_VAR = "ASYNCROSCOPY_PLUGINS"

_loaded: set[str] = set()
_errors: dict[str, str] = {}


def plugin_specs(value: str | None = None) -> list[str]:
    """The entries of ``value`` (default: the ``ASYNCROSCOPY_PLUGINS`` variable), in order, without blanks or repeats."""
    text = os.environ.get(ENV_VAR, "") if value is None else value
    parts = re.split("[,%s]" % re.escape(os.pathsep), text)
    seen: list[str] = []
    for part in (p.strip() for p in parts):
        if part and part not in seen:
            seen.append(part)
    return seen


def is_file_spec(spec: str) -> bool:
    """True when ``spec`` names a Python file rather than a module."""
    return spec.endswith(".py") or "/" in spec or "\\" in spec or Path(spec).is_file()


def _import_file(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file {path}")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"\W", "_", path.stem)
    name = f"asyncroscopy_plugin_{stem}_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look the module up while it runs
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_plugins(specs: list[str] | str | None = None) -> dict[str, str]:
    """Import every plug-in not yet imported in this process; returns {entry: error} for those that failed.

    ``specs`` defaults to ``ASYNCROSCOPY_PLUGINS``. Safe to call often: an entry is tried once, and the
    variable is read again on each call, so one set later in the process is still picked up.
    """
    entries = plugin_specs(specs) if specs is None or isinstance(specs, str) else list(specs)
    for spec in entries:
        if spec in _loaded:
            continue
        _loaded.add(spec)
        try:
            if is_file_spec(spec):
                _import_file(Path(spec))
            else:
                importlib.import_module(spec)
        except Exception as exc:  # a broken plug-in must not take the device down; it is reported instead
            _errors[spec] = f"{type(exc).__name__}: {exc}"
            print(f"[asyncroscopy.plugins] could not import {spec!r}: {_errors[spec]}", file=sys.stderr)
    return {spec: _errors[spec] for spec in entries if spec in _errors}


def plugin_errors() -> dict[str, str]:
    """{entry: error} for every plug-in that failed to import in this process."""
    return dict(_errors)


def unknown_hint() -> str:
    """Text for an "unknown name" error: what failed to import, or how to make a plug-in visible."""
    if _errors:
        return "; plug-ins that failed to import: " + ", ".join(f"{k} ({v})" for k, v in _errors.items())
    if not plugin_specs():
        return f"; a class defined in your own module is found only if {ENV_VAR} (or the stack YAML's plugins:) names it"
    return ""
