"""Plug-ins: a team's own profiles, drift trackers and strategies, imported in every device process.

A class registered with ``@register_profile`` (or ``@register_tracker``, ``@register_strategy``) only
exists once its module has been imported. The launcher runs each device in its own process, and that
process imports only the device's own module, so a team's decorator never runs there and the Mapper
answers "unknown instrument profile". ``PLUGINS.load()`` closes the gap: it imports every module or file
named in ``ASYNCROSCOPY_PLUGINS``, and every ``Registry`` (asyncroscopy/registry.py) calls it before saying
a name is unknown::

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
from collections.abc import Callable
from pathlib import Path
from asyncroscopy import settings

ENV_VAR = "ASYNCROSCOPY_PLUGINS"  # named in messages; asyncroscopy.settings reads it


def plugin_specs(value: str | None = None) -> list[str]:
    """The entries of ``value`` (default: the ``ASYNCROSCOPY_PLUGINS`` variable), in order, without blanks or repeats."""
    text = settings.get().plugins if value is None else value
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


class Plugins:
    """The plug-in modules named by ``ASYNCROSCOPY_PLUGINS`` (or by ``source``), each imported at most once."""

    def __init__(self, source: Callable[[], str] | None = None) -> None:
        self.source = source or (lambda: settings.get().plugins)
        self._tried: set[str] = set()
        self._errors: dict[str, str] = {}

    def specs(self) -> list[str]:
        """The entries named now."""
        return plugin_specs(self.source())

    def load(self, specs: list[str] | str | None = None) -> dict[str, str]:
        """Import every plug-in not yet tried in this process; returns {entry: error} for those that failed.

        ``specs`` defaults to the source, which is read again on each call, so one set
        later in the process is still picked up.
        """
        if specs is None:
            entries = self.specs()
        elif isinstance(specs, str):
            entries = plugin_specs(specs)
        else:
            entries = list(specs)
        for spec in entries:
            if spec in self._tried:
                continue
            self._tried.add(spec)
            try:
                if is_file_spec(spec):
                    _import_file(Path(spec))
                else:
                    importlib.import_module(spec)
            except Exception as exc:  # a broken plug-in must not take the device down; it is reported instead
                self._errors[spec] = f"{type(exc).__name__}: {exc}"
                print(f"[asyncroscopy.plugins] could not import {spec!r}: {self._errors[spec]}", file=sys.stderr)
        return {spec: self._errors[spec] for spec in entries if spec in self._errors}

    def errors(self) -> dict[str, str]:
        """{entry: error} for every plug-in that failed to import."""
        return dict(self._errors)

    def hint(self) -> str:
        """Text for an "unknown name" error: what failed to import, or how to make a plug-in visible."""
        if self._errors:
            return "; plug-ins that failed to import: " + ", ".join(f"{k} ({v})" for k, v in self._errors.items())
        if not self.specs():
            return f"; a class defined in your own module is found only if {ENV_VAR} (or the stack YAML's plugins:) names it"
        return ""


# The process's plug-ins. Every Registry imports through this one object, so each entry is tried once.
PLUGINS = Plugins()
