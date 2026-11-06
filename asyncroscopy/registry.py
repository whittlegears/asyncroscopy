"""One registry for named plug-in classes: map profiles, tuning strategies, drift trackers, specimens.

A ``Registry`` maps a name to a class. Classes arrive four ways:

- ``@registry.register`` on the class, when its module is imported;
- ``builtins``: a {name: module} table of modules imported the first time the name is asked for;
- ``ASYNCROSCOPY_PLUGINS`` (asyncroscopy/plugins.py), the team's own modules and files;
- the package entry-point ``group``, for installed packages (and for this package's own modalities,
  so a job device never imports instrument code by name).

Every source is loaded at most once per registry. A source that fails to load is remembered with its
error, and the "unknown name" error lists those failures, so a typo in a plug-in reads as that typo
instead of as a missing name.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from asyncroscopy.plugins import PLUGINS, Plugins

T = TypeVar("T", bound=type)


class Registry(Generic[T]):
    """Named classes of one kind, found from decorators, built-in modules, plug-ins and entry points.

    ``kind`` names the thing in error messages ("instrument profile"). ``key`` is the class attribute
    holding a class's registered name (``name`` for most, ``source`` for specimens).
    """

    def __init__(
        self,
        group: str,
        builtins: Mapping[str, str] | None = None,
        *,
        kind: str,
        key: str = "name",
        plugins: Plugins | None = PLUGINS,
    ) -> None:
        self.group = group
        self.builtins: dict[str, str] = dict(builtins or {})
        self.kind = kind
        self.key = key
        self._plugins = plugins
        self._classes: dict[str, T] = {}
        self._entry_points_loaded = False
        self._errors: dict[str, str] = {}

    # -- registering ---------------------------------------------------------
    def register(self, cls: T) -> T:
        """Make ``cls`` available under its ``key`` attribute; usable as a class decorator."""
        name = getattr(cls, self.key, "")
        if not name:
            raise ValueError(f"{cls.__name__} needs a {self.key}")
        self._classes[str(name)] = cls
        return cls

    def unregister(self, name: str) -> None:
        """Forget ``name`` (tests register throwaway classes)."""
        self._classes.pop(name, None)

    # -- looking up ----------------------------------------------------------
    def get(self, name: str) -> T:
        """The class registered as ``name``; ValueError naming the known ones and any load failures otherwise."""
        if name not in self._classes and name in self.builtins:
            self._import(self.builtins[name])
        if name not in self._classes:
            self.load()
        if name not in self._classes:
            raise ValueError(f"unknown {self.kind} {name!r}; registered: {sorted(self._classes)}{self.hint()}")
        return self._classes[name]

    def all(self) -> dict[str, T]:
        """Every class from every source, as a new {name: class} dict."""
        self.load()
        return dict(self._classes)

    def __contains__(self, name: object) -> bool:
        return name in self._classes

    def load(self) -> None:
        """Import the built-in modules, the plug-ins, and (once) the entry points."""
        for module in dict.fromkeys(self.builtins.values()):
            self._import(module)
        if self._plugins is not None:
            self._plugins.load()
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for point in entry_points(group=self.group):
            try:
                self.register(point.load())
            except Exception as exc:  # a broken plug-in must not take the device down; it is reported instead
                self._fail(f"entry point {point.name!r}", exc)

    def errors(self) -> dict[str, str]:
        """{source: error} for every built-in module or entry point that failed to load."""
        return dict(self._errors)

    def hint(self) -> str:
        """Text for an "unknown name" error: what failed to load, or how to make a plug-in visible."""
        text = "".join(f"; {source} failed to load ({error})" for source, error in self._errors.items())
        if self._plugins is not None:
            text += self._plugins.hint()
        return text

    # -- internals -----------------------------------------------------------
    def _import(self, module: str) -> None:
        label = f"built-in module {module!r}"
        if label in self._errors:
            return
        try:
            importlib.import_module(module)
        except Exception as exc:
            self._fail(label, exc)

    def _fail(self, label: str, exc: Exception) -> None:
        self._errors[label] = f"{type(exc).__name__}: {exc}"
        print(f"[asyncroscopy.registry] {self.group}: could not load {label}: {self._errors[label]}", file=sys.stderr)
