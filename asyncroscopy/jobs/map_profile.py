"""Instrument profiles: what one kind of instrument does inside a map, so one Mapper maps any of them.

The Mapper (asyncroscopy/instruments/electron_microscope/mapper.py, kept there so
configs and tools keep their names) owns everything a map does whatever the
instrument: plan resolution (area, grid and overlap, centre and size, revisit-only
plans), the stage-limit and dry-run checks, ``clock_s`` stamping and resume, the
run file, the budget check, and the stitch, score and revisit phases on the
analysis server. A profile owns what differs:

* ``setup`` / ``make_safe``: get the instrument ready at the start, and safe
  between tiles and whenever the job stops (blank the beam, retract the tip);
* ``before_move`` / ``move`` / ``after_move``: a probe instrument retracts,
  moves, then approaches again; a beam instrument just moves;
* ``acquire_tile``: one tile's image and any per-tile extras (an EDS spectrum);
* ``acquire_revisit``: one image at a smaller field of view at a found region;
* ``costs``: the budget model (electron dose in e/A^2, a tip force in nN, ...).
  Every profile also gets ``time_s`` from the Mapper's own duration estimate;
* ``needs``: the commands the instrument must offer, checked at reserve;
* ``image_shift_limit_m`` / ``get_image_shift`` / ``set_image_shift``: how live
  drift tracking moves the image without the stage (None: it cannot, and a plan
  asking for it is refused at reserve).

A profile is a small class registered by name, like the Tuner's strategies::

    from asyncroscopy.jobs.map_profile import InstrumentProfile, register_profile

    @register_profile
    class MyProfile(InstrumentProfile):
        name = "my_instrument"
        def acquire_tile(self, job, tile): ...

and chosen with the Mapper's ``instrument_profile`` device property. Built in:
``stem`` (the default), ``sem`` and ``yaml`` (any YAML twin with an image
acquisition). A device process finds your own profile when ``ASYNCROSCOPY_PLUGINS`` (or the
stack YAML's ``plugins:``) names its module or file (asyncroscopy/plugins.py); installed packages
can use the ``asyncroscopy.map_profiles`` entry point.
docs/Adding_New_Hardware/new_instrument.md ("Map any instrument") walks through one.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from asyncroscopy.plugins import load_plugins, unknown_hint

# Plan fields every profile takes (the area, the grid, the pipeline, budgets and settings).
GENERIC_PLAN_FIELDS = frozenset({
    "dry_run", "box_m", "size_m", "centre_m", "fov_m", "overlap", "order", "label", "stitch", "find_regions", "revisit", "budgets", "drift",
})
GENERIC_PLAN_DEFAULTS = {"overlap": 0.15, "order": "serpentine", "stitch": True, "find_regions": 0}
TIME_BUDGET = "time_s"  # every profile's: the Mapper's estimate of the whole job's duration


@dataclass
class Cost:
    """What a plan spends of one budget, worst case: compared with ``plan["budgets"][name]`` at reserve.

    ``message`` (formatted with ``budget``) replaces the default refusal text when given.
    """

    value: float
    units: str
    why: str = ""
    advice: str = ""
    message: str | None = None

    def refusal(self, budget) -> str:
        if self.message:
            return self.message.format(budget=budget, value=self.value, units=self.units)
        parts = f" ({self.why})" if self.why else ""
        advice = f"; {self.advice}" if self.advice else ""
        return f"plan would give up to {self.value:.3g} {self.units}{parts}, above the budget of {budget} {self.units}{advice}"


class InstrumentProfile:
    """What differs between instruments in a map. Subclass, set ``name``, override what your instrument does differently.

    ``self.mapper`` is the Mapper device: ``proxy(address)`` (use it for every call to the
    instrument: a dry run swaps in a recorder), ``aborting(job)``, and its device
    properties (``microscope_device_address`` is the instrument to drive, whatever it is).
    Every method that waits must return within a deadline and check ``aborting``.
    """

    name = ""
    doc = ""
    # What the Mapper's job_kind and plan_example say with this profile (None/empty: the STEM wording and example).
    job_kind = ""
    plan_example: dict | None = None
    # Plan fields this profile takes beyond GENERIC_PLAN_FIELDS, and their defaults.
    plan_fields: frozenset = frozenset()
    plan_defaults: dict = {}
    # Mapper device property naming a device -> commands that device must offer (checked at reserve).
    needs: dict = {}

    def __init__(self, mapper):
        self.mapper = mapper

    # -- helpers ---------------------------------------------------------
    def instrument(self):
        """The instrument (``microscope_device_address``), through the Mapper's proxy."""
        return self.mapper.proxy(self.mapper.microscope_device_address)

    def needed_commands(self, plan: dict) -> dict[str, list[str]]:
        """{device address: commands this plan needs on it}; from ``needs`` by default."""
        out: dict[str, list[str]] = {}
        for prop, commands in self.needs.items():
            address = getattr(self.mapper, prop, "")
            if address:
                out.setdefault(address, []).extend(commands)
        return out

    # -- plan ------------------------------------------------------------
    def resolve(self, p: dict) -> None:
        """Check and complete the profile's own plan fields in ``p`` (defaults already merged); raise ValueError to refuse."""

    def current_xy(self) -> tuple[float, float]:
        """The stage x and y now (the centre of a ``size_m`` plan without ``centre_m``)."""
        raise NotImplementedError

    def tile_size_m(self, p: dict) -> tuple[float, float]:
        """A tile's width and height in metres; square tiles of ``fov_m`` by default."""
        return float(p["fov_m"]), float(p["fov_m"])

    def resolve_revisit(self, p: dict, revisit: dict) -> dict:
        """The revisit settings (at least ``fov_m``) from the plan's ``revisit`` object."""
        return {"fov_m": float(revisit["fov_m"])}

    def tile_seconds(self, p: dict) -> float:
        """One tile's acquisition time, without the stage move (the Mapper adds estimate_move_s)."""
        return 0.0

    def revisit_seconds(self, revisit: dict) -> float:
        """One revisit image's acquisition time, without the move."""
        return 0.0

    def costs(self, p: dict, grid: dict, revisit: dict | None, revisits: int, budgets: dict) -> tuple[dict[str, Cost], dict]:
        """({budget name: Cost}, extra estimate fields) for a tiled plan; ``budgets`` are the plan's limits."""
        return {}, {}

    def revisit_only_costs(self, p: dict, revisit: dict, count: int, budgets: dict) -> tuple[dict[str, Cost], dict]:
        """The same for a plan that only takes ``count`` revisit images."""
        return {}, {}

    def stage_targets(self, plan: dict, xs: list[tuple[str, float]], ys: list[tuple[str, float]]) -> list[dict]:
        """Limit-check targets (JobDevice.step_targets) for the plan's extreme stage x and y values."""
        return []

    def move_target(self, plan: dict, name: str, arguments) -> dict | None:
        """For a dry run's check: the stage position a recorded command goes to ({"x": ..., "y": ...}, either or both), or None."""
        return None

    # -- run -------------------------------------------------------------
    def setup(self, job: dict) -> None:
        """Before the first step (and on a resume): instrument settings for the plan; end safe."""

    def make_safe(self, job: dict) -> None:
        """Between tiles, and when the job finishes, fails or is aborted. Must never raise for an instrument that is already safe."""

    def before_move(self, job: dict) -> None:
        """Just before every stage move (a probe retracts)."""

    def after_move(self, job: dict) -> None:
        """Just after every stage move arrives (a probe approaches)."""

    def move(self, job: dict, x: float, y: float) -> bool:
        """Move the stage to (x, y); False if the job was aborted during the move."""
        raise NotImplementedError

    def stage_reported(self, job: dict) -> list[float]:
        """Where the stage says it is now (at least [x, y], up to 5 axes)."""
        raise NotImplementedError

    def clock_s(self) -> float:
        """The instrument's clock (simulated time on a twin); tiles and revisits are timed with it."""
        from asyncroscopy.instruments.electron_microscope.mapper import Mapper

        return Mapper._clock_s(self.instrument())

    def acquire_tile(self, job: dict, tile: dict) -> dict | None:
        """Acquire one tile at the current position; None if aborted part way (a resume redoes the tile).

        Returns {"image_key": data key, "acquired": clock_s at the middle of the tile, and optionally
        "spectrum_key" plus "spectrum" ({name: (array, attrs)}, when not a key the Mapper reads)}.
        End with the instrument safe, even on an error.
        """
        raise NotImplementedError

    def acquire_revisit(self, job: dict, revisit: dict) -> str:
        """One image at ``revisit["fov_m"]`` at the current position; restore the tile settings; end safe. Returns its data key."""
        raise NotImplementedError

    # -- live drift tracking (plan "drift": {"track": true}; asyncroscopy/jobs/live_drift.py) ---------
    # A profile that can move the image without moving the stage (a beam's image or beam shift) says
    # how far (``image_shift_limit_m``), which commands it needs, and reads and writes the shift. One
    # that cannot leaves ``image_shift_limit_m`` None, and a plan asking for tracking is refused at reserve.
    image_shift_limit_m: float | None = None
    image_shift_commands: tuple = ()

    def get_image_shift(self, job: dict) -> list[float]:
        """The image shift [x, y] in metres now."""
        raise NotImplementedError

    def set_image_shift(self, job: dict, xy: list[float]) -> None:
        """Set the image shift [x, y] in metres (the tracker has already bounded it); raise if the instrument refuses."""
        raise NotImplementedError


PROFILES: dict[str, type[InstrumentProfile]] = {}
# Built-in profiles, imported the first time they are asked for (so this module imports no instrument code).
BUILT_IN = {
    "stem": "asyncroscopy.instruments.electron_microscope.map_profiles",
    "sem": "asyncroscopy.instruments.electron_microscope.map_profiles",
    "yaml": "asyncroscopy.instruments.yaml_sim.map_profile",
}
_entry_points_loaded = False


def register_profile(cls: type[InstrumentProfile]) -> type[InstrumentProfile]:
    """Make ``cls`` available as ``instrument_profile = cls.name``; usable as a class decorator."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name")
    PROFILES[cls.name] = cls
    return cls


def profiles() -> dict[str, type[InstrumentProfile]]:
    """Every profile: built in, registered, named in ``ASYNCROSCOPY_PLUGINS`` (asyncroscopy/plugins.py),
    and installed with the ``asyncroscopy.map_profiles`` entry point."""
    global _entry_points_loaded
    for module in set(BUILT_IN.values()):
        importlib.import_module(module)
    load_plugins()
    if not _entry_points_loaded:
        _entry_points_loaded = True
        from importlib.metadata import entry_points

        for point in entry_points(group="asyncroscopy.map_profiles"):
            try:
                register_profile(point.load())
            except Exception:  # a broken plug-in must not take the Mapper down
                pass
    return PROFILES


def get_profile(name: str) -> type[InstrumentProfile]:
    """The profile class called ``name``; ValueError listing the known ones otherwise."""
    if name not in PROFILES and name in BUILT_IN:
        importlib.import_module(BUILT_IN[name])
    if name not in PROFILES:
        known = profiles()
        if name not in known:
            raise ValueError(f"unknown instrument profile {name!r}; registered: {sorted(known)}{unknown_hint()}")
    return PROFILES[name]
