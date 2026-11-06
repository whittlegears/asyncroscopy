"""Instrument profiles: what one kind of instrument does inside a map, so one Mapper maps any of them.

The Mapper (asyncroscopy/jobs/mapping/mapper.py) owns everything a map does whatever the instrument:
plan resolution (plan.py), the stage-limit and dry-run checks, ``clock_s`` stamping and resume, the run
file (run_file.py), the budget check, and the stitch, score and revisit phases on the analysis server
(pipeline.py). A profile owns what differs:

* ``setup`` / ``make_safe``: get the instrument ready at the start, and safe between tiles and whenever
  the job stops (blank the beam, retract the tip); ``safe_commands`` really run in a dry run;
* ``check_start``: refuse to start or resume when something checked at reserve no longer holds;
* ``before_move`` / ``move`` / ``after_move``: a probe instrument retracts, moves, then approaches again;
  a beam instrument just moves;
* ``acquire_tile``: one tile's image and any per-tile extras (an EDS spectrum);
* ``acquire_revisit``: one image at a smaller field of view at a found region;
* ``costs``: the budget model (electron dose in e/A^2, a tip force in nN, ...). Every profile also gets
  ``time_s`` from the Mapper's own duration estimate;
* ``needs``: the commands the instrument must offer, checked at reserve;
* ``image_shift_limit_m`` / ``get_image_shift`` / ``set_image_shift``: how live drift tracking moves the
  image without the stage (None: it cannot, and a plan asking for it is refused at reserve);
* ``settings``: the profile's own Mapper device properties (a STEM's SCAN and EDS devices), set in the
  stack YAML under the Mapper's ``properties:`` like the generic ones.

A profile sees the Mapper only through a ``MapContext``: ``proxy`` (use it for every call to the
instrument: a dry run swaps in a recorder), ``instrument``, ``aborting``, ``clock_s``,
``wait_for_stage``, ``read_acquisition`` and ``settings``.

A profile is a small class registered by name::

    from asyncroscopy.jobs.mapping.profile import InstrumentProfile, register_profile

    @register_profile
    class MyProfile(InstrumentProfile):
        name = "my_instrument"
        def acquire_tile(self, job, tile): ...

and chosen with the Mapper's ``instrument_profile`` device property. This package's own profiles are
registered by the modality that owns them, through the ``asyncroscopy.map_profiles`` entry point
(pyproject.toml): ``stem`` (the default) and ``sem`` in instruments/electron_microscope/map_profile.py,
and ``yaml`` (any YAML twin with an image acquisition) in instruments/yaml_sim/map_profile.py. A device
process finds your own profile when ``ASYNCROSCOPY_PLUGINS`` (or the stack YAML's ``plugins:``) names its
module or file (asyncroscopy/plugins.py); installed packages use the same entry point.
docs/Adding_New_Hardware/new_instrument.md ("Map any instrument") walks through one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import tango

from asyncroscopy.data.data_reader import read_acquisition
from asyncroscopy.registry import Registry

# Plan fields every profile takes (the area, the grid, the pipeline, budgets and settings).
GENERIC_PLAN_FIELDS = frozenset({
    "dry_run", "box_m", "size_m", "centre_m", "fov_m", "overlap", "order", "label", "stitch", "find_regions", "revisit", "zoom", "budgets", "drift",
})
GENERIC_PLAN_DEFAULTS = {"overlap": 0.15, "order": "serpentine", "stitch": True, "find_regions": 0}
TIME_BUDGET = "time_s"  # every profile's: the Mapper's estimate of the whole job's duration
SUMMARY_KEYS_SHOWN = 3  # regions, revisits or spectra keys listed in a reply; the rest are on the map and in the run file
STAGE_POLL_S = 0.2  # how often a long stage move is checked for arrival (and the job for an abort)


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

    def refusal(self, budget: float) -> str:
        if self.message:
            return self.message.format(budget=budget, value=self.value, units=self.units)
        parts = f" ({self.why})" if self.why else ""
        advice = f"; {self.advice}" if self.advice else ""
        return f"plan would give up to {self.value:.3g} {self.units}{parts}, above the budget of {budget} {self.units}{advice}"


@dataclass(frozen=True)
class Setting:
    """One of a profile's own Mapper device properties: its type (str, float, int or bool), default and meaning."""

    type: type
    default: Any
    doc: str = ""

    def parse(self, values: list[str]) -> Any:
        """The value from a Tango property (a list of strings); the default when it is not set."""
        if not values:
            return self.default
        text = str(values[0]).strip()
        if self.type is bool:
            return text.lower() in {"1", "true", "yes", "on"}
        return self.type(text)


@dataclass(frozen=True)
class MapSettings:
    """The Mapper settings a profile may read: the generic device properties, and its own (``profile``)."""

    microscope_device_address: str
    stage_device_address: str = ""
    stage_sigma_m: float = 5e-8
    max_drift_m_per_s: float = 5e-11
    stage_timeout_s: float = 600.0
    simulated: bool = False
    profile: Mapping[str, Any] = field(default_factory=dict)


class MapContext:
    """What an instrument profile may use of the Mapper running it.

    ``proxy(address, timeout_s)`` is the Mapper's: in a dry run it returns a recorder. ``read_data(key)``
    reads one acquisition through DATA, and is None-safe for a Mapper without a DATA device.
    """

    def __init__(
        self,
        settings: MapSettings,
        proxy: Callable[..., Any],
        aborting: Callable[[dict], bool],
        data_address: str = "",
    ) -> None:
        self.settings = settings
        self._proxy = proxy
        self._aborting = aborting
        self._data_address = data_address

    def proxy(self, address: str, timeout_s: float | None = None) -> tango.DeviceProxy:
        """A proxy to ``address`` (a recorder in a dry run)."""
        return self._proxy(address, timeout_s) if timeout_s is not None else self._proxy(address)

    def instrument(self) -> tango.DeviceProxy:
        """The instrument the Mapper drives (``microscope_device_address``)."""
        return self.proxy(self.settings.microscope_device_address)

    def aborting(self, job: dict) -> bool:
        """True once the job was asked to stop; a long wait checks it and returns early."""
        return self._aborting(job)

    def data(self) -> tango.DeviceProxy | None:
        """The DATA device, or None when the Mapper has none."""
        return self.proxy(self._data_address) if self._data_address else None

    def read_acquisition(self, key: str) -> dict:
        """One acquisition's arrays by name, {name: (array, attrs)}, read through DATA."""
        return read_acquisition(key, self.data())

    def clock_s(self) -> float:
        """The instrument's clock_s (simulated time on a twin); this computer's time if it reports none."""
        try:
            return float(self.instrument().read_attribute("clock_s").value)
        except (tango.DevFailed, AttributeError, TypeError, ValueError):
            return time.time()

    def wait_for_stage(self, job: dict) -> bool:
        """Wait for a stage move to end (an instrument that returns before arriving says so in stage_moving);
        False if the job was aborted.

        A stage still moving after ``stage_timeout_s`` (a stuck encoder, a lost arrival signal) fails the
        step, so the job stops with the instrument safe and can be resumed, rather than waiting forever.
        """
        instrument = self.instrument()
        timeout = float(self.settings.stage_timeout_s)
        deadline = time.monotonic() + timeout
        while True:
            try:
                moving = bool(instrument.read_attribute("stage_moving").value)
            except tango.DevFailed:  # an instrument whose move returns only on arrival
                return True
            if not moving:
                return True
            if self.aborting(job):
                return False
            if time.monotonic() > deadline:
                raise RuntimeError(f"the stage still reports stage_moving after {timeout:g} s (stage_timeout_s); check the stage, then resume the job")
            time.sleep(STAGE_POLL_S)


class InstrumentProfile:
    """What differs between instruments in a map. Subclass, set ``name``, override what your instrument does differently.

    ``self.ctx`` is the profile's MapContext. Every method that waits must return within a deadline and
    check ``self.ctx.aborting(job)``.
    """

    name = ""
    doc = ""
    # What the Mapper's job_kind and plan_example say with this profile (empty/None: the Mapper's own).
    job_kind = ""
    plan_example: dict | None = None
    # Plan fields this profile takes beyond GENERIC_PLAN_FIELDS, and their defaults.
    plan_fields: frozenset = frozenset()
    plan_defaults: dict = {}
    # Plan fields that set how one image is taken (imsize, dwell_s): a zoom plan, and each of its tiers, takes them too.
    image_fields: frozenset = frozenset()
    # Plan fields that act on tiles only (a STEM's per_tile spectra): refused in a plan without tiles.
    tile_fields: frozenset = frozenset()
    # Plan fields that are one budget under another name: {"dose_budget_e_per_a2": "dose_e_per_a2"}.
    budget_fields: dict = {}
    # The profile's own Mapper device properties, by name (see Setting).
    settings: dict[str, Setting] = {}
    # Setting naming a device -> commands that device must offer (checked at reserve).
    needs: dict = {}
    # Commands that make the instrument safer: they really run in a dry run (everything else is recorded).
    safe_commands: frozenset = frozenset()

    def __init__(self, ctx: MapContext) -> None:
        self.ctx = ctx

    # -- helpers ---------------------------------------------------------
    def instrument(self) -> tango.DeviceProxy:
        """The instrument (``microscope_device_address``), through the context's proxy."""
        return self.ctx.instrument()

    def setting(self, name: str) -> Any:
        """One of this profile's own settings (declared in ``settings``)."""
        return self.ctx.settings.profile[name]

    def needed_commands(self, plan: dict) -> dict[str, list[str]]:
        """{device address: commands this plan needs on it}; from ``needs`` by default.

        A key of ``needs`` names a setting holding a device address; ``microscope_device_address`` is the instrument.
        """
        out: dict[str, list[str]] = {}
        for name, commands in self.needs.items():
            address = self.ctx.settings.microscope_device_address if name == "microscope_device_address" else self.setting(name)
            if address:
                out.setdefault(address, []).extend(commands)
        return out

    def example_plan(self, request: str) -> dict | None:
        """The example plan shown with a refused ``request``; None: the Mapper's own (an area, or a zoom)."""
        return dict(self.plan_example) if self.plan_example is not None else None

    def summary(self, job: dict) -> dict:
        """Extra fields for the job's results (a STEM's spectra keys); none by default."""
        return {}

    # -- plan ------------------------------------------------------------
    def resolve(self, p: dict) -> None:
        """Check and complete the profile's own plan fields in ``p`` (defaults already merged); raise ValueError to refuse."""

    def current_xy(self) -> tuple[float, float]:
        """The stage x and y now (the centre of a ``size_m`` plan without ``centre_m``)."""
        raise NotImplementedError

    def current_fov_m(self) -> float | None:
        """The instrument's field of view now (a revisit-only plan's default fov_m); None when it has none to read."""
        return None

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

    def sample_stage(self) -> str | None:
        """The STAGE device that records which sample is loaded (load_sample/get_sample); None when the instrument has none."""
        return None

    def stage_targets(self, plan: dict, xs: list[tuple[str, float]], ys: list[tuple[str, float]]) -> list[dict]:
        """Limit-check targets (JobDevice.step_targets) for the plan's extreme stage x and y values."""
        return []

    def move_target(self, plan: dict, name: str, arguments: Any) -> dict | None:
        """For a dry run's check: the stage position a recorded command goes to ({"x": ..., "y": ...}, either or both), or None."""
        return None

    def image_shift_target(self, name: str, arguments: Any) -> list[float] | None:
        """For a dry run's check: the image shift [x, y] a recorded command sets, or None when it sets none."""
        return None

    # -- run -------------------------------------------------------------
    def check_start(self, job: dict) -> None:
        """Before anything is set, at start and on every resume: raise RuntimeError when the plan no longer holds."""

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
        return self.ctx.clock_s()

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

    # -- live drift tracking (plan "drift": {"track": true}; live_drift.py) ---------
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


# Every profile: registered by decorator, by ASYNCROSCOPY_PLUGINS, or by the asyncroscopy.map_profiles entry point.
PROFILES: Registry[type[InstrumentProfile]] = Registry("asyncroscopy.map_profiles", kind="instrument profile")
register_profile = PROFILES.register
get_profile = PROFILES.get
profiles = PROFILES.all
