"""Mapper: acquire a grid of tiles over a stage box, then stitch and explore it, as one background job.

An agent (or a person, or a script) asks for a map in one call instead of issuing a thousand moves and
acquisitions. The plan (asyncroscopy/contracts/map_plan.schema.json; plan.py) fixes a stage box, field of
view, overlap and, for the instrument, its own fields (a STEM's detectors and per-tile EDS spectra, an
AFM's setpoint). The worker visits tiles in serpentine order (to limit backlash) and never leaves the box.

The Mapper is instrument-agnostic: what differs between a STEM, an SEM and an AFM is an instrument profile
(profile.py), chosen with the ``instrument_profile`` device property: ``stem`` (the default), ``sem``,
``yaml`` (any YAML twin with an image acquisition) or one registered by a plug-in. The profile acquires
each tile, makes the instrument safe between tiles and whenever the job stops (blanks the beam, retracts
the tip), prepares before a move and recovers after it, and says what a plan costs. ``reserve`` refuses a
plan over any budget it gives (``budgets``: electron dose for a beam, time for any instrument, tip force
for an AFM) before anything moves; the plan cannot change after that. The instrument is made safe
whenever the job ends: done, aborted or failed.

The job is a pipeline whose phases are steps of the same job, so abort, approval and progress cover all
of them: one step per tile (computed from the plan's grid when needed, plan.step_at), then stitch, score
and revisit on the analysis server (pipeline.py); a zoom plan has one step per tier. Every tile goes into
one HDF5 run file (run_file.py), kept open and flushed after each tile.

A dry run (plan ``"dry_run": true``) moves nothing: stage moves and acquisitions are recorded in the trace
and checked against the plan box (see JobDevice), the profile's safe commands still run, no image or run
file is made, and the summary says so.

Live drift tracking (plan ``"drift": {"track": true}``, off by default) corrects each tile with a bounded
image shift between the stage move and the acquisition (live_drift.py).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import tango
from tango.server import device_property

from asyncroscopy.contracts import ENVELOPE_VERSION
from asyncroscopy.data.record import current_sample, record_fields
from asyncroscopy.jobs.job_device import JobDevice, done_count
from asyncroscopy.routines.grid import EDGE_TOLERANCE_M, inside_box

from . import live_drift, plan as plans
from .pipeline import DRY_RUN_SKIPPED, Pipeline
from .profile import PROFILES, SUMMARY_KEYS_SHOWN, InstrumentProfile, MapContext, MapSettings, Setting
from .run_file import RunFile, job_trace_events

DEFAULT_STAGE_SIGMA_M = 5e-8  # a typical TEM goniometer read back by encoders, when neither the Mapper nor the STAGE says



class Mapper(JobDevice):
    JOB_KIND = "map acquisition: a grid of images over a stage box, stitched into a zoomable map when it finishes"
    PLAN_EXAMPLE = {"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7}  # a 600 nm square around the current stage position
    PLAN_SCHEMA = "map_plan"  # resolve_plan validates every plan against it
    # A dry run rehearses the moves and skips every acquisition and its analysis (DRY_RUN_SKIPPED).
    SUPPORTS_DRY_RUN = True

    instrument_profile = device_property(
        dtype=str,
        default_value="stem",
        doc="What kind of instrument this Mapper drives (asyncroscopy/jobs/mapping/profile.py): stem (the default), sem, yaml "
        "(a YAML twin with an image acquisition), or a profile registered by a plug-in. A profile may read more of this "
        "Mapper's properties (its own settings, e.g. a STEM's scan_device_address).",
    )
    microscope_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Instrument to drive (a microscope, or any device the instrument profile drives).")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device; when set, the run file is registered with Tiled.")
    stage_device_address = device_property(dtype=str, default_value="asyncroscopy/stage/default", doc="STAGE device whose x/y limits every plan is checked against at reserve; blank: not checked (the instrument still refuses moves outside them).")
    sample_device_address = device_property(dtype=str, default_value="asyncroscopy/sample/default", doc="SAMPLE device that says which sample is loaded; every run file records it and a resume onto another load is refused. Blank: not recorded.")
    output_directory = device_property(dtype=str, default_value="outputs/maps", doc="Where run files are written when there is no DATA device; with one, they go in its save path (which Tiled serves) under maps/.")
    max_tiles = device_property(dtype=int, default_value=4_000_000, doc="Refuse plans with more tiles than this (a 2000 x 2000 grid by default).")
    estimate_move_s = device_property(dtype=float, default_value=2.0, doc="Assumed time per stage move, for estimates only.")
    stage_timeout_s = device_property(dtype=float, default_value=600.0, doc="Longest one stage move may report stage_moving before the job fails (instrument made safe; resumable).")
    analysis_url = device_property(dtype=str, default_value="", doc="Analysis MCP server (e.g. http://127.0.0.1:8003/mcp) that stitches and scores finished maps; blank: no stitching.")
    stage_sigma_m = device_property(
        dtype=float,
        default_value=0.0,
        doc="1-sigma distance between where the stage lands and where it reports; bounds the stitching search. 0 (the default): "
        "the STAGE device's position_sigma_m (a twin's STAGE reports its stage model's), else 50 nm, which suits a typical TEM "
        "goniometer read back by encoders.",
    )
    max_drift_m_per_s = device_property(dtype=float, default_value=5e-11, doc="Most the sample drifts relative to the stage (m/s); bounds the stitching search and the fitted drift. 0: no drift model.")

    def init_device(self) -> None:
        self._map_profile: InstrumentProfile | None = None
        self._runs: dict[str, RunFile] = {}
        self._pipeline_: Pipeline | None = None
        super().init_device()

    # ------------------------------------------------------------------
    # The profile and what it sees
    # ------------------------------------------------------------------
    def _profile_settings(self, declared: dict[str, Setting]) -> dict[str, Any]:
        """The profile's own settings, from this Mapper's device properties in the Tango database (defaults otherwise)."""
        stored: dict = {}
        if declared:
            try:
                stored = tango.Util.instance().get_database().get_device_property(self.get_name(), list(declared))
            except tango.DevFailed as exc:  # no database (a -nodb server): every setting keeps its default
                self.warn_stream(f"profile settings not read from the database ({exc.args[0].desc.strip()[:200]}); using defaults")
        return {name: setting.parse(list(stored.get(name, []))) for name, setting in declared.items()}

    def _stage_sigma_m(self) -> float:
        """stage_sigma_m when set; else the STAGE device's position_sigma_m when it knows it; else DEFAULT_STAGE_SIGMA_M."""
        if float(self.stage_sigma_m) > 0:
            return float(self.stage_sigma_m)
        if self.stage_device_address:
            try:
                reported = float(self.proxy(self.stage_device_address, timeout_s=3.0).read_attribute("position_sigma_m").value)
            except (tango.DevFailed, AttributeError, TypeError, ValueError):
                reported = 0.0
            if reported > 0:
                return reported
        return DEFAULT_STAGE_SIGMA_M

    def profile(self) -> InstrumentProfile:
        """The instrument profile named by ``instrument_profile`` (``stem`` when blank), made once."""
        name = self.instrument_profile or "stem"
        if self._map_profile is None or self._map_profile.name != name:
            cls = PROFILES.get(name)
            settings = MapSettings(
                microscope_device_address=self.microscope_device_address,
                stage_device_address=self.stage_device_address,
                max_drift_m_per_s=float(self.max_drift_m_per_s),
                stage_timeout_s=float(self.stage_timeout_s),
                simulated=self._stack_simulated(),
                profile=self._profile_settings(cls.settings),
            )
            self._map_profile = cls(MapContext(settings, self.proxy, self.aborting, self.data_device_address))
        return self._map_profile

    def safe_commands(self, job: dict) -> frozenset:
        return self.profile().safe_commands(job["resolved_plan"])

    def pipeline(self) -> Pipeline:
        """The analysis phases, over one session to ``analysis_url``."""
        if self._pipeline_ is None or self._pipeline_.url != self.analysis_url:
            self._pipeline_ = Pipeline(self.analysis_url, float(self.max_drift_m_per_s), self.aborting, self._lock, self._save, self.warn_stream)
        return self._pipeline_

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def read_job_kind(self) -> str:
        return self.profile().job_kind or self.JOB_KIND

    def read_plan_example(self) -> str:
        example = self.profile().plan_example
        return json.dumps(example) if example is not None else super().read_plan_example()

    def example_plan(self, request: str) -> dict | None:
        example = self.profile().example_plan(request)
        if example is not None:
            return example
        return dict(plans.ZOOM_EXAMPLE) if '"zoom"' in request else dict(self.PLAN_EXAMPLE)

    def resolve_plan(self, plan: dict) -> dict:
        profile = self.profile()
        env = plans.PlanEnvironment(
            max_tiles=int(self.max_tiles),
            estimate_move_s=float(self.estimate_move_s),
            stage_sigma_m=self._stage_sigma_m(),
            analysis=self.pipeline().call if self.analysis_url else None,
        )
        resolved = plans.resolve(plan, profile, env)
        self._check_needs(profile, resolved)
        return resolved

    def _check_needs(self, profile: InstrumentProfile, plan: dict) -> None:
        """Refuse a plan whose instrument lacks a command the profile needs, at reserve, before anything moves.

        The commands are kept in the plan as ``drives`` ({device address: commands}), so auto mode can tell
        whether it runs every one of them before it starts the job."""
        needs = profile.needed_commands(plan)
        shift_commands = live_drift.needed_commands(plan, profile)
        if plan.get("zoom_only"):
            shift_commands = [*shift_commands, *profile.image_shift_commands, "get_stage"]
        if shift_commands:
            needs.setdefault(self.microscope_device_address, []).extend(shift_commands)
        for address, commands in needs.items():
            try:
                offered = {str(n).lower() for n in self.proxy(address, timeout_s=3.0).get_command_list()}
            except Exception as exc:  # DevFailed: the device is down; the job's first step would fail the same way
                raise ValueError(f"the {profile.name!r} instrument profile drives {address}, which did not answer ({str(exc).strip().splitlines()[0][:200]})") from None
            missing = [c for c in dict.fromkeys(commands) if c.lower() not in offered]
            if missing:
                raise ValueError(
                    f"the {profile.name!r} instrument profile needs {', '.join(missing)} on {address}, which does not offer them; "
                    "set this Mapper's instrument_profile to one the instrument supports"
                )
        plan["drives"] = {address: list(dict.fromkeys(commands)) for address, commands in needs.items()}

    def step_at(self, job: dict, index: int) -> dict:
        return plans.step_at(job["resolved_plan"], index)

    def step_targets(self, job: dict) -> list[dict]:
        """The stage x and y the plan goes to at its extremes, checked at reserve against the stage's limits (the
        profile says where they are: the STAGE device, or the instrument's own stage settings)."""
        plan = job["resolved_plan"]
        xs, ys = plans.stage_extremes(plan)
        return self.profile().stage_targets(plan, xs, ys)

    def phase(self, job: dict) -> str | None:
        """Which phase a job is in, for job_status."""
        if "resolved_plan" not in job:
            return None
        tiles = plans.tile_count(job["resolved_plan"])
        done = done_count(job)
        if done < tiles:
            return f"acquiring tiles ({done} of {tiles})"
        if job.get("state") in {"done", "aborted", "failed", "released"}:
            return None
        return self.step_at(job, done)["phase"] if done < job.get("step_count", tiles) else None

    def check_dry_call(self, job: dict, device: str, name: str, arguments: Any) -> dict | None:
        """A dry run's check of each recorded stage move (does the image it is for lie inside the plan box?)
        and image shift (live drift tracking: within the plan's max_total_m of the shift before the map?)."""
        profile = self.profile()
        shift = profile.image_shift_target(name, arguments)
        if shift is not None:
            return live_drift.check_shift(job, shift)
        plan = job["resolved_plan"]
        target = profile.move_target(plan, name, arguments)
        if target is None:
            return None
        x_min, x_max, y_min, y_max = plan["box_m"]
        step = self.current_step()
        phase = self.step_at(job, step)["phase"] if step is not None else "tiles"
        half_x, half_y = plans.tile_half(plan, float(plan["revisit"]["fov_m"]) if phase == "revisit" else None)
        tol = EDGE_TOLERANCE_M
        inside = all(
            low + half - tol <= float(target[axis]) <= high - half + tol
            for axis, low, high, half in (("x", x_min, x_max, half_x), ("y", y_min, y_max, half_y))
            if axis in target
        )
        xy = [float(target["x"]), float(target["y"])] if "x" in target and "y" in target else {k: float(v) for k, v in target.items()}
        return {"ok": inside, "inside_box": inside, "box_m": [x_min, x_max, y_min, y_max], "xy_m": xy, "image_half_width_m": half_x}

    # ------------------------------------------------------------------
    # Run file
    # ------------------------------------------------------------------
    def _run(self, job: dict) -> RunFile:
        """The job's run file (not opened)."""
        if "run_path" not in job:
            # Tiled only serves files under the DATA save path, so the run file must live there.
            base = Path(self.proxy(self.data_device_address).save_path) / "maps" if self.data_device_address else Path(self.output_directory)
            job["run_path"] = str(base / f"map_{job['reservation_id']}.h5")
        run = self._runs.get(job["reservation_id"])
        if run is None:
            run = self._runs[job["reservation_id"]] = RunFile(Path(job["run_path"]))
        return run

    def _register_run(self, path: Path) -> tuple[str | None, str | None]:
        """Register a finished run file with DATA; (key, note). A failure is a note: the data is on disk."""
        if not self.data_device_address or path.suffix != ".h5":
            return None, None
        try:
            return self.proxy(self.data_device_address).register_path(str(path)), None
        except Exception as exc:  # Tango raises DevFailed
            return None, f"the run file is complete but was not registered with Tiled ({exc}); DATA's reconcile registers it later"

    def _write_trace(self, job: dict) -> None:
        """This job's own events into the run file (run_file.job_trace_events); the run file names the trace file and offset."""
        run = self._run(job)
        offset = int(job.get("trace_offset", 0))
        events, dropped = job_trace_events(Path(self.trace_path), job["reservation_id"], offset)
        run.write_trace(events, str(self.trace_path), offset, dropped)

    def _close_run_file(self, job: dict) -> None:
        """Finish the run file once: trace, final name, Tiled registration; results go in result_keys."""
        if job.get("run_closed"):
            return
        self._write_trace(job)
        path, note = self._run(job).finish()
        self._runs.pop(job["reservation_id"], None)
        key, register_note = self._register_run(path)
        with self._lock:
            job["result_keys"]["run_file"] = str(path)
            if key is not None:
                job["result_keys"]["run"] = key
            notes = [n for n in (note, register_note) if n]
            if notes:
                job["result_keys"]["run_file_note"] = "; ".join(notes)
            job["run_closed"] = True
            self._save(job)

    def _open_run_file(self, job: dict, sample: dict) -> None:
        plan = job["resolved_plan"]
        run = self._run(job)
        attrs = {
            "schema_version": ENVELOPE_VERSION,
            "run_id": job["reservation_id"],
            "plan_id": job["plan_id"],
            "plan": json.dumps(plan),
            "grid": json.dumps({"rows": plan["grid"]["rows"], "cols": plan["grid"]["cols"]}),
            "axes": "images: rows=+y, columns=+x; positions in stage metres",
            "microscope": self.microscope_device_address,
            "instrument_profile": self.profile().name,
            "trace_offset": int(job.get("trace_offset") or 0),
        }
        if job.get("run"):
            attrs["run"] = str(job["run"])
        run.open(attrs, record_fields(self._name, "map_run", sample))
        job["result_keys"]["run_file"] = str(run.path)

    def on_start(self, job: dict) -> None:
        plan = job["resolved_plan"]
        profile = self.profile()
        profile.check_start(job)  # before anything is set: a refusal leaves the job resumable, nothing changed
        sample = self._sample_now()
        self._refuse_another_load(job, sample)  # also before anything is set
        # The instrument's settings for the plan, ending safe (beam blanked, tip retracted): it stays safe
        # between tiles and is made unsafe only while a tile is acquired (see the profile's acquire_tile).
        profile.setup(job)
        live_drift.start(job, profile)  # tracking on: the image shift before the map, put back at the end
        if plan.get("zoom_only") and "zoom_shift_before" not in job:
            job["zoom_shift_before"] = profile.get_image_shift(job)  # put back when the job ends
        if plan.get("revisit_only") or plan.get("zoom_only") or job.get("dry_run"):
            job["run_closed"] = True  # no tiles (or, in a dry run, no images), so no run file
        if not job.get("run_closed"):
            self._open_run_file(job, sample)

    def _sample_now(self) -> dict:
        """Which sample is loaded (record.current_sample), from the SAMPLE device; nulls and why when it cannot say."""
        address = self.sample_device_address
        try:
            sample = self.proxy(address, timeout_s=3.0) if address else None
        except Exception as exc:
            return {"sample_id": None, "sample_load_id": None, "sample_error": f"no proxy to {address}: {str(exc).splitlines()[0][:200]}"}
        return current_sample(sample)

    def _refuse_another_load(self, job: dict, sample: dict) -> None:
        """A resume onto another sample load would put the new load's features at the old load's positions."""
        recorded = self._run(job).attr("sample_load_id") if not job.get("run_closed") else None
        if not recorded:
            return
        if sample.get("sample_load_id") is None:
            raise RuntimeError(f"could not tell which sample is loaded ({sample.get('sample_error')}), so this map (load {recorded}) "
                               "was not resumed; resume it when the SAMPLE device answers")
        if sample["sample_load_id"] != recorded:
            raise RuntimeError(f"the sample was reloaded since this map started (load {recorded}, now {sample['sample_load_id']}); "
                               "its tiles would not line up: reserve a new map instead of resuming")

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def run_step(self, job: dict, index: int) -> dict:
        step = self.step_at(job, index)
        if step["phase"] == "tiles":
            return self._tile(job, step)
        profile = self.profile()
        live_drift.finish(job, profile)  # the tiles are done: the image shift goes back before any revisit move
        if job.get("dry_run") and not (step["phase"] == "revisit" and job["resolved_plan"].get("regions")):
            # Stitch, score and their revisits need images; a dry run took none. Given revisit points still rehearse their moves.
            return {"phase": step["phase"], "skipped": DRY_RUN_SKIPPED}
        self._close_run_file(job)  # the tiles are done: stitching reads the finished, registered run
        pipeline = self.pipeline()
        if step["phase"] == "select":
            return pipeline.select(job)
        if step["phase"] == "stitch":
            run = job["result_keys"].get("run") or job["result_keys"].get("run_file")
            return pipeline.stitch(job, run, plans.tile_count(job["resolved_plan"]))
        if step["phase"] == "score":
            return pipeline.score(job)
        if step["phase"] == "zoom":
            return pipeline.zoom(job, step["tier"], profile)
        return pipeline.revisit(job, step["region"], profile, self._move)

    def _tile(self, job: dict, tile: dict) -> dict | None:
        plan = job["resolved_plan"]
        index = tile["index"]
        half_x, half_y = plans.tile_half(plan)
        if not inside_box(plan["box_m"], tile["x"], tile["y"], half_x, half_y):
            raise RuntimeError(f"tile {index} at ({tile['x']:.3e}, {tile['y']:.3e}) lies outside the plan box; refusing to move")
        profile = self.profile()
        if not self._move(job, tile["x"], tile["y"]):
            return None  # aborted during a long move; a resume moves again
        reported = [float(v) for v in profile.stage_reported(job)]
        # Live drift tracking (plan drift.track): a bounded image shift cancels the drift predicted for this tile.
        correction = live_drift.correct(job, profile, tile, self._trace_drift_correction)
        # The profile makes the instrument unsafe (unblanks the beam) only for this tile's acquisitions, timed on
        # the instrument's clock, and safe again before returning, even on an error.
        acquired_tile = profile.acquire_tile(job, tile)
        if acquired_tile is None:
            return None  # aborted part way through the tile (a spectrum image); a resume redoes this tile
        image_key, acquired = acquired_tile["image_key"], acquired_tile["acquired"]
        spectrum, spectrum_key = acquired_tile.get("spectrum"), acquired_tile.get("spectrum_key")
        job.setdefault("first_tile_s", acquired)
        record = {"index": index, "row": tile["row"], "col": tile["col"], "commanded_xy_m": [tile["x"], tile["y"]], "stage_reported_m": reported, "clock_s": acquired}
        if job.get("dry_run"):
            # The commands above were recorded, not sent: there is no image to read or store.
            live_drift.measure(job, tile, None, acquired, None, float(self.max_drift_m_per_s))
            record["acquisition"] = DRY_RUN_SKIPPED
            if correction is not None:
                record["drift"] = correction
            return record
        run = self._run(job)
        images = profile.ctx.read_acquisition(image_key)
        if correction is not None:
            correction.update(live_drift.measure(job, tile, images, acquired, run, float(self.max_drift_m_per_s)) or {})
        if spectrum is None and spectrum_key is not None:
            spectrum = profile.ctx.read_acquisition(spectrum_key)
        record["key"] = image_key
        if spectrum_key is not None:
            record["spectrum_key"] = spectrum_key
        if correction is not None:
            record["drift"] = correction
        run.write_tile(plans.tile_count(plan), tile, plan, reported, acquired, image_key, images, spectrum, spectrum_key, live_drift.tile_arrays(job, correction))
        return record

    def _trace_drift_correction(self, job: dict, ok: bool, arguments: dict, clock_s: float, error: str | None = None) -> None:
        """One live drift correction into the Instrument Trace (Mapper.drift_correction)."""
        fields = {"job_id": job.get("job_id"), "run_id": job["reservation_id"], "plan_id": job.get("plan_id"), "step": arguments["tile"], "clock_s": clock_s, **self._dry(job)}
        if error is not None:
            fields["error"] = error
        self._trace.record("Mapper.drift_correction", ok, time.time(), arguments=arguments, **fields)

    def _move(self, job: dict, x: float, y: float) -> bool:
        """One stage move with the profile's preparation and recovery (a probe retracts, moves, approaches); False if aborted."""
        profile = self.profile()
        profile.before_move(job)
        if not profile.move(job, x, y):
            return False
        profile.after_move(job)
        return True

    def _restore_zoom_shift(self, job: dict) -> None:
        if job.get("zoom_shift_before") is not None:
            self.profile().set_image_shift(job, job["zoom_shift_before"])

    # ------------------------------------------------------------------
    # Ending
    # ------------------------------------------------------------------
    def on_suspend(self, job: dict, reason: str) -> None:
        self.profile().make_safe(job)  # already safe between tiles; said again so a pause never depends on it

    def on_stop(self, job: dict, reason: str) -> None:
        profile = self.profile()
        profile.make_safe(job)  # blank the beam, retract the tip
        self._restore_zoom_shift(job)
        live_drift.finish(job, profile)  # tracking on: the image shift goes back to its value before the map
        if not job.get("run_closed"):
            self._write_trace(job)
            self._run(job).close()  # a resume opens it again

    def on_finish(self, job: dict) -> dict:
        # A finished map must not leave the beam parked (or the tip engaged) on the last tile.
        profile = self.profile()
        profile.make_safe(job)
        live_drift.finish(job, profile)
        self._restore_zoom_shift(job)
        self._close_run_file(job)
        self.pipeline().link_pending(job)
        zoom = Pipeline.state(job).get("zoom")
        if zoom:
            return {"zoom_images": [t["image"] for t in zoom if t.get("image")],
                    "zoom_placed": [t["image"] for t in zoom if t.get("placement", {}).get("status") == "placed"]}
        return {}

    def _release(self, request: str) -> str:
        """Release; an unfinished run gets its final name and is sent for stitching in the background, so its tiles stay reachable."""
        reply = super()._release(request)
        try:
            reservation_id = self._parse(request, "reservation_id")["reservation_id"]
            job = self._job(reservation_id)
            if job is not None and not job.get("run_closed") and "run_path" in job and self._run(job).exists():
                self._close_run_file(job)
                run = job["result_keys"].get("run")
                if run and job["resolved_plan"].get("stitch"):
                    # Never a call to the analysis server from this Tango command: it would block the device.
                    threading.Thread(target=self._stitch_released, args=(job, run), name="mapper-release-stitch", daemon=True).start()
        except Exception as exc:  # the release itself succeeded; never turn it into an error
            self.warn_stream(f"Released, but the run file was not finalized: {exc}")
        return reply

    def _stitch_released(self, job: dict, run: str) -> None:
        with tango.EnsureOmniThread():
            self.pipeline().stitch_released(job, run)

    def summary(self, job: dict) -> dict:
        """A short reply: counts and a few keys. Every spectrum is a point of interest on the map
        (list_pois), and the run file and view_data have the rest."""
        plan = job["resolved_plan"]
        grid = plan.get("grid") or {}
        shown = SUMMARY_KEYS_SHOWN
        pipeline = {k: v for k, v in Pipeline.state(job).items() if not k.endswith("_job") and k != "drift_fit"}
        if "regions" in pipeline:
            regions = pipeline["regions"]
            pipeline["regions"] = {"count": len(regions), "first": [{k: r.get(k) for k in ("id", "x_m", "y_m", "score")} for r in regions[:shown]]}
        if "revisits" in pipeline:
            visits = pipeline["revisits"]
            pipeline["revisits"] = {
                "imaged": sum(1 for v in visits if "image" in v),
                "skipped": sum(1 for v in visits if "skipped" in v),
                "first": visits[:shown],
            }
        has_run = not (plan.get("revisit_only") or plan.get("zoom_only") or job.get("dry_run"))
        out = {
            "grid": {"rows": grid.get("rows"), "cols": grid.get("cols")},
            "box_m": plan.get("box_m"),
            "fov_m": plan.get("fov_m"),
            **({"run_file": str(self._run(job).path)} if has_run else {}),
            **pipeline,
        }
        tracked = live_drift.summary(job)
        if tracked:
            out["drift_tracking"] = tracked
        if job.get("dry_run"):
            out["acquisitions"] = DRY_RUN_SKIPPED  # no image, spectrum or run file is a stand-in for a real one
        out.update(self.profile().summary(job))
        return out

    def delete_device(self) -> None:
        super().delete_device()  # stops the workers first, so no run file is written while it closes
        for run in list(getattr(self, "_runs", {}).values()):
            try:
                run.close()
            except Exception as exc:
                self.warn_stream(f"run file {run.partial} did not close: {exc}")
        pipeline = getattr(self, "_pipeline_", None)
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception as exc:
                self.warn_stream(f"analysis session did not close: {exc}")


if __name__ == "__main__":
    Mapper.run_server()
