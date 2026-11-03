"""The ``yaml`` map profile: the Mapper tiles any YAML instrument that declares an image acquisition.

A YAML device whose file has an image acquisition and stage x/y settings (or a
``map:`` section, see spec.py) offers ``get_map_profile``; this profile reads it
at reserve and drives the device through its generated commands only:

* setup: the ``safe`` commands, then ``set_<field_of_view>`` and the plan's
  ``settings`` (e.g. an AFM's setpoint_v);
* every move: ``before_move`` commands (retract the tip), ``set_<stage x>`` and
  ``set_<stage y>``, then ``after_move`` (approach again);
* every tile: the image acquisition, then the ``safe`` commands, even on an error;
* revisits: the same image at a smaller ``set_<field_of_view>``, restored after;
* budgets: ``time_s`` (every profile) and the file's ``map.budgets`` (an AFM's
  ``tip_force_nn``), each per tile and per revisit, combined as a peak or a sum.

The tiles carry the image's own pixel size and the stage position, which is all
the analysis server needs to stitch them, so a YAML instrument gets stitched,
scored and revisited maps without Python.
"""

from __future__ import annotations

import json
import time
from typing import Any

import tango

from asyncroscopy.jobs.map_profile import Cost, InstrumentProfile, register_profile

FOV_TOLERANCE = 1e-6  # relative: the field_of_view setting must make the tile this close to fov_m


@register_profile
class YamlProfile(InstrumentProfile):
    name = "yaml"
    doc = "a YAML twin: its image acquisition per tile (the file's map: section), made safe between tiles; budgets time_s and the file's own"
    job_kind = "map acquisition: a grid of images from a YAML instrument over a stage box, made safe around every move, stitched into a zoomable map when it finishes"
    plan_example = {"size_m": [3e-6, 3e-6], "fov_m": 1e-6, "overlap": 0.25}
    plan_fields = frozenset({"settings"})

    # -- the device's description ----------------------------------------
    def _describe(self, settings: dict) -> dict:
        try:
            reply = json.loads(self.instrument().get_map_profile(json.dumps({"settings": settings})))
        except tango.DevFailed as exc:
            raise ValueError(
                f"{self.mapper.microscope_device_address} cannot be mapped with the yaml profile: get_map_profile failed "
                f"({exc.args[0].desc.strip()[:200] if exc.args else exc}). It needs an image acquisition and stage x and y settings"
            ) from None
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply

    @staticmethod
    def _info(plan: dict) -> dict:
        return plan["instrument"]

    # -- plan ------------------------------------------------------------
    def resolve(self, p: dict) -> None:
        settings = dict(p.get("settings") or {})
        info = self._describe(settings)
        fov_setting = info["field_of_view"]
        if fov_setting is None:
            width = float(info["tile_m"][0])
            if "fov_m" in p and abs(float(p["fov_m"]) - width) > FOV_TOLERANCE * width:
                raise ValueError(f"this instrument's tiles are a fixed {width:.3g} m wide (it has no field_of_view setting); leave out fov_m or give {width:.3g}")
            p["fov_m"] = width
        elif "fov_m" in p:
            info = self._describe({**settings, fov_setting["setting"]: float(p["fov_m"])})
            width = float(info["tile_m"][0])
            if abs(width - float(p["fov_m"])) > FOV_TOLERANCE * float(p["fov_m"]):
                raise ValueError(f"setting {fov_setting['setting']} to {float(p['fov_m']):.3g} gives {width:.3g} m wide images; its map field_of_view must be the image width in metres")
        p["imsize"] = int(info["shape"][1])
        p["instrument"] = info

    def current_xy(self) -> tuple[float, float]:
        x, y = self._describe({})["stage_xy_m"]
        return float(x), float(y)

    def tile_size_m(self, p: dict) -> tuple[float, float]:
        width, height = self._info(p)["tile_m"]
        return float(width), float(height)

    def resolve_revisit(self, p: dict, revisit: dict) -> dict:
        info = self._info(p)
        fov_setting = info["field_of_view"]
        if fov_setting is None:
            raise ValueError("this instrument's images have a fixed field of view, so a revisit cannot zoom in; leave out revisit")
        unused = sorted(set(revisit) - {"fov_m", "points", "map", "top_k"})
        if unused:
            raise ValueError(f"revisit {unused} are not used by the yaml profile (its images have a fixed size); give revisit fov_m")
        fov = float(revisit["fov_m"])
        zoomed = self._describe({**info["plan_settings"], fov_setting["setting"]: fov})
        return {"fov_m": fov, "tile_s": zoomed["tile_s"], "budgets": zoomed["budgets"]}

    def tile_seconds(self, p: dict) -> float:
        return float(self._info(p)["tile_s"])

    def revisit_seconds(self, revisit: dict) -> float:
        return float(revisit["tile_s"])

    def _costs(self, info: dict, tiles: int, revisit: dict | None, revisits: int) -> dict[str, Cost]:
        costs = {}
        for name, budget in info["budgets"].items():
            per_revisit = float((revisit or {}).get("budgets", {}).get(name, budget)["per_revisit"]) if revisits else 0.0
            if budget["combine"] == "max":
                value = max([float(budget["per_tile"])] * bool(tiles) + [per_revisit] * bool(revisits) or [0.0])
                why = f"{budget['doc'] or name}, the most any one image needs"
            else:
                value = float(budget["per_tile"]) * tiles + per_revisit * revisits
                why = f"{budget['doc'] or name}: {tiles} tiles" + (f" and {revisits} revisit images" if revisits else "")
            costs[name] = Cost(value, budget["units"], why=why, advice=f"change the plan's settings, or map less: see the instrument's map budgets ({name})")
        return costs

    def costs(self, p: dict, grid: dict, revisit: dict | None, revisits: int, budgets: dict) -> tuple[dict[str, Cost], dict]:
        tiles = grid["rows"] * grid["cols"]
        costs = self._costs(self._info(p), tiles, revisit, revisits)
        return costs, {f"max_{name}": round(cost.value, 6) for name, cost in costs.items()}

    def revisit_only_costs(self, p: dict, revisit: dict, count: int, budgets: dict) -> tuple[dict[str, Cost], dict]:
        costs = self._costs(self._info(p), 0, revisit, count)
        return costs, {f"max_{name}": round(cost.value, 6) for name, cost in costs.items()}

    def needed_commands(self, plan: dict) -> dict[str, list[str]]:
        return {self.mapper.microscope_device_address: list(self._info(plan)["commands"])}

    def stage_targets(self, plan: dict, xs: list[tuple[str, float]], ys: list[tuple[str, float]]) -> list[dict]:
        stage = self._info(plan)["stage"]
        device = self.mapper.microscope_device_address
        out = []
        for axis, values in (("x", xs), ("y", ys)):
            limits = stage[axis]
            for step, value in values:
                out.append({"step": step, "device": device, "attribute": limits["setting"], "value": float(value), "min": limits["min"], "max": limits["max"]})
        return out

    def move_target(self, plan: dict, name: str, arguments: Any) -> dict | None:
        stage = self._info(plan)["stage"]
        for axis in ("x", "y"):
            if name.lower() == f"set_{stage[axis]['setting']}":
                return {axis: float(list(arguments)[0])}
        return None

    # -- run -------------------------------------------------------------
    def _run(self, names: list[str]) -> None:
        device = self.instrument()
        for name in names:
            device.command_inout(name)

    def _set(self, name: str, value) -> None:
        self.instrument().command_inout(f"set_{name}", value)

    def setup(self, job: dict) -> None:
        info = self._info(job["resolved_plan"])
        self.make_safe(job)
        fov_setting = info["field_of_view"]["setting"] if info["field_of_view"] is not None else None
        for name, value in info["plan_settings"].items():
            if name != fov_setting:
                self._set(name, value)
        if fov_setting is not None:
            self._set(fov_setting, float(job["resolved_plan"]["fov_m"]))

    def make_safe(self, job: dict) -> None:
        self._run(self._info(job["resolved_plan"])["safe"])

    def before_move(self, job: dict) -> None:
        self._run(self._info(job["resolved_plan"])["before_move"])

    def after_move(self, job: dict) -> None:
        self._run(self._info(job["resolved_plan"])["after_move"])

    def move(self, job: dict, x: float, y: float) -> bool:
        stage = self._info(job["resolved_plan"])["stage"]
        self._set(stage["x"]["setting"], float(x))
        self._set(stage["y"]["setting"], float(y))
        return not self.mapper.aborting(job)

    def stage_reported(self, job: dict) -> list[float]:
        stage = self._info(job["resolved_plan"])["stage"]
        device = self.instrument()
        return [float(device.command_inout(f"get_{stage['x']['setting']}")), float(device.command_inout(f"get_{stage['y']['setting']}"))]

    def clock_s(self) -> float:
        return time.time()  # a YAML twin has no clock of its own; its waits are real (time_scale) seconds

    def acquire_tile(self, job: dict, tile: dict) -> dict | None:
        info = self._info(job["resolved_plan"])
        start = self.clock_s()
        try:
            image_key = self.instrument().command_inout(info["image"])
            acquired = (start + self.clock_s()) / 2.0
        finally:
            self.make_safe(job)
        return {"image_key": image_key, "acquired": acquired}

    def acquire_revisit(self, job: dict, revisit: dict) -> str:
        plan = job["resolved_plan"]
        fov_setting = self._info(plan)["field_of_view"]["setting"]
        self._set(fov_setting, float(revisit["fov_m"]))
        try:
            return self.instrument().command_inout(self._info(plan)["image"])
        finally:
            self.make_safe(job)
            self._set(fov_setting, float(plan["fov_m"]))
