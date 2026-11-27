"""The supervisor: one object that owns the queue, the atlas and the recipes.

The MCP server registers every Tango tool here and routes the agent's calls
through :meth:`Supervisor.run_tool` (origin ``agent``). The operator HTTP API
(:mod:`asyncroscopy.supervision.http`, used by SciAgentGUI and the built-in
page) calls the same methods with origin ``operator``. So there is one queue,
one history and one policy, whoever is asking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asyncroscopy.atlas import revisit
from asyncroscopy.atlas.revisit import Microscope, RevisitParams
from asyncroscopy.atlas.store import Atlas
from asyncroscopy.supervision.policy import ControlPolicy
from asyncroscopy.supervision.queue import ActionQueue, load_history
from asyncroscopy.supervision.recipes import (
    Capabilities,
    Recipe,
    RecipeStore,
    check,
    from_actions,
    new_run_id,
    resolve_params,
    substitute,
)

DEFAULT_PASSTHROUGH = ["*_get_*", "*_read_*", "list_devices", "get_data_from_key"]


@dataclass
class SupervisionConfig:
    enabled: bool = False
    dir: str = "outputs/supervision"
    atlas_dir: str = ""
    recipes_dir: str = ""
    specimen_id: str = "specimen"
    instrument_kind: str = "auto"                  # auto | twin | real
    operator_token: str = ""
    agent_wait_s: float = 50.0
    passthrough: list[str] = field(default_factory=lambda: list(DEFAULT_PASSTHROUGH))
    stop_commands: list[str] = field(default_factory=lambda: ["instrument.blank_beam"])
    detector: str = "haadf"
    policy: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, data: dict | None) -> SupervisionConfig:
        data = dict(data or {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ToolEntry:
    name: str                  # MCP tool name, e.g. "DigitalTwin_move_stage"
    role: str                  # role form, e.g. "instrument.move_stage"
    fn: Callable[..., Any]     # runs the command directly (no queue)
    device: str = ""


class Supervisor:
    def __init__(self, config: SupervisionConfig, *, microscope: Callable[[], Microscope | None] | None = None,
                 instrument_name: str = "", instrument_kind: str = "real",
                 detectors: Callable[[], set[str]] | None = None):
        self.config = config
        root = Path(config.dir)
        self.policy = ControlPolicy.from_config(config.policy)
        self.tools: dict[str, ToolEntry] = {}
        self.roles: dict[str, ToolEntry] = {}
        self._microscope_factory = microscope
        self._microscope: Microscope | None = None
        self.instrument_name = instrument_name
        self.instrument_kind = instrument_kind
        self._detectors = detectors or (lambda: set())
        self.queue = ActionQueue(policy=self.policy, history_path=root / "history.jsonl",
                                 stop_handler=self._stop_instrument, current_stage=self._current_stage,
                                 passthrough=config.passthrough)
        self.atlas = Atlas(config.atlas_dir or root / "atlas", specimen_id=config.specimen_id)
        self.recipes = RecipeStore(config.recipes_dir or root / "recipes")
        self.native: dict[str, Callable[[dict, Callable], Any]] = {
            "atlas.return_to_site": self._native_return,
            "atlas.save_site": self._native_save,
        }

    # ------------------------------------------------------------------ registry
    def register_tool(self, entry: ToolEntry) -> None:
        self.tools[entry.name] = entry
        self.roles.setdefault(entry.role, entry)

    def aliases(self, name: str) -> list[str]:
        entry = self.tools.get(name)
        if entry is not None:
            return [entry.role]
        for role, e in self.roles.items():          # atlas steps use bare command names
            if role.endswith("." + name):
                return [role, e.name]
        return []

    def to_role(self, name: str) -> str | None:
        if name in self.native or name.startswith(("atlas.", "recipe.")):
            return name
        entry = self.tools.get(name)
        return entry.role if entry else None

    def microscope(self) -> Microscope:
        if self._microscope is None and self._microscope_factory is not None:
            self._microscope = self._microscope_factory()
        if self._microscope is None:
            raise RuntimeError("no microscope is available for atlas operations (no instrument device found)")
        return self._microscope

    def set_microscope(self, mic: Microscope) -> None:
        self._microscope = mic

    def _current_stage(self) -> list[float]:
        return self.microscope().get_stage()

    def _stop_instrument(self) -> dict:
        """Out-of-band stop: run the configured stop commands directly, never through the queue."""
        results = {}
        for ref in self.config.stop_commands:
            entry = self.roles.get(ref) or self.tools.get(ref)
            if entry is None:
                results[ref] = "not available on this instrument"
                continue
            try:
                entry.fn()
                results[ref] = "ok"
            except Exception as exc:
                results[ref] = f"error: {type(exc).__name__}: {exc}"
        return results

    def capabilities(self) -> Capabilities:
        tools = set(self.roles) | set(self.native)
        try:
            detectors = {d.lower() for d in self._detectors()}
        except Exception:
            detectors = set()
        return Capabilities(tools=tools, detectors=detectors, instrument_kind=self.instrument_kind,
                            instrument=self.instrument_name)

    # ------------------------------------------------------------------ tool calls
    def run_tool(self, name: str, fn: Callable[[], Any], args: dict, origin: str = "agent",
                 site_id: str | None = None, wait_s: float | None = None) -> Any:
        """Run a device tool through the queue (or directly, for passthrough reads)."""
        names = [name, *self.aliases(name)]
        if self.queue.in_worker():
            return self.queue.child(name, args, fn, aliases=names[1:])
        if self.queue.is_passthrough(names):
            return fn()
        item = self.queue.submit(name, fn, args=args, origin=origin, site_id=site_id, aliases=names[1:])
        try:
            return self.queue.wait(item, self.config.agent_wait_s if wait_s is None else wait_s)
        except TimeoutError:
            a = item.action
            raise RuntimeError(
                f"{name} is queued as action {a.id} and has not run yet (status: {a.status}; queue "
                f"{'paused by the operator' if self.queue.state()['paused'] else 'busy'}). It will run when the "
                f"queue reaches it; check supervision_queue_status instead of calling it again.") from None

    def submit_tool(self, name: str, args: dict, origin: str = "operator") -> dict:
        """Queue a device tool by name without waiting (operator panel)."""
        entry = self.tools.get(name) or self.roles.get(name)
        if entry is None:
            raise KeyError(f"no tool {name!r}")
        item = self.queue.submit(entry.name, lambda: entry.fn(**args), args=args, origin=origin,
                                 aliases=[entry.role])
        return item.action.to_json()

    # ------------------------------------------------------------------ atlas
    def _step(self):
        return self.queue.step_fn(self.aliases)

    def _provenance(self, origin: str, extra: dict | None = None) -> dict:
        current = self.queue.current
        action = self.queue.actions.get(current) if current else None
        return {"requested_by": origin, "action_id": current,
                "run_id": action.run_id if action else None, **(extra or {})}

    def _native_save(self, args: dict, step) -> dict:
        mic = self.microscope()
        fov = args.get("fov_m") or (mic.current_fov() if hasattr(mic, "current_fov") else None)
        imsize = args.get("imsize") or (mic.current_imsize() if hasattr(mic, "current_imsize") else None) or 256
        if not fov:
            raise ValueError("fov_m is required (the instrument did not report its field of view)")
        site = revisit.save_site(self.atlas, mic, args.get("name") or "site", float(fov), int(imsize), step,
                                 landmark=bool(args.get("landmark", False)), notes=args.get("notes", ""),
                                 provenance=self._provenance(args.get("_origin", "agent")))
        return {"site_id": site.id, "name": site.name, "stage": site.stage, "fov_m": site.fov_m,
                "imsize": site.imsize, "landmark": site.landmark}

    def _native_return(self, args: dict, step) -> dict:
        site_id = args["site_id"]
        params = RevisitParams.from_json(args.get("params"))
        visit = revisit.return_to_site(self.atlas, site_id, self.microscope(), step, params,
                                       provenance=self._provenance(args.get("_origin", "agent")))
        return _visit_summary(visit)

    def _composite(self, tool: str, label: str, args: dict, origin: str, site_id: str | None,
                   body: Callable[[Callable], Any], wait: bool, run_id: str | None = None):
        step = self._step()
        item = self.queue.submit(tool, lambda: body(step), args=args, origin=origin, label=label,
                                 site_id=site_id, run_id=run_id)
        if not wait:
            return {"queued": item.action.to_json()}
        try:
            return self.queue.wait(item, self.config.agent_wait_s if origin == "agent" else None)
        except TimeoutError:
            raise RuntimeError(f"{label} is queued as action {item.action.id} and has not finished; "
                               f"check supervision_queue_status") from None

    def save_site(self, name: str, *, origin: str, notes: str = "", landmark: bool = False,
                  fov_m: float | None = None, imsize: int | None = None, wait: bool = True):
        args = {"name": name, "notes": notes, "landmark": landmark, "fov_m": fov_m, "imsize": imsize,
                "_origin": origin}
        return self._composite("atlas.save_site", f"save site {name!r}", _public(args), origin, None,
                               lambda step: self._native_save(args, step), wait)

    def return_to_site(self, site_id: str, *, origin: str, params: dict | None = None, wait: bool = True):
        site = self.atlas.get(site_id)
        args = {"site_id": site_id, "params": params or {}, "_origin": origin}
        return self._composite("atlas.return_to_site", f"return to {site.name!r} and acquire", _public(args),
                               origin, site_id, lambda step: self._native_return(args, step), wait)

    def calibrate_mapping(self, *, origin: str, fov_m: float | None = None, imsize: int | None = None,
                          wait: bool = True):
        def body(step):
            mic = self.microscope()
            fov = fov_m or mic.current_fov()
            size = imsize or mic.current_imsize() or 256
            return revisit.calibrate_mapping(self.atlas, mic, float(fov), int(size), step).to_json()
        return self._composite("atlas.calibrate_mapping", "calibrate stage-to-image mapping",
                               {"fov_m": fov_m, "imsize": imsize}, origin, None, body, wait)

    def reacquire_landmarks(self, *, origin: str, params: dict | None = None, wait: bool = True):
        def body(step):
            p = RevisitParams.from_json(params) if params else None
            return revisit.reacquire_landmarks(self.atlas, self.microscope(), step, p,
                                               provenance=self._provenance(origin))
        return self._composite("atlas.reacquire_landmarks", "reacquire landmarks and fit mount",
                               {"params": params or {}}, origin, None, body, wait)

    def new_mount(self, mount_id: str | None = None) -> str:
        return self.atlas.new_mount(mount_id)

    # ------------------------------------------------------------------ recipes
    def history(self) -> dict[str, dict]:
        return load_history(self.queue.history_path) if self.queue.history_path else {}

    def recipe_from_actions(self, name: str, action_ids: list[str], description: str = "",
                            params: dict | None = None) -> dict:
        history = self.history()
        actions = []
        for aid in action_ids:
            a = self.queue.actions.get(aid)
            data = a.to_json() if a else history.get(aid)
            if data is None:
                raise KeyError(f"no action {aid!r} in the run history")
            actions.append(data)
        actions.sort(key=lambda a: a.get("created", 0))
        recipe = from_actions(name, actions, self.to_role, description)
        if params:
            recipe.params = {k: (v if isinstance(v, dict) else {"default": v}) for k, v in params.items()}
        self.recipes.save(recipe)
        return recipe.to_json()

    def check_recipe(self, name: str) -> dict:
        return check(self.recipes.load(name), self.capabilities(), self.policy.require_twin_validation)

    def run_recipe(self, name: str, params: dict | None = None, *, origin: str, wait: bool = True):
        recipe: Recipe = self.recipes.load(name)
        verdict = check(recipe, self.capabilities(), self.policy.require_twin_validation)
        if not verdict["ok"]:
            raise RuntimeError(f"recipe {name!r} cannot start: " + "; ".join(verdict["problems"]))
        values = resolve_params(recipe, params)
        run_id = new_run_id()
        recipe_hash = recipe.content_hash()

        def body(step):
            outputs = []
            try:
                for s in recipe.steps:
                    args = substitute(s.args, values)
                    outputs.append(self._run_recipe_step(s.tool, args, origin))
                ok, error = True, None
                return {"run_id": run_id, "recipe": name, "hash": recipe_hash, "steps": len(outputs),
                        "outputs": outputs}
            except BaseException as exc:
                ok, error = False, f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.recipes.record_validation(name, ok=ok, instrument_kind=self.instrument_kind,
                                               instrument=self.instrument_name, run_id=run_id,
                                               recipe_hash=recipe_hash, error=error)

        return self._composite("recipe.run", f"recipe {name!r} ({self.instrument_kind})",
                               {"recipe": name, "params": values, "hash": recipe_hash}, origin, values.get("site_id"),
                               body, wait, run_id=run_id)

    def _run_recipe_step(self, tool: str, args: dict, origin: str) -> Any:
        if tool in self.native:
            args = {**args, "_origin": origin}
            return self.native[tool](args, self._step())
        entry = self.roles.get(tool)
        if entry is None:
            raise RuntimeError(f"recipe step tool {tool!r} is not available on this instrument")
        return self.queue.child(entry.name, args, lambda: entry.fn(**args), aliases=[entry.role])


def _public(args: dict) -> dict:
    return {k: v for k, v in args.items() if not k.startswith("_")}


def _visit_summary(visit: dict) -> dict:
    keys = ("id", "site", "site_name", "status", "reasons", "residual_nm", "sigma_nm", "ncc", "ambiguity",
            "consistency", "rotation_deg", "predicted_stage", "final_stage", "correction_from_prediction_nm",
            "warnings", "image_kind", "duration_s")
    return {k: visit.get(k) for k in keys if k in visit}
