"""Recipes: a good run, saved, edited, rehearsed on the twin, then run for real.

A recipe is a JSON file: an ordered list of steps, each a tool reference and
its arguments, with ``${name}`` placeholders for parameters. Tool references
use the *role* form (``instrument.move_stage``, ``scan.…``, ``atlas.return_to_site``)
rather than the Tango class name, so a recipe recorded on the digital twin
resolves to the same commands on a real microscope.

Before a recipe starts, :func:`check` compares what it needs (the tools it
calls and the detector names its arguments use) with what this instrument
publishes, and refuses if anything is missing, before the first step moves
anything.

Each recipe version is identified by a hash of its steps and parameters.
``ControlPolicy.require_twin_validation`` (on by default) means a recipe runs
on a real instrument only after that exact version ran successfully on a
digital twin. Editing a recipe changes its hash, so it has to be rehearsed
again. The check lives in the control layer, not in the agent's prompt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RECIPE_SUFFIX = ".recipe.json"
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
DETECTOR_ARGS = ("detector_list", "detector", "detector_name", "detectors")


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower()
    if not slug:
        raise ValueError("a recipe needs a name with at least one letter or digit")
    return slug[:64]


@dataclass
class Step:
    tool: str                         # role form: "instrument.move_stage", "atlas.return_to_site"
    args: dict = field(default_factory=dict)
    label: str = ""


@dataclass
class Recipe:
    name: str
    steps: list[Step]
    description: str = ""
    params: dict[str, dict] = field(default_factory=dict)     # name -> {"default": ..., "description": ...}
    requires: dict[str, list[str]] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    validations: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    # -- identity --------------------------------------------------------------
    def content_hash(self) -> str:
        body = {"steps": [asdict(s) for s in self.steps], "params": self.params}
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def derived_requirements(self) -> dict[str, list[str]]:
        tools = sorted({s.tool for s in self.steps})
        detectors: set[str] = set()
        for s in self.steps:
            for key in DETECTOR_ARGS:
                value = s.args.get(key)
                values = value if isinstance(value, list) else [value]
                for v in values:
                    if isinstance(v, str) and v and not _PLACEHOLDER.search(v):
                        detectors.add(v.lower())
        req = {"tools": tools, "detectors": sorted(detectors)}
        for key, extra in (self.requires or {}).items():     # hand-added requirements are kept
            req[key] = sorted(set(req.get(key, [])) | set(extra))
        return req

    def validated_on_twin(self) -> bool:
        h = self.content_hash()
        return any(v.get("hash") == h and v.get("ok") and v.get("instrument_kind") == "twin"
                   for v in self.validations)

    # -- serialization ---------------------------------------------------------
    def to_json(self) -> dict:
        data = asdict(self)
        data["hash"] = self.content_hash()
        data["requires"] = self.derived_requirements()
        data["validated_on_twin"] = self.validated_on_twin()
        return data

    @classmethod
    def from_json(cls, data: dict) -> Recipe:
        if not isinstance(data, dict):
            raise ValueError("a recipe must be a JSON object")
        if not data.get("name"):
            raise ValueError("a recipe needs a 'name'")
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("a recipe needs a non-empty 'steps' list")
        parsed = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict) or not isinstance(s.get("tool"), str) or "." not in s["tool"]:
                raise ValueError(f"step {i + 1} needs a 'tool' in role form, e.g. 'instrument.move_stage'")
            if not isinstance(s.get("args", {}), dict):
                raise ValueError(f"step {i + 1}: 'args' must be an object")
            parsed.append(Step(tool=s["tool"], args=dict(s.get("args", {})), label=s.get("label", "")))
        params = data.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ValueError("'params' must be an object of name -> {default, description}")
        recipe = cls(name=str(data["name"]), steps=parsed, description=data.get("description", ""),
                     params={k: (v if isinstance(v, dict) else {"default": v}) for k, v in params.items()},
                     requires={k: list(v) for k, v in (data.get("requires") or {}).items()
                               if k not in ("tools", "detectors")},
                     provenance=dict(data.get("provenance", {})),
                     validations=list(data.get("validations", [])),
                     created=data.get("created", time.time()), updated=data.get("updated", time.time()))
        used = {m for s in parsed for m in _placeholders(s.args)}
        unknown = used - set(recipe.params)
        if unknown:
            raise ValueError(f"steps use undeclared parameters: {', '.join(sorted(unknown))}")
        return recipe


def _placeholders(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(_PLACEHOLDER.findall(value))
    if isinstance(value, dict):
        return set().union(*(_placeholders(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_placeholders(v) for v in value)) if value else set()
    return set()


def substitute(value: Any, params: dict[str, Any]) -> Any:
    """Replace ``${name}``. A string that is exactly one placeholder takes the value's type."""
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value)
        if whole:
            return params[whole.group(1)]
        return _PLACEHOLDER.sub(lambda m: str(params[m.group(1)]), value)
    if isinstance(value, dict):
        return {k: substitute(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, params) for v in value]
    return value


def resolve_params(recipe: Recipe, given: dict[str, Any] | None) -> dict[str, Any]:
    given = dict(given or {})
    unknown = set(given) - set(recipe.params)
    if unknown:
        raise ValueError(f"unknown recipe parameters: {', '.join(sorted(unknown))}")
    out = {}
    for name, spec in recipe.params.items():
        if name in given:
            out[name] = given[name]
        elif spec.get("default") is not None:
            out[name] = spec["default"]
        else:
            raise ValueError(f"recipe parameter {name!r} has no default; pass a value")
    return out


# ----------------------------------------------------------------------------
# Capabilities
# ----------------------------------------------------------------------------

@dataclass
class Capabilities:
    """What this instrument publishes: tools in role form, and detector names."""

    tools: set[str] = field(default_factory=set)
    detectors: set[str] = field(default_factory=set)
    instrument_kind: str = "real"          # "twin" | "real"
    instrument: str = ""

    def missing(self, recipe: Recipe) -> dict[str, list[str]]:
        req = recipe.derived_requirements()
        out = {"tools": sorted(set(req.get("tools", [])) - self.tools),
               "detectors": sorted({d.lower() for d in req.get("detectors", [])} - self.detectors)}
        return {k: v for k, v in out.items() if v}

    def to_json(self) -> dict:
        return {"tools": sorted(self.tools), "detectors": sorted(self.detectors),
                "instrument_kind": self.instrument_kind, "instrument": self.instrument}


def check(recipe: Recipe, caps: Capabilities, require_twin_validation: bool = True) -> dict:
    """Can *recipe* start on this instrument? Nothing is sent to the instrument."""
    problems = []
    missing = caps.missing(recipe)
    for kind, names in missing.items():
        problems.append(f"this instrument has no {kind[:-1]} {', '.join(names)}")
    if caps.instrument_kind == "real" and require_twin_validation and not recipe.validated_on_twin():
        problems.append(f"recipe version {recipe.content_hash()} has not run successfully on a digital twin; "
                        f"replay it on the twin first (policy: require_twin_validation)")
    return {"ok": not problems, "problems": problems, "missing": missing, "hash": recipe.content_hash(),
            "instrument_kind": caps.instrument_kind, "validated_on_twin": recipe.validated_on_twin()}


# ----------------------------------------------------------------------------
# Building a recipe from a run
# ----------------------------------------------------------------------------

def from_actions(name: str, actions: Iterable[dict], to_role: Callable[[str], str | None],
                 description: str = "", run_id: str | None = None) -> Recipe:
    """Turn finished top-level actions (in order) into a recipe.

    Only actions with status ``done`` are kept; child steps are skipped, since
    the composite action that made them is replayed as a whole.
    """
    steps, used = [], []
    for a in actions:
        if a.get("parent_id") or a.get("status") != "done":
            continue
        role = to_role(a["tool"])
        if role is None:
            raise ValueError(f"action {a['id']} uses {a['tool']!r}, which has no role form here")
        args = copy.deepcopy(a.get("args") or {})
        if a.get("site_id") and role.startswith("atlas."):
            args.setdefault("site_id", a["site_id"])
        steps.append(Step(tool=role, args=args, label=a.get("label", "")))
        used.append(a["id"])
    if not steps:
        raise ValueError("none of the selected actions finished successfully; nothing to save")
    return Recipe(name=name, steps=steps, description=description,
                  provenance={"from_actions": used, "run_id": run_id, "saved_at": time.time()})


# ----------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------

class RecipeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{_slug(name)}{RECIPE_SUFFIX}"

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob(f"*{RECIPE_SUFFIX}")):
            try:
                r = self.load_path(p)
            except Exception as exc:
                out.append({"name": p.name, "error": str(exc)})
                continue
            out.append({"name": r.name, "description": r.description, "steps": len(r.steps),
                        "hash": r.content_hash(), "validated_on_twin": r.validated_on_twin(),
                        "params": r.params, "updated": r.updated})
        return out

    def load_path(self, path: Path) -> Recipe:
        return Recipe.from_json(json.loads(path.read_text(encoding="utf-8")))

    def load(self, name: str) -> Recipe:
        p = self.path(name)
        if not p.exists():
            raise KeyError(f"no recipe {name!r}")
        return self.load_path(p)

    def save(self, recipe: Recipe, overwrite: bool = True) -> Recipe:
        p = self.path(recipe.name)
        if p.exists() and not overwrite:
            raise FileExistsError(f"recipe {recipe.name!r} already exists")
        recipe.updated = time.time()
        data = asdict(recipe)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(p)
        return recipe

    def update(self, name: str, data: dict) -> Recipe:
        """Replace a recipe's editable content; validation history is kept (it is keyed by hash)."""
        old = self.load(name)
        merged = {**asdict(old), **data, "name": old.name, "validations": old.validations,
                  "provenance": {**old.provenance, "edited_at": time.time()}, "created": old.created}
        return self.save(Recipe.from_json(merged))

    def delete(self, name: str) -> bool:
        p = self.path(name)
        if p.exists():
            p.unlink()
            return True
        return False

    def record_validation(self, name: str, *, ok: bool, instrument_kind: str, instrument: str,
                          run_id: str, recipe_hash: str, error: str | None = None) -> None:
        recipe = self.load(name)
        recipe.validations.append({"hash": recipe_hash, "ok": ok, "instrument_kind": instrument_kind,
                                   "instrument": instrument, "run_id": run_id, "at": time.time(),
                                   "error": error})
        self.save(recipe)


def new_run_id() -> str:
    return f"run-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:4]}"
