"""What a startup script's YAML config means, declared once and used to validate it.

A startup script (startup_scripts/run_*.py) exposes:

    DEFAULT_CONFIG_PATH   YAML the script uses without --yaml
    SETTINGS              list[Setting]: the keys it reads, with their type, default and whether required
    is_config(raw)        True if a parsed YAML document is a config for this script
    load_config(path)     the script's own loader; raises on any invalid config
    main(argv)            runs it; accepts --yaml, and --parent-pid and --cleanup (lifecycle.py)

startup_scripts/registry.py finds every such script, so a new one needs no change elsewhere.
Nothing here describes how a config is shown; that belongs to startup_guis/.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MISSING = object()
# entries: a mapping of named entries (e.g. devices), each a mapping that ENABLED_KEY can switch off.
KINDS = ("str", "int", "bool", "path", "dir", "list", "yaml", "entries")
ENABLED_KEY = "enabled"


class ConfigError(ValueError):
    """Every problem found in a config, reported together."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# Addresses of lab machines stay out of git. A tracked config writes ${NAME};
# the value comes from the environment, or else from configs/site.env, which git
# ignores and the team shares privately (configs/site.env.example lists the names).
SITE_FILE = Path(__file__).resolve().parents[1] / "configs" / "site.env"
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def site_values(path: Path = SITE_FILE) -> dict[str, str]:
    """NAME=value lines from the site file, with the environment taking priority."""
    values: dict[str, str] = {}
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if not separator:
                raise ConfigError([f"{path.name} line {number}: expected NAME=value"])
            values[name.strip()] = value.strip().strip("\"'")
    values.update(os.environ)
    return values


def fill_placeholders(raw: Any, values: dict[str, str] | None = None) -> Any:
    """Replace every ${NAME} in the config's text values; raise if a name has no value."""
    values = site_values() if values is None else values
    missing: set[str] = set()

    def value_of(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            missing.add(name)
            return match.group(0)
        return values[name]

    def fill(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: fill(item) for key, item in node.items()}
        if isinstance(node, list):
            return [fill(item) for item in node]
        if isinstance(node, str):
            return _PLACEHOLDER.sub(value_of, node)
        return node

    filled = fill(raw)
    if missing:
        raise ConfigError(
            [f"set {', '.join(sorted(missing))} in configs/site.env or the environment (see configs/site.env.example)"]
        )
    return filled


@dataclass(frozen=True)
class Setting:
    path: str  # dotted YAML path, e.g. "tango.port"
    kind: str = "str"
    default: Any = None
    required: bool = False
    choices: tuple[str, ...] = ()  # known values; any value may still be given

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Setting {self.path}: unknown kind {self.kind!r}")


def get_path(raw: Any, path: str, default: Any = MISSING) -> Any:
    node = raw
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_path(raw: dict, path: str, value: Any) -> None:
    *parents, last = path.split(".")
    node = raw
    for part in parents:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[last] = value


def coerce(setting: Setting, value: Any) -> Any:
    """Convert a YAML value to the setting's type; raises ValueError with a readable reason."""
    kind = setting.kind
    if value is None or (value == "" and kind in ("int", "bool", "yaml", "entries")):
        if kind == "list":
            return []
        return copy.deepcopy(setting.default) if kind in ("yaml", "entries") else None
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"{setting.path} must be a whole number, not {value!r}")
        try:
            return int(str(value).strip())
        except ValueError:
            raise ValueError(f"{setting.path} must be a whole number, not {value!r}") from None
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes", "on", "1"}:
            return True
        if text in {"false", "no", "off", "0"}:
            return False
        raise ValueError(f"{setting.path} must be true or false, not {value!r}")
    if kind == "list":
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError(f"{setting.path} must be a list, not {value!r}")
    if kind == "entries":
        if not isinstance(value, dict):
            raise ValueError(f"{setting.path} must be a mapping of name to its settings, not {value!r}")
        return value
    if kind == "yaml":
        expected = type(setting.default) if setting.default is not None else None
        if expected in (dict, list) and not isinstance(value, expected):
            raise ValueError(f"{setting.path} must be a {'mapping' if expected is dict else 'list'}, not {value!r}")
        return value
    return str(value)


def is_enabled(entry: Any) -> bool:
    """Whether an entries item is switched on; raises ValueError for an unreadable flag."""
    if not isinstance(entry, dict):
        return True
    flag = coerce(Setting(ENABLED_KEY, "bool"), entry.get(ENABLED_KEY, True))
    return flag is not False


def read_settings(raw: Any, settings: list[Setting]) -> dict[str, Any]:
    """Coerced value of every setting, by path; raises ConfigError listing every problem at once."""
    if not isinstance(raw, dict):
        raise ConfigError(["the config must be a YAML mapping"])
    values, problems = {}, []
    for setting in settings:
        value = get_path(raw, setting.path)
        if value is MISSING or (setting.required and value in (None, "")):
            if setting.required:
                problems.append(f"missing required setting '{setting.path}'")
                continue
            value = setting.default
        try:
            values[setting.path] = coerce(setting, value)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ConfigError(problems)
    return values
