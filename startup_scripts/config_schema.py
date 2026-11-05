"""Settings the startup scripts declare once, used both to validate their YAML and to build the startup GUI.

A startup script that the GUI can drive exposes:

    TITLE                 window title
    LAUNCHER              script path relative to the project, e.g. "startup_scripts/run_mcp.py"
    DEFAULT_CONFIG_PATH   YAML the script uses without --yaml (and the GUI opens first,
                          unless GUI_DEFAULT_CONFIG_PATH is set)
    SETTINGS              list[Setting], one per editable YAML value
    DEVICES               optional DeviceTable for a mapping of switchable entries
    STATUS_PATTERNS       optional [(regex, badge text, is_error)] matched against output lines
    is_config(raw)        True if a parsed YAML document is a config for this script
    load_config(path)     the script's own loader; raises on any invalid config

The GUI edits the raw YAML in place, so keys it shows no widget for are kept as they are.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

MISSING = object()
KINDS = ("str", "int", "bool", "path", "dir", "list", "yaml")


class ConfigError(ValueError):
    """Every problem found in a config, reported together."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class Setting:
    path: str  # dotted YAML path, e.g. "tango.port"
    kind: str = "str"
    default: Any = None
    label: str = ""
    section: str = "General"
    help: str = ""
    required: bool = False
    choices: tuple[str, ...] = ()  # suggestions; any value may still be typed
    presets: dict[str, str] = field(default_factory=dict)  # one-click values, first is applied when the GUI opens

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Setting {self.path}: unknown kind {self.kind!r}")

    @property
    def title(self) -> str:
        return self.label or self.path.rsplit(".", 1)[-1].replace("_", " ").capitalize()


@dataclass(frozen=True)
class DeviceTable:
    """A YAML mapping of named entries, each switched on or off by a boolean key."""

    path: str = "devices"
    label: str = "Devices"
    enabled_key: str = "enabled"
    summary: Callable[[str, dict], str] | None = None


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
    """Convert a YAML or GUI value to the setting's type; raises ValueError with a readable reason."""
    kind = setting.kind
    if value is None or (value == "" and kind in ("int", "bool", "yaml")):
        return [] if kind == "list" else copy.deepcopy(setting.default) if kind == "yaml" else None
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
    if kind == "yaml":
        expected = type(setting.default) if setting.default is not None else None
        if expected in (dict, list) and not isinstance(value, expected):
            raise ValueError(f"{setting.path} must be a {'mapping' if expected is dict else 'list'}, not {value!r}")
        return value
    return str(value)


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
