"""Find the startup scripts and which one a config belongs to.

Every startup_scripts/run_*.py that follows the contract in config_schema.py is found here,
so adding a script, or a config for one, needs no change anywhere else.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
CONFIG_DIR = PROJECT_DIR / "configs"
CONTRACT = ("DEFAULT_CONFIG_PATH", "SETTINGS", "is_config", "load_config", "main")


@lru_cache(maxsize=None)
def scripts() -> tuple[ModuleType, ...]:
    """Every startup script module that follows the contract, by file name."""
    found = []
    for path in sorted(SCRIPTS_DIR.glob("run_*.py")):
        module = importlib.import_module(f"startup_scripts.{path.stem}")
        if all(hasattr(module, name) for name in CONTRACT):
            found.append(module)
    return tuple(found)


def script_name(script: ModuleType) -> str:
    return script.__name__.rsplit(".", 1)[-1]


def script_path(script: ModuleType) -> Path:
    return Path(script.__file__).resolve()


def script_for(raw: Any) -> ModuleType | None:
    """The script a parsed YAML document configures, or None."""
    for script in scripts():
        if script.is_config(raw):
            return script
    return None


def config_files(directory: Path = CONFIG_DIR) -> list[tuple[Path, ModuleType]]:
    """Every YAML in the configs folder that some startup script runs, with that script."""
    found = []
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        script = script_for(raw)
        if script is not None:
            found.append((path, script))
    return found
