"""Every shipped config parses with the launcher that consumes it.

A broken YAML is the most common way a hackathon team loses an hour, so every
file in configs/ is loaded here with the same code the launchers use.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def _launcher(name):
    spec = importlib.util.spec_from_file_location(f"launcher_{name}", ROOT / "startup_scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"launcher_{name}"] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "instrument" in raw:
        config = _launcher("run_servers").load_config(path)
        assert config.instrument.file.exists(), f"{path.name}: instrument file {config.instrument.file} missing"
    elif "mcp" in raw:
        _launcher("run_mcp").load_config(path)
    elif "agent" in raw:
        _launcher("run_llm").load_config(path)
    elif "model_size" in raw:
        _launcher("run_segmentation").load_config(path)
    else:
        pytest.fail(f"{path.name}: no launcher recognises its top-level keys {sorted(raw)}")
