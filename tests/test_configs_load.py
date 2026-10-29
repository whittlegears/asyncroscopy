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
        if config.instrument.file is not None:
            assert config.instrument.file.exists(), f"{path.name}: instrument file {config.instrument.file} missing"
        else:  # given as module_name (load_config has checked it imports)
            assert importlib.util.find_spec(config.instrument.module_name), f"{path.name}: {config.instrument.module_name} not importable"
        yaml_path = config.instrument.properties.get("yaml_path")
        if yaml_path:  # an instrument described in YAML: the file must load
            from asyncroscopy.instruments.yaml_sim.spec import load_spec

            load_spec(ROOT / yaml_path[0])
    elif "mcp" in raw:
        _launcher("run_mcp").load_config(path)
    elif "agent" in raw:
        _launcher("run_llm").load_config(path)
    elif "model_size" in raw:
        _launcher("run_segmentation").load_config(path)
    else:
        pytest.fail(f"{path.name}: no launcher recognises its top-level keys {sorted(raw)}")


MINIMAL_MCP = """
tango: {host: localhost, port: 9094}
mcp:
  name: X
  transport: streamable-http
  http_host: 127.0.0.1
  http_port: 8000
  data_device_address: asyncroscopy/data/default
  quiet: true
  blocked_classes: []
  blocked_functions: {}
"""


def test_an_agent_endpoint_without_an_approval_gate_is_refused_unless_said_on_purpose(tmp_path):
    run_mcp = _launcher("run_mcp")
    ungated = tmp_path / "ungated.yaml"
    ungated.write_text(MINIMAL_MCP, encoding="utf-8")
    with pytest.raises(ValueError, match="names no approvals device"):
        run_mcp.load_config(ungated)
    opted_out = tmp_path / "sim.yaml"
    opted_out.write_text(MINIMAL_MCP + "  safety: {allow_ungated: true}\n", encoding="utf-8")
    assert run_mcp.load_config(opted_out).mcp.approvals_device_address == ""
    operator = tmp_path / "operator.yaml"
    operator.write_text(MINIMAL_MCP + "  endpoint_role: operator\n", encoding="utf-8")
    assert run_mcp.load_config(operator).mcp.endpoint_role == "operator"


def test_every_hardware_stack_starts_its_approval_gate_in_approve_mode():
    for name in ("Spectra300.yaml", "Spectra300eels.yaml", "covalent-F200.yaml", "covalent-talos.yaml"):
        raw = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        approvals = [d for d in raw["devices"].values() if d.get("module_name") == "asyncroscopy.safety.approvals"]
        assert approvals and approvals[0]["properties"]["default_mode"] == "approve", name
