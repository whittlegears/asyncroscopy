"""Every shipped config parses with the launcher that consumes it.

A broken YAML is the most common way a hackathon team loses an hour, so every
file in configs/ is loaded here with the same code the launchers use.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
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
        else:  # given as module_name and load_config checked it imports
            assert importlib.util.find_spec(config.instrument.module_name), f"{path.name}: {config.instrument.module_name} not importable"
        yaml_path = config.instrument.properties.get("yaml_path")
        if yaml_path:
            from asyncroscopy.instruments.yaml_sim.spec import load_spec

            load_spec(ROOT / yaml_path[0])
    elif "mcp" in raw:
        _launcher("run_mcp").load_config(path)
    elif "agent" in raw:
        _launcher("run_llm").load_config(path)
    elif "pytemlib" in raw:
        _launcher("run_pytemlib").load_config(path)
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
  needs_approval: {}
"""


def test_a_minimal_agent_or_operator_endpoint_config_loads(tmp_path):
    run_mcp = _launcher("run_mcp")
    agent = tmp_path / "agent.yaml"
    agent.write_text(MINIMAL_MCP, encoding="utf-8")
    assert run_mcp.load_config(agent).mcp.endpoint_role == "agent"
    operator = tmp_path / "operator.yaml"
    operator.write_text(MINIMAL_MCP + "  endpoint_role: operator\n", encoding="utf-8")
    assert run_mcp.load_config(operator).mcp.endpoint_role == "operator"


def test_an_agent_endpoint_starts_its_operator_endpoint_on_loopback(tmp_path):
    run_mcp = _launcher("run_mcp")
    path = tmp_path / "agent.yaml"
    held = "needs_approval: {Mapper: [abort_all]}"
    extra = "  operator_port: 8001\n  trace: outputs/trace/mcp_agent.jsonl\n"
    path.write_text(MINIMAL_MCP.replace("needs_approval: {}", held) + extra, encoding="utf-8")
    agent = run_mcp.load_config(path)
    operator = run_mcp.operator_config(agent).mcp
    assert (operator.endpoint_role, operator.http_host, operator.http_port) == ("operator", "127.0.0.1", 8001)
    assert operator.needs_approval == {} and agent.mcp.needs_approval == {"Mapper": ["abort_all"]}
    assert operator.permissions_file == agent.mcp.permissions_file
    assert (operator.trace, operator.trace_follow) == ("outputs/trace/mcp_operator.jsonl", ["outputs/trace"])
    assert operator.auth is None and operator.name != agent.mcp.name
    assert run_mcp.operator_config(run_mcp.operator_config(agent)) is None


def test_an_operator_port_is_refused_where_it_cannot_work(tmp_path):
    run_mcp = _launcher("run_mcp")
    same = tmp_path / "same.yaml"
    same.write_text(MINIMAL_MCP + "  operator_port: 8000\n", encoding="utf-8")
    on_operator = tmp_path / "operator.yaml"
    on_operator.write_text(MINIMAL_MCP + "  endpoint_role: operator\n  operator_port: 8001\n", encoding="utf-8")
    for path in (same, on_operator):
        with pytest.raises(ValueError, match="operator_port"):
            run_mcp.load_config(path)


def test_one_stacks_endpoints_share_one_permissions_file_from_any_directory(tmp_path, monkeypatch):
    run_mcp = _launcher("run_mcp")
    shared = (ROOT / "outputs" / "permissions" / "permissions.json").resolve()
    found = set()
    for where in (ROOT, tmp_path):
        monkeypatch.chdir(where)
        for name in ("mcp_dt.yaml", "mcp_operator.yaml", "mcp.yaml", "mcp_raman.yaml", "mcp_raman_operator.yaml"):
            config = run_mcp.load_config(ROOT / "configs" / name)
            found.add(config.mcp.permissions_file)
            operator = run_mcp.operator_config(config)
            if operator is not None:
                found.add(operator.permissions_file)
    assert found == {str(shared)}
    elsewhere = tmp_path / "stack" / "agent.yaml"
    elsewhere.parent.mkdir()
    elsewhere.write_text(MINIMAL_MCP + "  permissions_file: state/permissions.json\n", encoding="utf-8")
    assert run_mcp.load_config(elsewhere).mcp.permissions_file == str((tmp_path / "stack" / "state" / "permissions.json").resolve())
