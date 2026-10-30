"""Every agent endpoint configuration hides at least what the twin's agent endpoint hides, and keeps a trace.

The twin's agent configuration (configs/mcp_dt.yaml) is the one that is exercised every day, so its block
lists are the reference. A configuration for another instrument may block more, never less: the Spectra 300's
used to offer agents the data server's configuration commands and kept no trace.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs" / "mcp_dt.yaml"


def _mcp(path: Path) -> dict:
    """The ``mcp`` section of a configuration run_mcp.py starts (one with an ``instrument`` section is a device stack)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {} if "instrument" in raw else raw.get("mcp") or {}


def _agent_configs() -> list[Path]:
    found = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        mcp = _mcp(path)
        if mcp and str(mcp.get("endpoint_role", "agent")) == "agent":
            found.append(path)
    return found


def test_the_reference_is_an_agent_configuration():
    assert REFERENCE in _agent_configs()


@pytest.mark.parametrize("path", _agent_configs(), ids=lambda p: p.name)
def test_agent_configuration_blocks_at_least_what_the_twin_blocks(path):
    reference, mcp = _mcp(REFERENCE), _mcp(path)
    missing_classes = set(reference.get("blocked_classes") or []) - set(mcp.get("blocked_classes") or [])
    assert not missing_classes, f"{path.name} offers agents the classes {sorted(missing_classes)}"
    blocked = mcp.get("blocked_functions") or {}
    for device_class, commands in (reference.get("blocked_functions") or {}).items():
        missing = set(commands) - set(blocked.get(device_class) or [])
        assert not missing, f"{path.name} offers agents {device_class} commands {sorted(missing)}"


@pytest.mark.parametrize("path", _agent_configs(), ids=lambda p: p.name)
def test_agent_configuration_keeps_a_trace(path):
    mcp = _mcp(path)
    assert str(mcp.get("trace") or "").strip(), f"{path.name} writes no trace of agent tool calls"
    assert str(mcp.get("endpoint_role", "")) == "agent", f"{path.name} should state endpoint_role: agent"
