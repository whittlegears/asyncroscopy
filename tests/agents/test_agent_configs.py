"""Every agent endpoint configuration hides at least what the twin's agent endpoint hides, and keeps a trace.

The twin's agent configuration (configs/mcp_dt.yaml) is the one that is exercised every day, so its block
lists are the reference. A configuration for another instrument may block more, never less: the Spectra 300's
used to offer agents the data server's configuration commands and kept no trace.
"""

from pathlib import Path

import pytest
import yaml

from asyncroscopy.instruments.electron_microscope.hardware.corrector_ceos import CEOSCorrector
from asyncroscopy.instruments.electron_microscope.hardware.corrector_digital_twin import DigitalTwinCorrector
from asyncroscopy.utils.commands import declared_hints

ROOT = Path(__file__).resolve().parents[2]
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
def test_agent_configuration_holds_at_least_what_the_twin_holds(path):
    reference, mcp = _mcp(REFERENCE), _mcp(path)
    missing_classes = set(reference.get("blocked_classes") or []) - set(mcp.get("blocked_classes") or [])
    assert not missing_classes, f"{path.name} offers agents the classes {sorted(missing_classes)}"
    held = mcp.get("needs_approval") or {}
    for device_class, commands in (reference.get("needs_approval") or {}).items():
        missing = set(commands) - set(held.get(device_class) or [])
        assert not missing, f"{path.name} runs {device_class} commands {sorted(missing)} without a person's approval"


@pytest.mark.parametrize("path", _agent_configs(), ids=lambda p: p.name)
def test_agent_configuration_keeps_a_trace(path):
    mcp = _mcp(path)
    assert str(mcp.get("trace") or "").strip(), f"{path.name} writes no trace of agent tool calls"
    assert str(mcp.get("endpoint_role", "")) == "agent", f"{path.name} should state endpoint_role: agent"


@pytest.mark.parametrize("dev_class", ["CEOSCorrector", "DigitalTwinCorrector"])
def test_every_corrector_write_needs_approval_in_the_reference(dev_class):
    device = {"CEOSCorrector": CEOSCorrector, "DigitalTwinCorrector": DigitalTwinCorrector}[dev_class]
    writes = {"correct", "correct_aberration", "measure_aberrations", "measure_c1a1", "acquire_tableau"}
    assert writes <= set(declared_hints(device))
    assert writes <= set(_mcp(REFERENCE)["needs_approval"][dev_class])
