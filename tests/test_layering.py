"""The dependency rule: core <- contracts / trace / data / safety <- instruments / jobs <- mcp <- agents.

Each lower package (and every module in it) is imported in a fresh interpreter, which must then hold
no module of asyncroscopy.mcp or asyncroscopy.agents; mcp none of asyncroscopy.instruments or
asyncroscopy.agents (it learns what an instrument is from the devices it discovers); jobs none of
asyncroscopy.instruments (each modality registers its own map profile).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOWER = ["asyncroscopy.core", "asyncroscopy.contracts", "asyncroscopy.trace", "asyncroscopy.data", "asyncroscopy.safety"]

PROBE = """
import importlib, json, pkgutil, sys
package = importlib.import_module(sys.argv[1])
for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
    importlib.import_module(info.name)
print(json.dumps(sorted(m for m in sys.modules if m.startswith(tuple(sys.argv[2].split(","))))))
"""
ABOVE = "asyncroscopy.mcp,asyncroscopy.agents"


def loaded_by(package: str, forbidden: str) -> list[str]:
    """The forbidden modules (comma-separated prefixes) that importing every module of ``package`` loads."""
    done = subprocess.run([sys.executable, "-c", PROBE, package, forbidden], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("package", LOWER)
def test_lower_layers_never_load_mcp_or_agents(package):
    assert loaded_by(package, ABOVE) == []


def test_mcp_never_loads_instruments_or_agents():
    assert loaded_by("asyncroscopy.mcp", "asyncroscopy.instruments,asyncroscopy.agents") == []


def test_jobs_never_load_instruments():
    assert loaded_by("asyncroscopy.jobs", "asyncroscopy.instruments,asyncroscopy.mcp,asyncroscopy.agents") == []


def test_core_imports_nothing_from_asyncroscopy():
    assert [m for m in loaded_by("asyncroscopy.core", "asyncroscopy.") if not m.startswith("asyncroscopy.core")] == []
