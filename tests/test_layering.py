"""The dependency rule: core <- contracts / trace / data / safety <- instruments / jobs <- mcp <- agents.

Each lower package (and every module in it) is imported in a fresh interpreter, which must then hold
no module of asyncroscopy.mcp or asyncroscopy.agents.
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
print(json.dumps(sorted(m for m in sys.modules if m.startswith(("asyncroscopy.mcp", "asyncroscopy.agents")))))
"""


@pytest.mark.parametrize("package", LOWER)
def test_lower_layers_never_load_mcp_or_agents(package):
    done = subprocess.run([sys.executable, "-c", PROBE, package], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"importing {package} loaded {loaded}"


def test_core_imports_nothing_from_asyncroscopy():
    done = subprocess.run(
        [sys.executable, "-c", PROBE.replace('("asyncroscopy.mcp", "asyncroscopy.agents")', '("asyncroscopy.",)'), "asyncroscopy.core"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    loaded = [m for m in json.loads(done.stdout.strip().splitlines()[-1]) if not m.startswith("asyncroscopy.core")]
    assert loaded == []
