"""asyncroscopy.testing as an out-of-tree team uses it: its own interpreter, its own test context."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def test_helpers_serve_an_instrument_through_the_mcp_server_and_gate(tmp_path):
    # A fresh interpreter: PyTango allows one in-process test context each, and this suite has one.
    done = subprocess.run([sys.executable, str(HERE / "out_of_tree_check.py"), str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["indent_doc"].startswith("Make one indent")
    assert result["envelope"]
    # What a small model reads: the purpose once, and no DevVoid type lines.
    assert result["retract_doc"].count("Lift the tip") == 1 and "DevVoid" not in result["retract_doc"] and "Output Description" not in result["retract_doc"]
    assert result["held"] == {"SimIndenter_indent": True, "SimIndenter_retract": False, "IndentGrid_reserve": False}
