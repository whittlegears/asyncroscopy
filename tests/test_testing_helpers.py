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
    # Approve mode: the indent waits for an operator; the safe stop and the plan check run (and succeed).
    assert result["outcome"] == {"SimIndenter_indent": "held", "SimIndenter_retract": "ran", "IndentGrid_reserve": "ran"}
    assert result["async_descriptions_match"]
    assert ":param" not in result["move_doc"] and "Input Description:" in result["move_doc"]
    assert result["override_by_tool_name"] == "low"
    assert result["first_override_wins"] == "high"  # listed first, in either form
    assert [w for w in result["risk_warnings"] if "SimIndentr.indent" in w]
    assert any("'SimIndentr.indent' matches no command" in line for line in result["startup_warnings"])
    assert "session-scoped" in result["second_context"]
    assert result["risks"] == {"SimIndenter_indent": "high", "SimIndenter_retract": "low", "IndentGrid_reserve": "low", "IndentGrid_start": "high"}
