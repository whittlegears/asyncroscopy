"""startup_scripts/show_agent_view.py: the table a team reads to see what its agent can do."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("show_agent_view", Path(__file__).resolve().parents[1] / "startup_scripts" / "show_agent_view.py")
view = importlib.util.module_from_spec(spec)
spec.loader.exec_module(view)


def test_each_tool_says_its_risk_what_auto_mode_does_and_whether_the_model_gets_it():
    info = {
        "risks": {"Twin_get_stage": "low", "Twin_move_stage": "high", "Twin_set_column_valves": "high", "Pump_prime": "high", "Mapper_start": "high"},
        "auto": {"Twin_move_stage": "routine", "Twin_set_column_valves": "held", "Pump_prime": "held", "Mapper_start": "job"},
        "agent_tools": ["*_get_stage", "*_move_stage", "find_tools"],
    }
    names = ["Twin_get_stage", "Twin_move_stage", "Twin_set_column_valves", "Pump_prime", "Mapper_start", "find_tools"]
    rows = {r["tool"]: r for r in view.rows_from(info, names, ["pump_*"])}
    assert rows["Twin_get_stage"] == {"tool": "Twin_get_stage", "risk": "low", "auto": "runs", "model": "yes"}
    assert rows["Twin_move_stage"]["auto"] == "runs (routine)"
    assert rows["Twin_set_column_valves"]["auto"] == "held until switched on" and rows["Twin_set_column_valves"]["model"] == "no"
    assert rows["Pump_prime"]["auto"] == "runs (switched on)"
    assert rows["Mapper_start"]["auto"].startswith("runs if its job")
    assert rows["find_tools"]["model"] == "yes" and rows["find_tools"]["risk"] == "low"
