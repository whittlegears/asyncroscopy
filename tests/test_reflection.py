"""Tests for the post-run skill reflection helpers (stdlib-only, no langchain)."""

from asyncroscopy.mcp.backends.reflection import (
    propose_skill_tool,
    proposals_from_tool_calls,
    record_trace,
    reflection_task_message,
    should_reflect,
    trace_char_cap,
    trace_line_cap,
)


class TestShouldReflect:
    def test_fires_at_the_threshold(self):
        assert should_reflect(4, 4, True) is True
        assert should_reflect(9, 4, True) is True

    def test_quiet_below_the_threshold(self):
        assert should_reflect(3, 4, True) is False
        assert should_reflect(0, 4, True) is False

    def test_zero_threshold_disables_reflection(self):
        assert should_reflect(100, 0, True) is False

    def test_no_skills_service_disables_reflection(self):
        assert should_reflect(100, 4, False) is False


class TestRecordTrace:
    def test_long_lines_are_truncated(self):
        trace = []
        record_trace(trace, "x" * (trace_char_cap + 50))
        assert len(trace[0]) == trace_char_cap + 1
        assert trace[0].endswith("…")

    def test_the_trace_stops_growing_at_the_line_cap(self):
        trace = []
        for index in range(trace_line_cap + 10):
            record_trace(trace, f"line {index}")
        assert len(trace) == trace_line_cap + 1
        assert trace[-1] == "... trace truncated ..."


class TestTaskMessage:
    def test_carries_prompt_trace_answer_and_roster(self):
        message = reflection_task_message(
            "focus then acquire",
            "done",
            ["get_focus({})", "-> 1.2"],
            [{"id": "beam-alignment", "name": "Beam", "description": "Align first."}],
        )
        assert "focus then acquire" in message
        assert "get_focus({})" in message
        assert "done" in message
        assert "- beam-alignment: Align first." in message

    def test_empty_trace_and_roster_are_stated_not_blank(self):
        message = reflection_task_message("task", "answer", [], [])
        assert "no tool calls recorded" in message
        assert "none" in message


class TestProposalExtraction:
    def test_extracts_name_and_content(self):
        calls = [{"name": "propose_skill", "args": {"name": "Focus Recovery", "content": "# Steps"}}]
        assert proposals_from_tool_calls(calls) == [("Focus Recovery", "# Steps")]

    def test_ignores_other_tools_and_incomplete_args(self):
        calls = [
            {"name": "something_else", "args": {"name": "x", "content": "y"}},
            {"name": "propose_skill", "args": {"name": "", "content": "y"}},
            {"name": "propose_skill", "args": {"name": "x"}},
            {"name": "propose_skill"},
        ]
        assert proposals_from_tool_calls(calls) == []

    def test_empty_input_yields_nothing(self):
        assert proposals_from_tool_calls([]) == []
        assert proposals_from_tool_calls(None) == []


class TestToolSchema:
    def test_the_bound_tool_requires_name_and_content(self):
        function = propose_skill_tool["function"]
        assert function["name"] == "propose_skill"
        assert function["parameters"]["required"] == ["name", "content"]
