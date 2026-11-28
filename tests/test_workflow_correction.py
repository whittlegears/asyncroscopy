"""The CEOS correction LangGraph workflow, run against the real digital-twin corrector and ANALYSIS devices.

Each Tango command is wrapped as an async tool named the way the MCP bridge
names it (``{Class}_{command}``, one string argument ``arg``), so the graph
sees what it would see over MCP. The twin's coefficients are reset before and
after every test, since the device is shared by the whole session.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("langgraph.types")  # the stubs provide langgraph but not this

from asyncroscopy.agent_workflows import correction  # noqa: E402
from asyncroscopy.agent_workflows.runner import WorkflowRunner  # noqa: E402
from asyncroscopy.instruments.electron_microscope.hardware.corrector_digital_twin import DEFAULT_ABERRATIONS  # noqa: E402


def _reset(proxy):
    proxy.set_aberrations_coeff_sim("{}")  # the twin merges this over its defaults
    proxy.acquire_tableau("Enhanced 40")  # clears the running total of corrections


@pytest.fixture
def twin(corrector_proxy):
    _reset(corrector_proxy)
    yield corrector_proxy
    _reset(corrector_proxy)


@pytest.fixture
def limited_twin(limited_corrector_proxy):
    _reset(limited_corrector_proxy)
    yield limited_corrector_proxy
    _reset(limited_corrector_proxy)


class Tools:
    """Proxy commands as MCP-style tools, recording every call. ``fail``: command -> call number that raises."""

    def __init__(self, corrector, analysis, fail=None):
        self.corrector, self.analysis, self.fail = corrector, analysis, fail or {}
        self.calls: list[tuple[str, str | None]] = []

    def _make(self, proxy, command, takes_arg=True):
        async def call(arg=None):
            self.calls.append((command, arg))
            if self.fail.get(command) == sum(1 for c, _ in self.calls if c == command):
                raise RuntimeError(f"DevFailed: {command} lost the connection")
            method = getattr(proxy, command)
            return method(arg) if takes_arg else method()

        return call

    def tools(self):
        return {
            "DigitalTwinCorrector_acquire_tableau": self._make(self.corrector, "acquire_tableau"),
            "DigitalTwinCorrector_correct_aberration": self._make(self.corrector, "correct_aberration"),
            "DigitalTwinCorrector_get_correction_limits": self._make(self.corrector, "get_correction_limits", takes_arg=False),
            "ANALYSIS_correction": self._make(self.analysis, "correction"),
        }

    def count(self, command):
        return sum(1 for c, _ in self.calls if c == command)


def _run(tools, answer, request=None):
    """Run to the end, answering each pause with ``answer(pause)``. Returns (final step, pauses)."""

    async def go():
        runner = WorkflowRunner()
        runner.register("correction", lambda cp: correction.build_graph(tools, checkpointer=cp))
        step = await runner.start("correction", {"request": request or {}})
        pauses = []
        while step["waiting_for"] is not None:
            pauses.append(step["waiting_for"])
            step = await runner.resume(step["thread_id"], answer(step["waiting_for"]))
        return step, pauses

    return asyncio.run(go())


def _coefficients(proxy):
    return json.loads(proxy.get_aberrations_coeff_sim())


def test_approved_corrections_reach_the_target_on_the_twin(twin, analysis_proxy):
    tools = Tools(twin, analysis_proxy)
    assert correction.missing_tools(tools.tools()) == []
    step, pauses = _run(tools.tools(), lambda pause: "approve")

    report = correction.describe(step["state"])
    assert step["finished"] and report["status"] == "done"
    assert report["stop_reason"] == "target_reached"
    assert report["target_mrad"] == 25.0  # the ANALYSIS device's convergence semi-angle
    assert report["final_angle_mrad"] >= 25.0 > report["start_angle_mrad"]
    angles = [h["angle_mrad"] for h in report["history"]]
    assert angles == sorted(angles)  # every approved correction helped
    # One pause per correction, each showing the command it will send and the score behind it.
    assert len(pauses) == report["corrections_applied"] == tools.count("correct_aberration") >= 2
    first = pauses[0]
    assert first["kind"] == "approve_correction"
    assert first["proposal"]["aberration"] == "B2"  # the largest gain on the twin's defaults
    assert first["command"].startswith("B2 ")
    assert first["score"]["aberration_free_angle_mrad"] == report["start_angle_mrad"]
    assert _coefficients(twin)["B2"] != DEFAULT_ABERRATIONS["B2"]


def test_rejected_corrections_never_reach_the_corrector(twin, analysis_proxy):
    tools = Tools(twin, analysis_proxy)
    step, pauses = _run(tools.tools(), lambda pause: {"action": "reject"})

    report = step["state"]["report"]
    assert report["stop_reason"] == "all_rejected"
    assert tools.count("correct_aberration") == 0 and report["applied"] == []
    assert tools.count("acquire_tableau") == 1  # a rejection tries the next best from the same tableau
    assert len({p["proposal"]["aberration"] for p in pauses}) == len(pauses) > 1
    assert _coefficients(twin) == pytest.approx(DEFAULT_ABERRATIONS)


def test_round_limit_stops_the_loop(twin, analysis_proxy):
    tools = Tools(twin, analysis_proxy)
    step, pauses = _run(tools.tools(), lambda pause: True, request={"max_rounds": 2, "target_mrad": 40.0})

    report = step["state"]["report"]
    assert report["stop_reason"] == "step_limit"
    assert report["corrections_applied"] == 2 == len(pauses) == tools.count("correct_aberration")
    assert len(report["history"]) == 3  # the measurement after the last correction is scored too


def test_tool_failure_undoes_the_runs_corrections(twin, analysis_proxy):
    tools = Tools(twin, analysis_proxy, fail={"correct_aberration": 2})
    step, pauses = _run(tools.tools(), lambda pause: "approve")

    report = step["state"]["report"]
    assert report["status"] == "failed" and report["stop_reason"] == "tool_failure"
    assert "lost the connection" in report["errors"][0]
    restore = report["restore"]
    assert restore["attempted"] and "no command that reads back" in restore["note"]
    assert [u["aberration"] for u in restore["undone"]] == [report["applied"][0]["aberration"]]
    assert restore["checked"]["back_to_start"] is True
    assert _coefficients(twin) == pytest.approx(DEFAULT_ABERRATIONS, abs=1e-18)


def test_failure_before_any_correction_has_nothing_to_undo(twin, analysis_proxy):
    tools = Tools(twin, analysis_proxy, fail={"correction": 1})
    step, pauses = _run(tools.tools(), lambda pause: "approve")

    report = step["state"]["report"]
    assert report["status"] == "failed" and pauses == []
    assert report["restore"]["undone"] == [] and "Nothing to undo" in report["restore"]["note"]
    assert tools.count("correct_aberration") == 0


def test_operator_stop_can_ask_for_a_restore(twin, analysis_proxy):
    tools = Tools(twin, analysis_proxy)
    answers = iter(["approve", {"action": "stop", "restore": True}])
    step, _ = _run(tools.tools(), lambda pause: next(answers))

    report = step["state"]["report"]
    assert report["status"] == "stopped" and report["stop_reason"] == "operator_stop"
    assert len(report["restore"]["undone"]) == 1
    assert _coefficients(twin) == pytest.approx(DEFAULT_ABERRATIONS, abs=1e-18)


def test_proposals_are_shrunk_to_the_corrector_limits(limited_twin, analysis_proxy):
    # The limited twin allows 5 nm of A1 per step; give it 40 nm so A1 is the worst aberration.
    limited_twin.set_aberrations_coeff_sim(json.dumps({"A1": [40e-9, 0.0]}))
    tools = Tools(limited_twin, analysis_proxy)
    answers = iter(["approve", "stop"])
    step, pauses = _run(tools.tools(), lambda pause: next(answers))

    proposal = pauses[0]["proposal"]
    assert proposal["aberration"] == "A1" and proposal["clipped"]
    assert proposal["size"] <= 5e-9 == proposal["limit"]
    assert step["state"]["report"]["errors"] == []  # the server accepted it
    assert _coefficients(limited_twin)["A1"][0] == pytest.approx(35e-9, rel=1e-6)


def test_a_fast_tableau_is_refused_before_anything_runs(twin, analysis_proxy):
    tools = Tools(twin, analysis_proxy)
    step, pauses = _run(tools.tools(), lambda pause: "approve", request={"tableau": "Fast 18"})

    assert step["state"]["report"]["status"] == "failed" and pauses == []
    assert "third order" in step["state"]["report"]["errors"][0]
    assert tools.calls == []


def test_answers_and_missing_tools():
    assert correction.parse_answer(True) == ("approve", False)
    assert correction.parse_answer({"approved": False}) == ("reject", False)
    assert correction.parse_answer({"action": "stop", "restore": True}) == ("stop", True)
    assert correction.parse_answer("maybe") == ("reject", False)
    assert correction.missing_tools({"X_acquire_tableau": None}) == ["correction", "correct_aberration"]
    assert correction.correction_argument("C1", [2e-9]) == "C1 2e-09"
    assert set(correction.WORKFLOW["inputs"]) == set(correction.CorrectionRequest.__annotations__)
