"""The C1/A1 hill-climb LangGraph workflow, through the real instrument MCP server over the digital-twin corrector.

``correct_aberration`` and ``get_correction_limits`` are the twin corrector's
real commands reached through the MCP tool wrappers, so its limits apply. The
image and its score are synthetic (see tests/workflow_twin.py): sharpness peaks
where the twin's coefficient for the tuned aberration is zero.
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

pytest.importorskip("langgraph.types")  # the stubs provide langgraph but not this

from fastmcp import Client  # noqa: E402

from asyncroscopy.agent_workflows import aberration_climb  # noqa: E402
from asyncroscopy.agent_workflows import climb as maths  # noqa: E402
from tests.workflow_twin import SyntheticImages, client_tools, mcp_server, run_workflow  # noqa: E402

C1_START = 24e-9
A1_START = [4e-9, -3e-9]


def _reset(proxy, coefficients=None):
    proxy.set_aberrations_coeff_sim(json.dumps(coefficients or {}))  # merged over the twin's defaults
    proxy.acquire_tableau("Enhanced 40")  # clears the running total of corrections


def _coefficient(proxy, name):
    return json.loads(proxy.get_aberrations_coeff_sim())[name]


@pytest.fixture
def twin(corrector_proxy):
    _reset(corrector_proxy, {"C1": [C1_START], "A1": A1_START})
    yield corrector_proxy
    _reset(corrector_proxy)


@pytest.fixture
def limited_twin(limited_corrector_proxy):
    _reset(limited_corrector_proxy, {"A1": A1_START})
    yield limited_corrector_proxy
    _reset(limited_corrector_proxy)


def _server(tango_ctx, monkeypatch, device):
    return mcp_server(tango_ctx, monkeypatch, [device])


def _run(server, proxy, answer, request, fail_on=None, width=10e-9):
    """Run the workflow over MCP; return (final step, pauses, logged MCP calls)."""
    name = request.get("name", "C1")
    images = SyntheticImages(lambda: math.hypot(*_coefficient(proxy, name)) / width, fail_on=fail_on)
    log: list = []

    async def go():
        async with Client(server.mcp) as client:
            names = [t.name for t in await client.list_tools()]
            tools = {**client_tools(client, names, log), **images.tools()}
            assert aberration_climb.missing_tools(tools) == []
            build = lambda cp: aberration_climb.build_graph(tools, checkpointer=cp)  # noqa: E731
            return await run_workflow("aberration_climb", build, {"request": request}, answer)

    step, pauses = asyncio.run(go())
    return step, pauses, log


def _corrections(log):
    return [args["args"] for name, args in log if name.endswith("_correct_aberration")]


def test_c1_climb_reaches_zero_defocus(tango_ctx, monkeypatch, twin):
    server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/default")
    step, pauses, log = _run(server, twin, lambda pause: {"approved": True}, {"name": "C1", "max_iterations": 12})

    report = aberration_climb.describe(step["state"])
    assert step["finished"] and report["status"] == "done", report["errors"]
    assert len(pauses) == 1 and pauses[0]["kind"] == "approve_aberration_climb"
    plan = pauses[0]["plan"]
    assert plan["step_m"] == 10e-9 and plan["max_offset_m"] == pytest.approx(50e-9)
    assert plan["corrector_max_step_m"] is None  # the default twin has no limits
    # The twin subtracts what is sent, so the offset that cancels C1 is +C1_START.
    assert report["offset_m"][0] == pytest.approx(C1_START, abs=2e-9)
    assert _coefficient(twin, "C1")[0] == pytest.approx(C1_START - report["offset_m"][0], abs=1e-15)
    assert report["improvement"] > 3
    assert all(c.startswith("C1 ") and len(c.split()) == 2 for c in _corrections(log))
    assert [h["trial"] for h in report["history"][:3]] == ["baseline", "initial 0", "initial 1"]


def test_a1_climb_uses_eight_probes_and_the_shared_maths(tango_ctx, monkeypatch, twin):
    server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/default")
    step, _, log = _run(server, twin, lambda pause: True, {"name": "A1", "max_iterations": 1})

    history = step["state"]["report"]["history"]
    assert len(history) == 1 + 8 + 1
    explored = history[:9]
    expected = maths.propose_next_point([h["offset_m"] for h in explored], [h["score"] for h in explored], 2e-9, 10e-9)
    assert history[9]["offset_m"] == pytest.approx(list(expected), abs=1e-15)
    assert all(c.startswith("A1 ") and len(c.split()) == 3 for c in _corrections(log))
    assert step["state"]["report"]["stop_reason"] == "max_iterations"


def test_declined_plan_sends_no_correction(tango_ctx, monkeypatch, twin):
    server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/default")
    step, pauses, log = _run(server, twin, lambda pause: {"approved": False}, {"name": "C1"})

    assert step["state"]["report"]["status"] == "declined" and len(pauses) == 1
    assert _corrections(log) == []
    assert _coefficient(twin, "C1") == [C1_START]


def test_failure_undoes_every_correction(tango_ctx, monkeypatch, twin):
    server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/default")
    step, _, log = _run(server, twin, lambda pause: True, {"name": "A1", "max_iterations": 4}, fail_on=6)

    report = step["state"]["report"]
    assert report["status"] == "failed" and "scan timed out" in report["errors"][0]
    assert report["restored_to_start"] and report["offset_m"] == pytest.approx([0.0, 0.0], abs=1e-18)
    assert _coefficient(twin, "A1") == pytest.approx(A1_START, abs=1e-18)
    assert len(_corrections(log)) >= 6  # five probes sent, then the undo


def test_hard_limit_holds_when_the_best_value_is_beyond_it(tango_ctx, monkeypatch, twin):
    server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/default")
    request = {"name": "C1", "step_m": 5e-9, "max_offset_m": 10e-9, "max_iterations": 10}
    step, _, _ = _run(server, twin, lambda pause: True, request)

    report = step["state"]["report"]
    assert report["status"] == "done"
    assert all(abs(h["offset_m"][0]) <= 10e-9 * (1 + 1e-6) for h in report["history"])
    assert report["offset_m"][0] == pytest.approx(10e-9, abs=0.5e-9)  # as near C1_START as allowed


def test_corrections_are_split_to_the_corrector_max_step(tango_ctx, monkeypatch, limited_twin):
    server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/limited")
    request = {"name": "A1", "step_m": 8e-9, "max_offset_m": 16e-9, "max_iterations": 3}
    step, pauses, log = _run(server, limited_twin, lambda pause: True, request)

    report = step["state"]["report"]
    assert pauses[0]["plan"]["corrector_max_step_m"] == 5e-9
    assert report["status"] == "done" and report["errors"] == []  # the corrector refused nothing
    sizes = [math.hypot(*map(float, c.split()[1:])) for c in _corrections(log)]
    assert max(sizes) <= 5e-9 * (1 + 1e-9)
    assert _coefficient(limited_twin, "A1") == pytest.approx([a - o for a, o in zip(A1_START, report["offset_m"])], abs=1e-15)


def test_c1_climb_stays_inside_the_total_limit_since_the_tableau(tango_ctx, monkeypatch, limited_corrector_proxy):
    """The limited twin allows 25 nm of C1 corrections in total between tableaus, undoing included."""
    _reset(limited_corrector_proxy, {"C1": [C1_START]})
    try:
        server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/limited")
        request = {"name": "C1", "step_m": 8e-9, "max_offset_m": 30e-9, "max_iterations": 10}
        step, pauses, log = _run(server, limited_corrector_proxy, lambda pause: True, request)
        report = step["state"]["report"]
        assert pauses[0]["plan"]["corrector_max_total_m"] == 25e-9
        assert report["status"] == "done" and report["errors"] == []  # nothing refused
        assert "max_total_since_tableau" in (report["stop_reason"] or "")
        sent = sum(abs(float(c.split()[1])) for c in _corrections(log))
        assert sent <= 25e-9 * (1 + 1e-9)
        assert _coefficient(limited_corrector_proxy, "C1")[0] == pytest.approx(C1_START - report["offset_m"][0], abs=1e-15)
    finally:
        _reset(limited_corrector_proxy)


def test_bad_requests_are_refused_before_any_correction(tango_ctx, monkeypatch, twin):
    server = _server(tango_ctx, monkeypatch, "asyncroscopy/corrector/default")
    step, pauses, log = _run(server, twin, lambda pause: True, {"name": "C3"})

    report = step["state"]["report"]
    assert report["status"] == "failed" and pauses == [] and _corrections(log) == []
    assert "name must be one of" in report["errors"][0]


def test_plan_and_tool_checks():
    plan = aberration_climb.plan_request({"name": "A1", "metric": "entropy"})
    assert "needs the pixels" in plan["problems"][0]
    assert plan["max_measurements"] == 1 + 8 + aberration_climb.DEFAULT_MAX_ITERATIONS
    assert aberration_climb.correction_text("A1", [2e-9, -1e-9]) == "A1 2e-09 -1e-09"
    assert aberration_climb.missing_tools({"X_correct_aberration": None}) == ["acquire_scanned_image", "resolution"]
    assert set(aberration_climb.WORKFLOW["inputs"]["request"]) == set(aberration_climb.AberrationRequest.__annotations__)
