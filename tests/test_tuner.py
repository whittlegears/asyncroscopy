"""Tuner job: autofocus and stigmation on the twin, with its safety rules.

The tuner twin (tests/conftest.py, asyncroscopy/digitaltwin/tuner) hides a focus
error the job cannot read: ``set_focus_error`` sets it, and ``_focus_truth`` (what
export_ground_truth reports) gives the residual C1 and A1 at the field, specimen
height included. The "done when" of docs/designs/concepts_brief.md item 6:

* 10 of 10 seeds starting at +/-200 nm defocus, on a specimen with a height
  gradient, end with |C1| below 5 nm within the declared dose;
* a corrector with the correction sign inverted aborts and restores the snapshot;
* an over-budget plan is refused before the beam is unblanked;
* a 40-line example strategy registered from a test works.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import tango
from fastmcp.exceptions import ToolError

from asyncroscopy.instruments.electron_microscope import tuner as tuner_module
from asyncroscopy.instruments.electron_microscope.mapper import frame_dose
from asyncroscopy.instruments.electron_microscope.tuner import STRATEGIES, Strategy, register_strategy
from asyncroscopy.mcp.mcp_server import MCPServer
from asyncroscopy.routines import focus

FOCUS_PLAN = {"corrections": ["focus"], "strategy": "image_metric", "dose_budget_e_per_a2": 1500.0}


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def wait(proxy, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = call(proxy, "job_status", {"job_id": job_id})
        if status["state"] in ("done", "failed", "aborted"):
            return status
        time.sleep(0.05)
    raise TimeoutError(status)


def run(proxy, plan, timeout=180):
    reserved = call(proxy, "reserve", {"plan": plan})
    assert "error" not in reserved, reserved
    started = call(proxy, "start", {"reservation_id": reserved["reservation_id"]})
    status = wait(proxy, started["job_id"], timeout)
    results = call(proxy, "results", {"job_id": started["job_id"]})
    call(proxy, "release", {"reservation_id": reserved["reservation_id"]})
    return reserved, status, results


@pytest.fixture(scope="session")
def tuner_twin(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/tuner"))
    proxy.ping()
    return proxy, tango.Util.instance().get_device_by_name("asyncroscopy/digitaltwin/tuner")


@pytest.fixture(scope="session")
def tuner_proxy(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/tuner/default"))
    proxy.set_timeout_millis(60_000)
    return proxy


@pytest.fixture(scope="session")
def wrongsign_tuner_proxy(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/tuner/wrongsign"))
    proxy.set_timeout_millis(60_000)
    return proxy


def corrector(tango_ctx, name):
    return tango.DeviceProxy(tango_ctx.get_device_access(f"asyncroscopy/corrector/{name}"))


@pytest.fixture
def fresh(tuner_twin, tango_ctx):
    """Put the tuner twin back to a known column: no hidden error, zero defocus and stigmator, beam blanked."""
    proxy, device = tuner_twin
    corrector(tango_ctx, "tuner").set_aberrations_coeff_sim(json.dumps({}))  # the defaults
    device.set_focus_error(0.0, (0.0, 0.0))
    proxy.set_defocus(0.0)
    proxy.set_stigmator([0.0, 0.0])
    proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    proxy.blank_beam()
    return proxy, device


def setup_error(twin, device, seed: int, c1_m: float, a1_m=(0.0, 0.0)):
    """A field and a hidden error for one seed: its own spot on the stage, so dose never piles up."""
    x = -4e-7 + 2e-7 * (seed % 5)
    y = -1.5e-7 if seed < 5 else 1.5e-7
    twin.move_stage([x, y, 0.0, 0.0, 0.0])
    twin.set_defocus(0.0)  # a new specimen: the defocus the last seed tuned to means nothing here
    twin.set_stigmator([0.0, 0.0])
    device.set_focus_error(c1_m, a1_m)
    return x, y


# ----------------------------------------------------------------------
# Pure routines
# ----------------------------------------------------------------------
def test_search_1d_finds_a_noisy_symmetric_peak_without_leaving_its_range():
    rng = np.random.default_rng(0)
    seen = []

    def score(x):
        seen.append(x)
        return 1.0 / (1.0 + ((x - 37e-9) / 40e-9) ** 2) + rng.normal(0, 0.005)

    found = focus.search_1d(score, 0.0, 3e-7, coarse_points=13, fine_points=9, rounds=2)
    assert abs(found["best"] - 37e-9) < 3e-9
    assert found["evaluations"] == focus.search_1d_evaluations(13, 9, 2) == len(seen)
    assert min(seen) >= -3e-7 and max(seen) <= 3e-7


def test_search_2d_finds_both_components():
    found = focus.search_2d(lambda a, b: -((a - 2e-8) ** 2 + (b + 3e-8) ** 2), (0.0, 0.0), 1e-7)
    assert found["best"] == pytest.approx([2e-8, -3e-8], abs=1e-9)
    assert found["evaluations"] == focus.search_2d_evaluations()


def test_metrics_prefer_the_sharper_image_and_ignore_brightness():
    rng = np.random.default_rng(1)
    sharp = np.zeros((64, 64))
    sharp[rng.integers(0, 64, 40), rng.integers(0, 64, 40)] = 100.0
    sharp += 10.0
    kernel = np.outer(*(2 * [np.exp(-np.arange(-6, 7) ** 2 / 8.0)]))
    blurred = np.real(np.fft.ifft2(np.fft.fft2(sharp) * np.fft.fft2(kernel / kernel.sum(), s=sharp.shape)))
    for metric in focus.METRICS:
        assert focus.sharpness(sharp, metric) > focus.sharpness(blurred, metric)
    # Normalised variance does not depend on the beam current (the gradient metrics subtract shot noise, which does).
    assert focus.sharpness(3 * sharp) == pytest.approx(focus.sharpness(sharp))
    assert focus.error_grew(1e-8, 2e-8) and not focus.error_grew(1e-8, 0.5e-8)


# ----------------------------------------------------------------------
# The twin's hidden focus truth
# ----------------------------------------------------------------------
def test_twin_hides_its_focus_error_and_blurs_with_it(fresh, tuner_twin):
    twin, device = fresh
    twin.set_fov(1e-7)
    device.set_focus_error(150e-9, (0.0, 0.0))
    assert twin.get_defocus() == 0.0  # the error is not reported
    truth = json.loads(twin.export_ground_truth())["focus"]
    assert truth["hidden_error"]["c1_m"] == pytest.approx(150e-9)
    assert truth["c1_m"] == pytest.approx(150e-9, abs=1e-12)  # the stage is at x = y = 0: no height

    def score():
        twin.unblank_beam()
        try:
            return focus.sharpness(tuner_module.read_image(twin.acquire_scanned_image(["haadf"]), "haadf"))
        finally:
            twin.blank_beam()

    blurred = score()
    twin.set_defocus(-150e-9)  # cancels the hidden error
    assert score() > 1.15 * blurred


def test_twin_autofocus_and_image_shift(fresh):
    twin, device = fresh
    device.set_focus_error(80e-9, (20e-9, -10e-9))
    twin.auto_focus()
    truth = device._focus_truth()
    assert abs(truth["c1_m"]) < 5e-9 and truth["a1_abs_m"] < 5e-9
    twin.set_image_shift([3e-9, -2e-9])
    assert list(twin.get_image_shift()) == pytest.approx([3e-9, -2e-9])
    with pytest.raises(tango.DevFailed):
        twin.set_image_shift([1.0, 0.0])
    twin.set_image_shift([0.0, 0.0])


# ----------------------------------------------------------------------
# Done when: 10 of 10 seeds within 5 nm, within the declared dose
# ----------------------------------------------------------------------
def test_image_metric_focuses_ten_of_ten_seeds_from_200_nm(fresh, tuner_proxy):
    twin, device = fresh
    residuals, doses = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        c1 = (1 if seed % 2 else -1) * 200e-9 + rng.normal(0.0, 10e-9)
        x, y = setup_error(twin, device, seed, c1)
        dose_before = float(device._dose.dose_at(x, y))
        reserved, status, results = run(tuner_proxy, FOCUS_PLAN)
        assert status["state"] == "done", (seed, status["last_error"])
        tune = results["summary"]["tune"]
        residual = device._focus_truth()["c1_m"]
        residuals.append(residual)
        doses.append(tune["dose_e_per_a2"])
        # Within the declared dose: the job's count, its reserve-time worst case and the twin's own dose map agree.
        assert tune["dose_e_per_a2"] <= FOCUS_PLAN["dose_budget_e_per_a2"]
        assert tune["dose_e_per_a2"] <= reserved["estimate"]["max_dose_e_per_a2"] + 1e-9
        assert float(device._dose.dose_at(x, y)) - dose_before == pytest.approx(tune["dose_e_per_a2"], rel=0.05)
        assert tune["sharpness_after"] > tune["sharpness_before"]
        assert json.loads(twin.get_parameters())["beam_blanked"] is True
    print("residual C1 (nm):", [round(r * 1e9, 2) for r in residuals], "dose (e/A^2):", round(max(doses), 1))
    assert max(abs(r) for r in residuals) < 5e-9, [r * 1e9 for r in residuals]


def test_image_metric_corrects_stigmation_where_the_microscope_has_a_stigmator(fresh, tuner_proxy):
    twin, device = fresh
    setup_error(twin, device, 3, 120e-9, (30e-9, -20e-9))
    plan = {**FOCUS_PLAN, "corrections": ["focus", "stigmation"], "dose_budget_e_per_a2": 5000.0, "max_steps": 120}
    _reserved, status, results = run(tuner_proxy, plan)
    assert status["state"] == "done", status
    truth = device._focus_truth()
    assert abs(truth["c1_m"]) < 5e-9 and truth["a1_abs_m"] < 5e-9, truth
    assert results["summary"]["tune"]["dose_e_per_a2"] <= plan["dose_budget_e_per_a2"]


def test_optistem_strategy_runs_the_vendor_autofocus(fresh, tuner_proxy):
    twin, device = fresh
    setup_error(twin, device, 6, -180e-9, (15e-9, 5e-9))
    _reserved, status, results = run(tuner_proxy, {"strategy": "optistem", "corrections": ["focus", "stigmation"], "dose_budget_e_per_a2": 1000.0})
    assert status["state"] == "done", status
    tune = results["summary"]["tune"]
    assert tune["steps"] == 3 and tune["images"] == 2  # sharpness before and after, and one vendor run
    assert abs(device._focus_truth()["c1_m"]) < 5e-9


def test_corrector_strategy_converges_on_a_noisy_corrector(fresh, tuner_proxy, tango_ctx):
    twin, device = fresh
    setup_error(twin, device, 7, 0.0)
    noisy = corrector(tango_ctx, "tuner")
    noisy.set_aberrations_coeff_sim(json.dumps({"C1": [60e-9], "A1": [25e-9, -15e-9]}))
    _reserved, status, results = run(tuner_proxy, {"strategy": "corrector", "corrections": ["focus", "stigmation"], "dose_budget_e_per_a2": 3000.0})
    assert status["state"] == "done", status
    truth = json.loads(noisy.get_aberrations_coeff_sim())
    assert abs(truth["C1"][0]) < 5e-9 and float(np.hypot(*truth["A1"])) < 5e-9, truth
    assert results["summary"]["tune"]["strategy_result"]["iterations"] >= 1


# ----------------------------------------------------------------------
# Done when: a wrong-sign corrector aborts and restores the snapshot
# ----------------------------------------------------------------------
def test_a_wrong_sign_corrector_stops_and_restores_the_snapshot(fresh, wrongsign_tuner_proxy, tango_ctx):
    twin, _device = fresh
    wrong = corrector(tango_ctx, "wrongsign")
    start = {"C1": [50e-9], "A1": [10e-9, 5e-9]}
    wrong.set_aberrations_coeff_sim(json.dumps(start))
    twin.set_defocus(7e-9)
    _reserved, status, results = run(wrongsign_tuner_proxy, {"strategy": "corrector", "corrections": ["focus", "stigmation"], "dose_budget_e_per_a2": 3000.0})
    assert status["state"] == "failed"
    assert "error_grew" in status["last_error"]["message"]
    tune = results["summary"]["tune"]
    assert tune["outcome"] == "failed" and "corrector_C1_undone_m" in tune["restored"] and "restore_errors" not in tune
    after = json.loads(wrong.get_aberrations_coeff_sim())
    assert after["C1"] == pytest.approx(start["C1"], abs=1e-15)
    assert after["A1"] == pytest.approx(start["A1"], abs=1e-15)
    assert twin.get_defocus() == pytest.approx(7e-9)  # the microscope's own focus untouched
    assert tune["steps"] == 3  # sharpness, one measurement, one correction measured: stopped at the first growth
    assert json.loads(twin.get_parameters())["beam_blanked"] is True


# ----------------------------------------------------------------------
# Done when: an over-budget plan is refused before the beam is unblanked
# ----------------------------------------------------------------------
def test_over_budget_plan_is_refused_before_the_beam_is_unblanked(fresh, tuner_proxy):
    twin, device = fresh
    before = (len(device._acquisitions), device._dose.total_electrons(), json.loads(twin.get_parameters())["beam_blanked"], twin.get_defocus())
    per_frame = frame_dose(50.0, 2e-6, 1e-7, 256)
    reply = call(tuner_proxy, "reserve", {"plan": {**FOCUS_PLAN, "dose_budget_e_per_a2": 10 * per_frame}})
    assert reply["error"]["code"] == "plan_rejected"
    assert "above the budget" in reply["error"]["message"] and "33 frames" in reply["error"]["message"]
    after = (len(device._acquisitions), device._dose.total_electrons(), json.loads(twin.get_parameters())["beam_blanked"], twin.get_defocus())
    assert after == before  # nothing imaged, no dose, the beam still blanked, nothing moved
    # Bad plans are refused by name, too.
    assert "unknown plan field" in call(tuner_proxy, "reserve", {"plan": {**FOCUS_PLAN, "focus_rnage_m": 1e-7}})["error"]["message"]
    assert "dose_budget_e_per_a2 is required" in call(tuner_proxy, "reserve", {"plan": {"strategy": "optistem"}})["error"]["message"]
    assert "unknown strategy" in call(tuner_proxy, "reserve", {"plan": {**FOCUS_PLAN, "strategy": "magic"}})["error"]["message"]


def test_a_strategy_that_exceeds_its_declared_dose_is_stopped_and_restored(fresh, tuner_proxy):
    twin, device = fresh

    @register_strategy
    class Greedy(Strategy):
        """Declares two images, then keeps imaging: the context stops it at the budget."""

        name = "greedy_test"

        def exposures(self, plan, frames_per):
            return 2

        def run(self, ctx):
            ctx.set_defocus(ctx.snapshot["defocus_m"] + 5e-8)
            while True:
                ctx.sharpness()

    try:
        twin.set_defocus(1e-8)
        per_frame = frame_dose(50.0, 2e-6, 1e-7, 256)
        _reserved, status, results = run(tuner_proxy, {"strategy": "greedy_test", "dose_budget_e_per_a2": 4.5 * per_frame})
        assert status["state"] == "failed" and "dose_budget" in status["last_error"]["message"]
        assert results["summary"]["tune"]["images"] == 4  # the fifth frame would pass the budget
        assert twin.get_defocus() == pytest.approx(1e-8)
    finally:
        STRATEGIES.unregister("greedy_test")


def test_abort_mid_search_restores_the_snapshot(fresh, tuner_proxy):
    twin, device = fresh
    setup_error(twin, device, 2, 150e-9)
    twin.set_defocus(2e-8)
    reserved = call(tuner_proxy, "reserve", {"plan": FOCUS_PLAN})
    job_id = call(tuner_proxy, "start", {"reservation_id": reserved["reservation_id"]})["job_id"]
    deadline = time.time() + 30
    while time.time() < deadline and "measurements" not in str(call(tuner_proxy, "job_status", {"job_id": job_id}).get("phase")):
        time.sleep(0.01)
    call(tuner_proxy, "abort", {"job_id": job_id})
    status = wait(tuner_proxy, job_id)
    assert status["state"] == "aborted"
    assert twin.get_defocus() == pytest.approx(2e-8)
    assert json.loads(twin.get_parameters())["beam_blanked"] is True
    call(tuner_proxy, "release", {"reservation_id": reserved["reservation_id"]})


# ----------------------------------------------------------------------
# Done when: a 40-line example strategy registered from a test works
# ----------------------------------------------------------------------
class HillClimb(Strategy):
    """Step the defocus while the image gets sharper; turn round and halve the step when it does not."""

    name = "hill_climb_example"
    corrections = frozenset({"focus"})
    TRIALS = 40

    def exposures(self, plan, frames_per):
        return self.TRIALS

    def run(self, ctx):
        centre, half = ctx.snapshot["defocus_m"], float(ctx.plan["focus_range_m"])
        here, step, direction = centre, half / 3.0, 1.0
        best = ctx.sharpness_at(defocus=here)
        for _ in range(self.TRIALS - 1):
            if step < 2e-9:
                break
            trial = here + direction * step
            if abs(trial - centre) > half:  # stay inside the plan's range
                direction, step = -direction, step / 2.0
                continue
            score = ctx.sharpness_at(defocus=trial)
            if score > best:
                here, best = trial, score
            else:
                direction, step = -direction, step / 2.0
        ctx.set_defocus(here)
        return {"defocus_m": here, "sharpness": best}


def test_a_forty_line_example_strategy_registered_from_a_test(fresh, tuner_proxy):
    twin, device = fresh
    register_strategy(HillClimb)
    try:
        setup_error(twin, device, 4, -160e-9)
        _reserved, status, results = run(tuner_proxy, {"strategy": "hill_climb_example", "dose_budget_e_per_a2": 2000.0})
        assert status["state"] == "done", status
        tune = results["summary"]["tune"]
        assert tune["sharpness_after"] > tune["sharpness_before"]
        assert abs(device._focus_truth()["c1_m"]) < 30e-9  # a crude search, but far better than 160 nm
    finally:
        STRATEGIES.unregister("hill_climb_example")


# ----------------------------------------------------------------------
# Through the MCP server's run_job and its approval gate
# ----------------------------------------------------------------------
JSON_REQUEST = type("CommandInfo", (), {
    "in_type": tango.CmdArgType.DevString, "out_type": tango.CmdArgType.DevString,
    "in_type_desc": ':param request: JSON {"plan": {...}, "reservation_id": "..."}.', "out_type_desc": "",
})()


def test_run_job_tuner_goes_through_the_approval_gate(fresh, tuner_proxy, approvals_proxy, monkeypatch, tmp_path):
    twin, device = fresh
    setup_error(twin, device, 8, 90e-9)
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer(
        "test", "localhost", 1234, blocked_classes=[], blocked_functions={}, data_device_address="asyncroscopy/data/default",
        verbose=False, endpoint_role="agent", approvals_device_address="asyncroscopy/approvals/default", trace_path=str(tmp_path / "trace.jsonl"),
    )
    server._approvals = lambda: approvals_proxy
    server._envelope_prefixes = {"Tuner"}
    server._devices = {"Tuner": tuner_proxy}
    server.tools = {"Tuner": {
        verb: server._create_wrapper(getattr(tuner_proxy, verb), JSON_REQUEST, verb, "Tuner", tool_prefix="Tuner") for verb in ("reserve", "start")
    }}
    plan = {"strategy": "optistem", "dose_budget_e_per_a2": 1000.0}
    approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
    try:
        with pytest.raises(ToolError, match="APPROVAL REQUIRED") as held:
            server.run_job("Tuner", plan)
        assert json.loads(twin.get_parameters())["beam_blanked"] is True and abs(device._focus_truth()["c1_m"]) > 80e-9  # nothing ran
        request_id = str(held.value).split("request_id=")[1].split(")")[0]
        approvals_proxy.approve(json.dumps({"request_id": request_id, "who": "operator"}))
        reply = server.run_job("Tuner", plan)
        status = wait(tuner_proxy, reply["job_id"])
        assert status["state"] == "done", status
        assert abs(device._focus_truth()["c1_m"]) < 5e-9
        call(tuner_proxy, "release", {"reservation_id": reply["reservation_id"]})
    finally:
        approvals_proxy.set_mode(json.dumps({"mode": "auto", "who": "test-teardown"}))


def test_a_dry_run_tune_is_refused_at_reserve_by_name(tuner_proxy):
    # Every measurement reads an image or a corrector reply, which a dry run does not produce.
    refused = call(tuner_proxy, "reserve", {"plan": {**FOCUS_PLAN, "dry_run": True}})
    assert refused["error"]["code"] == "dry_run_unsupported" and "Tuner does not support dry runs" in refused["error"]["message"]
    reserved = call(tuner_proxy, "reserve", {"plan": FOCUS_PLAN})  # nothing was held: a real tune reserves at once
    assert "reservation_id" in reserved, reserved
    call(tuner_proxy, "release", {"reservation_id": reserved["reservation_id"]})


def test_auto_mode_starts_a_tune_only_when_it_drives_nothing_held(fresh, tuner_proxy, approvals_proxy, monkeypatch, tmp_path):
    from asyncroscopy.safety.policy import risk_of

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer(
        "test", "localhost", 1234, blocked_classes=[], blocked_functions={}, data_device_address="asyncroscopy/data/default",
        verbose=False, endpoint_role="agent", approvals_device_address="asyncroscopy/approvals/default", trace_path=str(tmp_path / "trace.jsonl"),
    )
    server._approvals = lambda: approvals_proxy

    class Named:
        def __init__(self, name):
            self.name = name

        def dev_name(self):
            return self.name

    server._envelope_prefixes = {"Tuner"}
    server._stem_prefixes = {"DigitalTwin"}
    server._devices = {"Tuner": tuner_proxy, "DigitalTwin": Named("asyncroscopy/digitaltwin/tuner"), "Corrector": Named("asyncroscopy/corrector/tuner")}
    for prefix, commands in (("DigitalTwin", ("blank_beam", "unblank_beam", "set_fov", "set_defocus", "set_stigmator", "acquire_scanned_image", "auto_focus")),
                             ("Corrector", ("measure_c1a1", "correct_aberration"))):
        for command in commands:
            server.tool_risks[f"{prefix}_{command}"] = risk_of(prefix, command, stem=prefix == "DigitalTwin")
    server.tools = {"Tuner": {
        verb: server._create_wrapper(getattr(tuner_proxy, verb), JSON_REQUEST, verb, "Tuner", tool_prefix="Tuner") for verb in ("reserve", "start")
    }}
    reply = server.run_job("Tuner", {"strategy": "optistem", "dose_budget_e_per_a2": 1000.0})
    assert wait(tuner_proxy, reply["job_id"])["state"] == "done"
    call(tuner_proxy, "release", {"reservation_id": reply["reservation_id"]})
    with pytest.raises(ToolError, match="Corrector_correct_aberration"):
        server.run_job("Tuner", {"strategy": "corrector", "corrections": ["focus", "stigmation"], "dose_budget_e_per_a2": 3000.0})

    request_id = next(p["request_id"] for p in json.loads(approvals_proxy.list_pending()) if p["action"] == "Tuner_start")
    approvals_proxy.deny(json.dumps({"request_id": request_id, "who": "test"}))
    assert server.approval_status(request_id)["state"] == "denied"  # frees the held reservation
