"""Dose-aware acquisition: a measured beam current with its provenance, and a budget that counts EDS.

The "done when" checks of item 3 in docs/designs/concepts_brief.md:
- the twin's DoseMap ends within 5 % of the plan's estimate, with EDS spectra included;
- an over-budget plan is refused at reserve, before anything is unblanked;
- AutoScript (on the recorded fake client) refuses a screen-current root outside its calibration;
- a plan with a dose budget on a stack that is not simulated is refused when the current is assumed.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from autoscript_core.common import ApplicationServerException

import asyncroscopy.instruments.electron_microscope.auto_script as auto_script_module
from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope as EM
from asyncroscopy.instruments.electron_microscope.mapper import Mapper, frame_dose
from tests.fake_autoscript import FakeAutoScriptClient, screen_current_a
from tests.test_parity import autoscript_device


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def wait(mapper, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = call(mapper, "job_status", {"job_id": job_id})
        if status["state"] in ("done", "failed", "aborted"):
            return status
        time.sleep(0.05)
    raise TimeoutError(status)


# 2 x 2 tiles of 160 nm that abut (each point is imaged once), 50 A pixels, and a 1 ms spectrum at each
# tile centre: about 12.5 e/A^2 from the frame and 31 e/A^2 from the spectrum (over a 10 nm square).
EDS_PLAN = {
    "box_m": [-1.6e-7, 1.6e-7, -1.6e-7, 1.6e-7], "fov_m": 1.6e-7, "overlap": 0.0, "imsize": 32, "dwell_s": 1e-4,
    "per_tile": ["eds_point"], "eds_exposure_s": 1e-3, "stitch": False,
}


def twin_state(twin) -> dict:
    truth = json.loads(twin.export_ground_truth())
    return {"acquisitions": len(truth["acquisitions"]), "dose": truth["dose"]["max_dose_e_per_a2"], "blanked": json.loads(twin.get_parameters())["beam_blanked"]}


def test_over_budget_plan_is_refused_before_any_unblank(dose_mapper_proxy, dose_twin_proxy):
    frame_only = frame_dose(50.0, EDS_PLAN["dwell_s"], EDS_PLAN["fov_m"], EDS_PLAN["imsize"])
    before = twin_state(dose_twin_proxy)
    # Above the frames' dose, below frames plus spectra: only counting EDS refuses it.
    reply = call(dose_mapper_proxy, "reserve", {"plan": {**EDS_PLAN, "dose_budget_e_per_a2": 2 * frame_only}})
    assert reply["error"]["code"] == "plan_rejected"
    assert "from EDS spectra" in reply["error"]["message"]
    assert twin_state(dose_twin_proxy) == before  # nothing acquired, no dose, the beam never left its state


def test_twin_dose_map_matches_the_plan_estimate_with_eds(dose_mapper_proxy, dose_twin_proxy):
    assert twin_state(dose_twin_proxy)["dose"] == 0.0, "this is the only test that exposes the dose twin"
    reserve = call(dose_mapper_proxy, "reserve", {"plan": {**EDS_PLAN, "dose_budget_e_per_a2": 100.0}})
    assert "estimate" in reserve, reserve
    estimate = reserve["estimate"]
    assert estimate["beam_current_source"] == "measured" and estimate["eds_dose_e_per_a2"] > 0
    started = call(dose_mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    status = wait(dose_mapper_proxy, started["job_id"])
    assert status["state"] == "done", status
    call(dose_mapper_proxy, "release", {"reservation_id": reserve["reservation_id"]})

    delivered = twin_state(dose_twin_proxy)["dose"]
    planned = estimate["max_dose_e_per_a2"]
    assert delivered == pytest.approx(planned, rel=0.05)
    frames_only = planned - estimate["eds_dose_e_per_a2"]
    assert abs(delivered - frames_only) / delivered > 0.05  # without EDS the estimate would miss


# ----------------------------------------------------------------------
# AutoScript: measured current, and the screen-current root checked before it is written
# ----------------------------------------------------------------------
@pytest.fixture
def autoscript(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_script_module, "SCREEN_SETTLE_S", 0.0)
    client = FakeAutoScriptClient()
    return autoscript_device(client, tmp_path), client


def test_autoscript_reports_default_current_until_the_screen_measures_it(autoscript):
    device, client = autoscript
    parameters = json.loads(EM.get_parameters(device))
    assert parameters["beam_current_source"] == "default" and "screen" in parameters["beam_current_reason"]
    client.detectors.screen.insert()
    measured = EM.get_screen_current(device)
    client.detectors.screen.retract()
    parameters = json.loads(EM.get_parameters(device))
    assert parameters["beam_current_source"] == "measured"
    assert parameters["beam_current_pa"] == pytest.approx(measured) == pytest.approx(screen_current_a(80.0) * 1e12)


def test_fake_refuses_an_out_of_range_screen_current_root(autoscript):
    device, client = autoscript
    with pytest.raises(ValueError, match="insert the screen"):  # a retracted screen measures nothing to fit
        EM.calibrate_screen_current(device)
    client.detectors.screen.insert()
    EM.calibrate_screen_current(device)
    focus_writes = len(client.writes("optics.monochromator.focus"))

    # 250 pA needs about 180 V; the calibration spans 10-150 V (up to about 171 pA), so the polynomial's root
    # there is extrapolation. The fake gun would take 180 V; the device refuses before writing.
    with pytest.raises(ValueError, match="outside the calibrated range"):
        EM.set_screen_current(device, 250.0)
    with pytest.raises(ValueError, match="outside the calibrated range"):
        EM.set_screen_current(device, -5.0)
    assert len(client.writes("optics.monochromator.focus")) == focus_writes  # nothing reached the gun

    EM.set_screen_current(device, 100.0)
    written = client.writes("optics.monochromator.focus")[-1]["value"]
    assert 10.0 <= written <= 150.0
    assert EM.get_screen_current(device) == pytest.approx(100.0, rel=0.02)

    # And the instrument itself refuses a focus outside its range, as AutoScript's server does.
    with pytest.raises(ApplicationServerException):
        client.optics.monochromator.focus = 1e3


# ----------------------------------------------------------------------
# A dose budget needs a measured current unless the stack is simulated
# ----------------------------------------------------------------------
class RecordingMicroscope:
    def __init__(self, parameters=None, device=None):
        self.calls: list[str] = []
        self.parameters, self.device = parameters, device

    def get_parameters(self):
        self.calls.append("get_parameters")
        return EM.get_parameters(self.device) if self.device is not None else json.dumps(self.parameters)

    def __getattr__(self, name):  # any other command (move_stage, unblank_beam, ...) is recorded too
        def record(*args):
            self.calls.append(name)

        return record


def mapper_stub(microscope, **extra):
    stub = SimpleNamespace(
        microscope_device_address="m", analysis_url="", max_tiles=10_000, estimate_move_s=1.0,
        default_beam_current_pa=50.0, eds_dose_spread_m=1e-8, proxy=lambda address, timeout_s=None: microscope, **extra,
    )
    for name in ("_beam_current", "_stack_simulated", "_refuse_assumed_current"):
        setattr(stub, name, getattr(Mapper, name).__get__(stub))
    return stub


PLAN = {"box_m": [0.0, 4e-7, 0.0, 4e-7], "fov_m": 2e-7, "imsize": 32, "stitch": False}


def test_budgeted_plan_with_an_assumed_current_is_refused_on_a_stack_that_is_not_simulated(monkeypatch):
    monkeypatch.delenv("ASYNCROSCOPY_SIMULATED", raising=False)
    microscope = RecordingMicroscope({"defocus_m": 0.0})  # reports no current, and does not say it is simulated
    with pytest.raises(ValueError, match="beam current is assumed"):
        Mapper.resolve_plan(mapper_stub(microscope), {**PLAN, "dose_budget_e_per_a2": 1e3})
    assert microscope.calls == ["get_parameters"]  # refused at reserve: nothing moved, nothing unblanked

    plan = Mapper.resolve_plan(mapper_stub(microscope), PLAN)  # without a budget the estimate says what it assumed
    assert plan["estimate"]["beam_current_source"] == "default" and plan["estimate"]["beam_current_reason"]

    # A simulated stack may budget with an assumed current: the `simulated` flag, or the environment hook.
    Mapper.resolve_plan(mapper_stub(microscope, simulated=True), {**PLAN, "dose_budget_e_per_a2": 1e3})
    monkeypatch.setenv("ASYNCROSCOPY_SIMULATED", "1")
    Mapper.resolve_plan(mapper_stub(microscope), {**PLAN, "dose_budget_e_per_a2": 1e3})
    with pytest.raises(ValueError, match="beam current is assumed"):
        Mapper.resolve_plan(mapper_stub(microscope, simulated=False), {**PLAN, "dose_budget_e_per_a2": 1e3})


def test_autoscript_budgeted_plan_needs_the_screen_current(autoscript, monkeypatch):
    monkeypatch.delenv("ASYNCROSCOPY_SIMULATED", raising=False)
    device, client = autoscript
    microscope = RecordingMicroscope(device=device)
    with pytest.raises(ValueError, match="beam current is assumed"):
        Mapper.resolve_plan(mapper_stub(microscope), {**PLAN, "dose_budget_e_per_a2": 1e3})
    assert client.writes("optics.blanker") == [] and client.writes("specimen.stage.absolute_move") == []

    client.detectors.screen.insert()
    EM.get_screen_current(device)
    client.detectors.screen.retract()
    plan = Mapper.resolve_plan(mapper_stub(microscope), {**PLAN, "dose_budget_e_per_a2": 1e3})
    assert plan["estimate"]["beam_current_source"] == "measured"
    assert plan["estimate"]["beam_current_pa"] == pytest.approx(screen_current_a(80.0) * 1e12)
