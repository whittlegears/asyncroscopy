"""Hardware-safety rules found by an adversarial review of the STEM stack (no Tango server needed).

* A screen-current reading goes stale when the optics that set the current change, or with age;
  a dose budget on hardware refuses a default or stale current (routines/beam.py).
* OptiSTEM is refused where the stigmator cannot be restored, its result is checked against the
  plan's ranges, and a failure restores the defocus and stigmator.
* A resumed Tuner job first undoes corrector corrections an earlier attempt left outstanding.
* A dry run records or refuses every DeviceProxy write, and passes reads by the safety policy.
* A device may not contradict its stack's simulated flag.
* A LAN address counts as a remote bind.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
import yaml

import asyncroscopy.instruments.electron_microscope.auto_script as auto_script_module
from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope as EM
from asyncroscopy.jobs.tuning.tuner import TuneContext, TuneFailed, Tuner
from asyncroscopy.jobs import JobDevice
from asyncroscopy.jobs.recording_proxy import DRY_RUN_PLACEHOLDER, DryRunRefused, RecordingProxy
from asyncroscopy.mcp.binding import bind_host, is_loopback_host, is_remote_host
from asyncroscopy.routines.beam import DEFAULT_MAX_BEAM_CURRENT_AGE_S, BeamCurrentRefused, beam_current, usable_beam_current
from asyncroscopy.safety.policy import is_read_only
from startup_scripts import run_servers
from tests.fake_autoscript import FakeAutoScriptClient
from tests.test_parity import autoscript_device


# ----------------------------------------------------------------------
# 1. The beam current goes stale
# ----------------------------------------------------------------------
MEASURED = {"beam_current_pa": 40.0, "beam_current_source": "measured", "beam_current_age_s": 10.0}


def test_usable_beam_current_rules():
    assert usable_beam_current(MEASURED, simulated=False).pa == 40.0
    with pytest.raises(BeamCurrentRefused, match="old"):
        usable_beam_current({**MEASURED, "beam_current_age_s": DEFAULT_MAX_BEAM_CURRENT_AGE_S + 1}, simulated=False)
    with pytest.raises(BeamCurrentRefused, match="no age"):
        usable_beam_current({k: v for k, v in MEASURED.items() if k != "beam_current_age_s"}, simulated=False)
    with pytest.raises(BeamCurrentRefused, match="is assumed"):  # a missing source is a default, not a measurement
        usable_beam_current({"beam_current_pa": 40.0}, simulated=False)
    with pytest.raises(BeamCurrentRefused, match="is assumed"):
        usable_beam_current({"beam_current_pa": 40.0, "beam_current_source": "default"}, simulated=False)
    with pytest.raises(BeamCurrentRefused, match="is assumed"):
        usable_beam_current({}, simulated=False)
    assert usable_beam_current({**MEASURED, "beam_current_age_s": 100.0}, simulated=False, max_age_s=50.0 * 3).pa == 40.0
    # A simulated stack may budget with its twin's current, or an assumed one; never with a current <= 0.
    assert usable_beam_current({}, simulated=True, default_pa=50.0).source == "default"
    for bad in (0.0, -5.0, float("nan"), "x"):
        with pytest.raises(BeamCurrentRefused):
            usable_beam_current({**MEASURED, "beam_current_pa": bad}, simulated=True)
    assert beam_current({}, simulated=False).refused  # the non-raising form says why


@pytest.fixture
def autoscript(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_script_module, "SCREEN_SETTLE_S", 0.0)
    client = FakeAutoScriptClient()
    return autoscript_device(client, tmp_path), client


def measure(device, client):
    client.detectors.screen.insert()
    EM.get_screen_current(device)
    client.detectors.screen.retract()
    parameters = json.loads(EM.get_parameters(device))
    assert parameters["beam_current_source"] == "measured", parameters
    return parameters


@pytest.mark.parametrize("change, what", [
    (lambda c: setattr(c.optics.monochromator, "focus", 120.0), "monochromator_focus_v"),
    (lambda c: c.optics._state.__setitem__("spot_size_index", 3), "spot_size"),  # set at the console, not through us
    (lambda c: setattr(c.optics, "optical_mode", "Tem"), "optical_mode"),
])
def test_a_change_to_the_optics_makes_the_measured_current_stale(autoscript, change, what):
    device, client = autoscript
    measure(device, client)
    change(client)
    parameters = json.loads(EM.get_parameters(device))
    assert parameters["beam_current_source"] == "default" and what in parameters["beam_current_reason"]
    with pytest.raises(BeamCurrentRefused, match="is assumed"):
        usable_beam_current(parameters, simulated=False)


def test_set_screen_current_clears_the_measurement(autoscript):
    device, client = autoscript
    measure(device, client)
    client.detectors.screen.insert()
    EM.calibrate_screen_current(device)
    assert json.loads(EM.get_parameters(device))["beam_current_source"] == "default"
    EM.get_screen_current(device)
    assert json.loads(EM.get_parameters(device))["beam_current_source"] == "measured"
    with pytest.raises(ValueError):
        EM.set_screen_current(device, 1e6)  # refused, but the old reading is dropped all the same
    assert json.loads(EM.get_parameters(device))["beam_current_source"] == "default"


def test_a_stale_measured_current_is_refused_by_the_tuner_on_hardware(autoscript, monkeypatch):
    device, client = autoscript
    measure(device, client)
    microscope = FakeTunerMicroscope(parameters_from=lambda: EM.get_parameters(device))
    tuner = tuner_stub(microscope, simulated=False)
    plan = {"strategy": "image_metric", "dose_budget_e_per_a2": 1e4}
    assert Tuner.resolve_plan(tuner, plan)["estimate"]["beam_current_source"] == "measured"
    device._measured_beam_current["at"] -= DEFAULT_MAX_BEAM_CURRENT_AGE_S + 60  # 16 minutes later
    with pytest.raises(ValueError, match="old"):
        Tuner.resolve_plan(tuner, plan)
    measure(device, client)
    client.optics.monochromator.focus = 150.0  # someone raises the current after the reading
    with pytest.raises(ValueError, match="optics changed"):
        Tuner.resolve_plan(tuner, plan)


# ----------------------------------------------------------------------
# A Tuner without a Tango server
# ----------------------------------------------------------------------
class FakeTunerMicroscope:
    def __init__(self, stigmator=True, parameters_from=None, autofocus_to=None):
        self.defocus, self.stig, self.fov = 0.0, [0.0, 0.0], 2e-7
        self.calls: list = []
        self.has_stigmator = stigmator
        self.parameters_from = parameters_from
        self.autofocus_to = autofocus_to

    def command_list_query(self):
        names = ["get_defocus", "set_defocus", "auto_focus", "blank_beam", "unblank_beam", "get_parameters"]
        if self.has_stigmator:
            names += ["get_stigmator", "set_stigmator"]
        return [SimpleNamespace(cmd_name=n) for n in names]

    def get_parameters(self):
        if self.parameters_from:
            return self.parameters_from()
        return json.dumps({"beam_current_pa": 50.0, "beam_current_source": "measured", "beam_current_age_s": 1.0})

    def get_defocus(self):
        return self.defocus

    def set_defocus(self, value):
        self.calls.append(("set_defocus", value))
        self.defocus = float(value)

    def get_stigmator(self):
        return list(self.stig)

    def set_stigmator(self, value):
        self.calls.append(("set_stigmator", list(value)))
        self.stig = [float(v) for v in value]

    def get_fov(self):
        return self.fov

    def set_fov(self, value):
        self.calls.append(("set_fov", value))
        self.fov = value

    def blank_beam(self):
        self.calls.append(("blank_beam",))

    def unblank_beam(self):
        self.calls.append(("unblank_beam",))

    def auto_focus(self):
        self.calls.append(("auto_focus",))
        self.defocus, self.stig = self.autofocus_to


class FakeCorrector:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def correct_aberration(self, command):
        if self.fail:
            raise RuntimeError("corrector not answering")
        self.sent.append(command)
        return json.dumps({"result": {}})


def tuner_stub(microscope, simulated=True, corrector=None, **extra):
    """Tuner's own methods on a plain object (a Tango device cannot be built without a server)."""
    methods = {name: value for name, value in vars(Tuner).items() if callable(value) or isinstance(value, staticmethod)}
    methods["_stack_simulated"] = JobDevice._stack_simulated  # the stack's simulated flag, read as every job device reads it
    cls = type("TunerStub", (), methods)
    stub = cls()
    scan = SimpleNamespace(imsize=64, dwell_time=1e-6, scan_region=[0.0, 0.0, 1.0, 1.0])
    devices = {"m": microscope, "s": scan, "c": corrector}
    saved = []
    attrs = dict(
        microscope_device_address="m", scan_device_address="s", corrector_device_address="c" if corrector else "",
        data_device_address="", default_beam_current_pa=50.0, autofocus_frames=8, corrector_measure_frames=4,
        allow_corrector_writes=False, vendor_dose_calibrated=False, max_beam_current_age_s=DEFAULT_MAX_BEAM_CURRENT_AGE_S,
        simulated=simulated, _lock=threading.RLock(), saved=saved,
        proxy=lambda address, timeout_s=None: devices[address], _save=lambda job: saved.append(json.loads(json.dumps(job.get("tune")))),
        aborting=lambda job: False, scan=scan, **extra,
    )
    for name, value in attrs.items():
        setattr(stub, name, value)
    return stub


# ----------------------------------------------------------------------
# 2. OptiSTEM and the stigmator; its result checked against the plan
# ----------------------------------------------------------------------
def test_optistem_is_refused_where_the_stigmator_cannot_be_restored():
    tuner = tuner_stub(FakeTunerMicroscope(stigmator=False))
    with pytest.raises(ValueError, match="stigmator"):
        Tuner.resolve_plan(tuner, {"strategy": "optistem", "dose_budget_e_per_a2": 1e4})
    Tuner.resolve_plan(tuner, {"strategy": "image_metric", "dose_budget_e_per_a2": 1e4})  # focus alone still runs


def test_optistem_out_of_range_fails_and_the_stigmators_are_restored():
    microscope = FakeTunerMicroscope(autofocus_to=(5e-7, [2e-8, -1e-8]))  # 500 nm: beyond focus_range_m 300 nm
    microscope.defocus, microscope.stig = 1e-8, [3e-9, 4e-9]
    tuner = tuner_stub(microscope)
    job = {"resolved_plan": Tuner.resolve_plan(tuner, {"strategy": "optistem", "corrections": ["focus", "stigmation"], "dose_budget_e_per_a2": 1e4})}
    Tuner.on_start(tuner, job)
    with pytest.raises(TuneFailed, match="out_of_range"):
        TuneContext(tuner, job).auto_focus()
    Tuner.on_stop(tuner, job, "failed")
    assert microscope.defocus == pytest.approx(1e-8) and microscope.stig == pytest.approx([3e-9, 4e-9])
    assert job["tune"]["restored"]["stigmator_m"] == pytest.approx([3e-9, 4e-9])
    assert microscope.calls[microscope.calls.index(("auto_focus",)) + 1] == ("blank_beam",)  # blanked straight after the run


def test_vendor_and_corrector_strategies_are_refused_on_hardware_until_their_dose_is_calibrated():
    tuner = tuner_stub(FakeTunerMicroscope(), simulated=False)
    with pytest.raises(ValueError, match="vendor_dose_calibrated"):
        Tuner.resolve_plan(tuner, {"strategy": "optistem", "dose_budget_e_per_a2": 1e4})
    tuner.vendor_dose_calibrated = True
    estimate = Tuner.resolve_plan(tuner, {"strategy": "optistem", "dose_budget_e_per_a2": 1e4})["estimate"]
    assert "own settings" in estimate["dose_assumption"] and "vendor_dose_calibrated" in estimate["dose_assumption"]
    assert "plan's" in Tuner.resolve_plan(tuner, {"strategy": "image_metric", "dose_budget_e_per_a2": 1e4})["estimate"]["dose_assumption"]


# ----------------------------------------------------------------------
# 3. A resume undoes what an earlier attempt left outstanding
# ----------------------------------------------------------------------
def earlier_attempt(tuner, outstanding):
    plan = Tuner.resolve_plan(tuner, {"strategy": "corrector", "corrections": ["focus"], "dose_budget_e_per_a2": 1e4})
    snapshot = {"defocus_m": 0.0, "stigmator_m": [0.0, 0.0], "fov_m": 2e-7, "imsize": 64, "dwell_s": 1e-6, "scan_region": [0.0, 0.0, 1.0, 1.0]}
    return {"resolved_plan": plan, "tune": {
        "strategy": "corrector", "outcome": "failed", "snapshot": snapshot, "corrector_sent": outstanding,
        "restore_errors": ["corrector C1: timeout"], "history": [], "steps": 2,
    }}


def test_a_resume_undoes_outstanding_corrector_offsets_before_a_new_snapshot():
    corrector = FakeCorrector()
    microscope = FakeTunerMicroscope()
    tuner = tuner_stub(microscope, corrector=corrector)
    job = earlier_attempt(tuner, {"C1": [1.5e-7]})
    microscope.defocus = 2e-8  # where the failed attempt left it
    Tuner.on_start(tuner, job)
    # Undone in max_step_m pieces, negated, before anything else moved; the defocus put back to the old snapshot.
    assert corrector.sent == ["C1 -1.000000e-07", "C1 -5.000000e-08"]
    assert microscope.calls.index(("set_fov", 1e-7)) > max(i for i, c in enumerate(microscope.calls) if c[0] == "set_defocus")
    assert job["tune"]["corrector_sent"] == {} and job["tune"]["outcome"] == "running"
    assert job["tune"]["snapshot"]["defocus_m"] == 0.0  # the new snapshot is taken after the undo
    assert any(s and s.get("outcome") == "restored_before_resume" for s in tuner.saved)


def test_a_resume_is_refused_when_the_outstanding_offsets_cannot_be_undone():
    microscope = FakeTunerMicroscope()
    tuner = tuner_stub(microscope, corrector=FakeCorrector(fail=True))
    job = earlier_attempt(tuner, {"C1": [5e-8]})
    with pytest.raises(TuneFailed, match="resume_refused"):
        Tuner.on_start(tuner, job)
    assert ("set_fov", 1e-7) not in microscope.calls  # no new snapshot, nothing set up over it (only the old fov put back)
    assert job["tune"]["corrector_sent"] == {"C1": [5e-8]}  # still known, for the operator and the next try


def test_a_restart_mid_tune_is_restored_on_resume():
    microscope = FakeTunerMicroscope()
    tuner = tuner_stub(microscope)
    job = {"resolved_plan": Tuner.resolve_plan(tuner, {"strategy": "image_metric", "dose_budget_e_per_a2": 1e4})}
    Tuner.on_start(tuner, job)
    assert tuner.saved and tuner.saved[-1]["outcome"] == "running"  # the snapshot is on disk before the first change
    microscope.defocus = 2.5e-7  # the device died mid-search
    Tuner.on_start(tuner, job)
    assert job["tune"]["snapshot"]["defocus_m"] == 0.0


# ----------------------------------------------------------------------
# 4. Dry run: every DeviceProxy write recorded or refused
# ----------------------------------------------------------------------
class FakeProxy:
    def __init__(self):
        self.calls = []

    def get_command_list(self):
        return ["get_stage", "get_tip", "move_stage", "blank_beam"]

    def get_attribute_list(self):
        return ["imsize"]

    def info(self):
        return SimpleNamespace(dev_class="Opentrons")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.calls.append(name) or name


@pytest.mark.parametrize("method, args", [
    ("write_attributes_asynch", ([("imsize", 64)],)),
    ("write_attribute_reply", (1,)),
    ("set_attribute_config", (SimpleNamespace(name="imsize", max_value="1e9"),)),
    ("poll_attribute", ("imsize", 100)),
    ("write_pipe", ("pipe", ("blob", []))),
    ("write_attributes", ([("imsize", 64)],)),
    ("put_property", ({"x": 1},)),
])
def test_dry_run_records_every_proxy_write_and_performs_none(method, args):
    fake, recorded = FakeProxy(), []
    proxy = RecordingProxy(fake, "a/b/c", lambda *call: recorded.append(call))
    assert getattr(proxy, method)(*args) is None
    assert fake.calls == [] and len(recorded) == 1 and recorded[0][4] is False


def test_dry_run_refuses_unknown_proxy_methods_and_reads_by_policy():
    fake, recorded = FakeProxy(), []
    proxy = RecordingProxy(fake, "a/b/c", lambda *call: recorded.append(call))
    with pytest.raises(DryRunRefused):
        proxy.subscribe_event  # noqa: B018 - neither a known read nor a known write
    assert proxy.get_stage() == "get_stage" and proxy.imsize is not None
    assert proxy.move_stage([0, 0]) == DRY_RUN_PLACEHOLDER
    # A get_ command the deployment marks high is not read-only by policy: recorded, not run.
    overrides = {"opentrons.get_tip": "high"}
    assert not is_read_only("Opentrons", "get_tip", overrides) and is_read_only("Opentrons", "get_stage", overrides)
    assert not is_read_only("X", "blank_beam") and not is_read_only("X", "set_defocus")  # low risk is not read-only
    strict = RecordingProxy(fake, "a/b/c", lambda *call: recorded.append(call), risk_overrides=overrides)
    assert strict.get_tip() == DRY_RUN_PLACEHOLDER and strict.get_stage() == "get_stage"
    assert fake.calls == ["get_stage", "get_stage"]
    assert [c[1] for c in recorded] == ["move_stage", "get_tip"]


# ----------------------------------------------------------------------
# 5. A device may not contradict the stack's simulated flag
# ----------------------------------------------------------------------
def write_config(tmp_path, stack: bool, device_flag) -> object:
    raw = yaml.safe_load((run_servers.PROJECT_DIR / "configs" / "Test.yaml").read_text(encoding="utf-8"))
    raw["simulated"] = stack
    if device_flag is not None:
        raw["devices"]["scan"]["properties"] = {"simulated": device_flag}
    path = tmp_path / "stack.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_a_device_simulated_flag_that_differs_from_the_stack_is_refused(tmp_path):
    with pytest.raises(ValueError, match="top-level simulated is false"):
        run_servers.load_config(write_config(tmp_path, False, True))
    with pytest.raises(ValueError, match="top-level simulated is true"):
        run_servers.load_config(write_config(tmp_path, True, False))
    with pytest.raises(ValueError, match="true or false"):
        run_servers.load_config(write_config(tmp_path, False, "maybe"))
    config = run_servers.load_config(write_config(tmp_path, False, False))  # repeating the stack's flag is fine
    scan = next(d for d in config.support_devices if d.key == "scan")
    assert scan.properties["simulated"] == ["false"] and config.instrument.properties["simulated"] == ["false"]


# ----------------------------------------------------------------------
# 6. Binding: anything but loopback is remote
# ----------------------------------------------------------------------
def test_a_lan_ip_is_a_remote_bind():
    for host in ("192.168.1.20", "10.0.0.5", "0.0.0.0", "::", "", "my-pc", "fe80::1%eth0", "172.16.0.1"):
        assert is_remote_host(host), host
        assert bind_host(host, allow_remote_bind=False, simulated=False)[0] == "127.0.0.1", host
        assert bind_host(host, allow_remote_bind=True, simulated=False) == (host, None)
        assert bind_host(host, allow_remote_bind=False, simulated=True) == (host, None)
    for host in ("127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost", "::ffff:127.0.0.1"):
        assert is_loopback_host(host), host
        assert bind_host(host, allow_remote_bind=False, simulated=False) == (host, None)
    assert "allow_remote_bind" in bind_host("192.168.1.20", False, False)[1]
