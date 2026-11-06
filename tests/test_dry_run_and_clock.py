"""Dry run, the stack's simulated flag, and one time source (items 2 and 4 of docs/designs/concepts_brief.md).

* A dry-run map on the twin moves nothing: the twin's ground truth shows no new
  stage move or acquisition, and the trace lists every recorded move_stage,
  checked against the plan box and marked dry_run. It is approved like a real start.
* On a stack that is not simulated, Approvals refuses auto mode and the MCP
  endpoint binds 127.0.0.1 unless its config allows other machines.
* With time_scale 0 and drift on, the drift the stitch fits from tiles timed on the
  twin's clock (clock_s) matches the twin's drift, and a revisit lands on its target.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import tango
from fastmcp.exceptions import ToolError

from asyncroscopy.jobs.mapping.mapper import Mapper
from asyncroscopy.jobs.mapping.pipeline import DRY_RUN_SKIPPED, Pipeline
from asyncroscopy.jobs.recording_proxy import DRY_RUN_PLACEHOLDER, RecordingProxy
from asyncroscopy.core.net import bind_host
from dataclasses import replace

from asyncroscopy.mcp.descriptors import describe_native
from asyncroscopy.mcp.mcp_server import MCPServer, stack_simulated

ROOT = Path(__file__).resolve().parents[1]
# 3 x 3 tiles: a 300 nm box of 120 nm tiles at 15 % overlap.
DRY_PLAN = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1.2e-7, "overlap": 0.15, "imsize": 32, "dwell_s": 1e-6, "per_tile": ["eds_point"], "eds_exposure_s": 0.05}


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def wait_done(proxy, job_id, timeout=300):
    deadline = time.time() + timeout
    while True:
        status = call(proxy, "job_status", {"job_id": job_id})
        if status["state"] in {"done", "failed", "aborted"}:
            return status
        assert time.time() < deadline, status
        time.sleep(0.05)


def trace_events(path: Path, run_id: str) -> list[dict]:
    return [e for e in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if e.get("run_id") == run_id]


# ----------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------
class TestRecordingProxy:
    class Fake:
        def __init__(self):
            self.calls = []
            self.stage_moving = False

        def get_command_list(self):
            return ["move_stage", "get_stage", "blank_beam", "unblank_beam", "acquire_scanned_image"]

        def get_attribute_list(self):
            return ["stage_moving", "imsize"]

        def read_attribute(self, name):
            self.calls.append(("read_attribute", name))
            return type("Value", (), {"value": 1.0})()

        def __getattr__(self, name):
            return lambda *args: self.calls.append((name, args)) or name

    def test_reads_pass_writes_are_recorded_and_blanking_runs(self):
        fake, recorded = self.Fake(), []
        proxy = RecordingProxy(fake, "asyncroscopy/instrument/default", lambda *call: recorded.append(call), safe_commands=frozenset({"blank_beam"}))
        assert proxy.get_stage() == "get_stage"  # a read: passed through, not recorded
        assert proxy.read_attribute("clock_s").value == 1.0
        assert proxy.stage_moving is False
        assert proxy.move_stage([1e-7, 0, 0, 0, 0]) == DRY_RUN_PLACEHOLDER
        assert proxy.acquire_scanned_image(["haadf"]) == DRY_RUN_PLACEHOLDER
        proxy.imsize = 64
        proxy.write_attribute("dwell_time", 1e-6)
        assert proxy.blank_beam() == "blank_beam"  # the one write that really runs
        assert [c[0] for c in fake.calls] == ["get_stage", "read_attribute", "blank_beam"]
        assert [(c[1], c[4]) for c in recorded] == [
            ("move_stage", False), ("acquire_scanned_image", False), ("write_attribute", False), ("write_attribute", False), ("blank_beam", True),
        ]
        assert recorded[2][3] == {"attribute": "imsize", "value": 64}


def test_a_dry_run_map_moves_nothing_and_traces_every_checked_move(mapper_proxy, mapping_twin_proxy, job_dir):
    before = json.loads(mapping_twin_proxy.export_ground_truth())
    mapping_twin_proxy.unblank_beam()  # the dry run must still blank it for real

    reserved = call(mapper_proxy, "reserve", {"plan": {**DRY_PLAN, "dry_run": True}})
    assert reserved.get("dry_run") is True and reserved["estimate"]["tiles"] == 9, reserved
    assert call(mapper_proxy, "get_reservation", {"reservation_id": reserved["reservation_id"]})["dry_run"] is True
    started = call(mapper_proxy, "start", {"reservation_id": reserved["reservation_id"]})
    assert wait_done(mapper_proxy, started["job_id"])["state"] == "done"
    results = call(mapper_proxy, "results", {"job_id": started["job_id"]})
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})

    # Nothing moved, nothing was acquired, the sample saw no electron; the beam was blanked for real.
    after = json.loads(mapping_twin_proxy.export_ground_truth())
    assert len(after["moves"]) == len(before["moves"])
    assert len(after["acquisitions"]) == len(before["acquisitions"])
    assert after["simulated_time_s"] == before["simulated_time_s"]
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True

    # The summary says what happened in place of each acquisition; there is no image, spectrum or run file.
    summary = results["summary"]
    assert summary["acquisitions"] == DRY_RUN_SKIPPED and "run_file" not in summary and "spectra" not in summary
    assert "run_file" not in results["result_keys"]
    assert results["dry_run"]["recorded"] > 0 and results["dry_run"]["performed"] > 0 and results["dry_run"]["failed_checks"] == 0

    events = trace_events(job_dir / "trace" / "jobs.jsonl", reserved["reservation_id"])
    assert events and all(e.get("dry_run") is True for e in events)
    steps = [json.loads(e["result"]) for e in events if e["action"] == "Mapper.step"]
    assert len(steps) == 9 and all(s["acquisition"] == DRY_RUN_SKIPPED and "key" not in s for s in steps)
    moves = [e for e in events if e["action"] == "move_stage"]
    assert len(moves) == 9
    box = DRY_PLAN["box_m"]
    for move in moves:
        assert move["performed"] is False and move["ok"] is True
        assert move["check"]["inside_box"] is True and move["check"]["box_m"] == box
        x, y = move["arguments"][0][:2]
        assert move["check"]["xy_m"] == [x, y]
    assert {e["action"] for e in events if e.get("performed") is False} >= {"unblank_beam", "acquire_scanned_image", "place_beam", "acquire_spectrum", "write_attribute"}
    assert {e["action"] for e in events if e.get("performed") is True} == {"blank_beam"}


def test_a_dry_run_is_approved_like_a_real_start(monkeypatch, mapper_proxy, approvals_proxy):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False, approvals_device_address="asyncroscopy/approvals/default")
    monkeypatch.setattr(server.gate, "approvals", lambda: approvals_proxy)
    reserved = call(mapper_proxy, "reserve", {"plan": {**DRY_PLAN, "dry_run": True}})
    request = json.dumps({"reservation_id": reserved["reservation_id"]})
    approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "test"}))
    try:
        with pytest.raises(ToolError, match="APPROVAL REQUIRED"):
            start = replace(describe_native("Mapper_start"), risk="high")
            server.gate.check(start, {"request": request}, "agent", None, describe=lambda: "Reserved plan: " + mapper_proxy.get_reservation(request))
        pending = [p for p in json.loads(approvals_proxy.list_pending()) if p["action"] == "Mapper_start"]
        assert pending and '"dry_run": true' in pending[-1]["reason"]  # the operator sees it is a rehearsal
        for p in pending:
            approvals_proxy.deny(json.dumps({"request_id": p["request_id"], "who": "test"}))
    finally:
        approvals_proxy.set_mode(json.dumps({"mode": "auto", "who": "test"}))
        call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})


def test_dry_run_is_fixed_at_reserve(mapper_proxy):
    refused = call(mapper_proxy, "reserve", {"plan": {**DRY_PLAN, "dry_run": "yes"}})
    assert refused["error"]["code"] == "bad_request"
    reserved = call(mapper_proxy, "reserve", {"plan": DRY_PLAN})
    try:
        flipped = call(mapper_proxy, "start", {"reservation_id": reserved["reservation_id"], "dry_run": True})
        assert flipped["error"]["code"] == "bad_request" and "set in the plan" in flipped["error"]["message"]
        assert call(mapper_proxy, "get_reservation", {"reservation_id": reserved["reservation_id"]})["dry_run"] is False
    finally:
        call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})


# ----------------------------------------------------------------------
# The simulated flag
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def hardware_approvals(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/approvals/hardware"))


def test_auto_mode_is_refused_on_a_stack_that_is_not_simulated(hardware_approvals, approvals_proxy):
    assert hardware_approvals.get_simulated() is False and approvals_proxy.get_simulated() is True
    assert hardware_approvals.get_mode() == "approve"  # default_mode auto, but not on hardware
    with pytest.raises(tango.DevFailed, match="not simulated"):
        hardware_approvals.set_mode(json.dumps({"mode": "auto", "who": "operator"}))
    assert hardware_approvals.get_mode() == "approve"
    hardware_approvals.set_mode(json.dumps({"mode": "approve", "who": "operator"}))  # tightening is always allowed


def test_the_endpoint_binds_every_interface_only_on_a_simulator_or_when_the_config_says_so():
    assert bind_host("0.0.0.0", allow_remote_bind=False, simulated=False)[0] == "127.0.0.1"
    assert "allow_remote_bind" in bind_host("0.0.0.0", allow_remote_bind=False, simulated=False)[1]
    assert bind_host("0.0.0.0", allow_remote_bind=True, simulated=False) == ("0.0.0.0", None)
    assert bind_host("0.0.0.0", allow_remote_bind=False, simulated=True) == ("0.0.0.0", None)
    assert bind_host("127.0.0.1", allow_remote_bind=False, simulated=False) == ("127.0.0.1", None)
    assert stack_simulated("localhost", 1, None) is False  # no approvals device: not known, so not simulated


def _launcher(name):
    spec = importlib.util.spec_from_file_location(f"launcher_{name}_flag", ROOT / "startup_scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[f"launcher_{name}_flag"] = module
    spec.loader.exec_module(module)
    return module


def test_every_stack_config_says_whether_it_is_simulated_and_every_device_gets_it():
    import yaml

    run_servers = _launcher("run_servers")
    hardware = {"Spectra300.yaml", "Spectra300eels.yaml", "ThinkPad-utkarsh-covalent-setup.yaml", "covalent-F200.yaml", "covalent-talos.yaml", "Jupiter-local.yaml"}
    stacks = [p for p in sorted((ROOT / "configs").glob("*.yaml")) if "instrument" in (yaml.safe_load(p.read_text(encoding="utf-8")) or {})]
    assert {p.name for p in stacks} >= hardware
    for path in stacks:
        raw = run_servers.read_config(path)  # with any extends: chain merged in
        assert isinstance(raw.get("simulated"), bool), f"{path.name} must say simulated: true or false"
        config = run_servers.load_config(path)
        assert config.simulated is (path.name not in hardware), path.name
        flag = [str(config.simulated).lower()]
        assert config.instrument.properties["simulated"] == flag
        assert all(device.properties["simulated"] == flag for device in config.support_devices), path.name


def test_a_stack_config_without_the_flag_counts_as_hardware(tmp_path):
    run_servers = _launcher("run_servers")
    text = (ROOT / "configs" / "DigitalTwin.yaml").read_text(encoding="utf-8").replace("simulated: true", "")
    path = tmp_path / "noflag.yaml"
    path.write_text(text, encoding="utf-8")
    assert run_servers.load_config(path).simulated is False
    path.write_text("simulated: maybe\n" + text, encoding="utf-8")
    with pytest.raises(ValueError, match="simulated must be true or false"):
        run_servers.load_config(path)


def test_run_mcp_passes_allow_remote_bind(tmp_path):
    run_mcp = _launcher("run_mcp")
    text = (ROOT / "configs" / "mcp.yaml").read_text(encoding="utf-8")
    assert "--allow-remote-bind" not in run_mcp.build_command(run_mcp.load_config(ROOT / "configs" / "mcp.yaml"))
    opened = tmp_path / "open.yaml"
    opened.write_text(text.replace("http_host: 127.0.0.1", "http_host: 0.0.0.0\n  allow_remote_bind: true"), encoding="utf-8")
    assert "--allow-remote-bind" in run_mcp.build_command(run_mcp.load_config(opened))


# ----------------------------------------------------------------------
# One time source
# ----------------------------------------------------------------------
def test_twins_report_their_simulated_clock_and_hardware_its_own(tango_ctx):
    twin = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/drift"))
    start = twin.read_attribute("clock_s").value
    assert start == json.loads(twin.get_parameters())["simulated_time_s"]
    position = list(twin.get_stage())
    twin.move_stage([position[0] + 1e-8, *position[1:]])
    assert twin.read_attribute("clock_s").value >= start + 2.0  # the move's overhead, in simulated time (time_scale 0)
    autoscript = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/autoscriptmicroscope/default"))
    assert abs(autoscript.read_attribute("clock_s").value - time.time()) < 60  # hardware: Unix time


@pytest.mark.slow
def test_drift_fit_and_revisits_use_the_twins_clock(monkeypatch, tango_ctx):
    from tests.test_map_pipeline import InProcessAnalysis

    twin = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/drift"))
    mapper = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/drift"))
    mapper.set_timeout_millis(60_000)
    assert json.loads(twin.get_parameters())["time_scale"] == 0  # simulated time races ahead of wall time
    twin.set_defocus(0.0)
    servers, finished = [], []

    def server(_mapper, tool, arguments):
        if not servers:
            servers.append(InProcessAnalysis([Path(arguments["run"]).parent]))
        reply = servers[0](_mapper, tool, arguments)
        if tool == "analysis_job_results":
            finished.append(reply)
        return reply

    monkeypatch.setattr(Mapper, "analysis_url", "in-process")
    monkeypatch.setattr(Pipeline, "call", lambda self, tool, arguments: server(self, tool, arguments))

    revisit_fov = 5e-8
    plan = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1.0e-7, "overlap": 0.25, "imsize": 128, "dwell_s": 2e-5,
            "find_regions": 3, "revisit": {"fov_m": revisit_fov, "dwell_s": 2e-5}}
    first_acquisition = len(json.loads(twin.export_ground_truth())["acquisitions"])
    reserved = call(mapper, "reserve", {"plan": plan})
    started = call(mapper, "start", {"reservation_id": reserved["reservation_id"]})
    assert wait_done(mapper, started["job_id"])["state"] == "done"
    results = call(mapper, "results", {"job_id": started["job_id"]})
    call(mapper, "release", {"reservation_id": reserved["reservation_id"]})
    tiles = reserved["estimate"]["tiles"]

    truth = json.loads(twin.export_ground_truth())
    images = [a for a in truth["acquisitions"][first_acquisition:] if a["kind"] == "image"]
    tile_images, revisit_images = images[:tiles], images[tiles:]

    # The twin's drift: steady, so its rate is the change over the run divided by the simulated time.
    first, last = tile_images[0], tile_images[-1]
    true_rate = (np.array(last["drift_end_m"]) - np.array(first["drift_start_m"])) / (last["t_end_s"] - first["t_start_s"])
    stitched = next(r for r in finished if r.get("summary", {}).get("drift"))
    drift = stitched["summary"]["drift"]
    times, xy = np.array(drift["times_s"]), np.array(drift["xy_m"])
    fitted_rate = (xy[-1] - xy[0]) / (times[-1] - times[0])
    # The fit is timed in simulated seconds: its span is the tiles' span on the twin's clock, not wall time.
    assert times[-1] == pytest.approx(last["t_start_s"] - first["t_start_s"], rel=0.05)
    assert np.linalg.norm(fitted_rate - true_rate) < 0.1 * np.linalg.norm(true_rate), (fitted_rate, true_rate)

    # Where each revisit landed: map coordinates are the sample's, shifted by one constant (the stitch's anchor).
    meta = servers[0].tools.store.map_metadata(results["result_keys"]["map"])
    pixel, (x0, y0) = meta["pixel_size_m"], meta["origin_xy_m"]
    centres = np.array([[x0 + (c + (plan["imsize"] - 1) / 2) * pixel, y0 + (r + (plan["imsize"] - 1) / 2) * pixel] for r, c in meta["tile_positions_px"]])
    offset = np.median(np.array([a["center_true_m"] for a in tile_images]) - centres, axis=0)
    visits = [v for v in results["summary"]["revisits"]["first"] if "image" in v]
    assert visits and len(revisit_images) == len(visits)
    for visit, image in zip(visits, revisit_images):
        intended = np.array(visit["stage_xy_m"]) + np.array(visit["drift_xy_m"])  # the region, or a point moved inward from the box edge
        landed = np.array(image["center_true_m"]) - offset
        assert np.linalg.norm(landed - intended) < 0.1 * revisit_fov, (visit, landed, intended)
