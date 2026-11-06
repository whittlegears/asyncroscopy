"""The action space (asyncroscopy.mcp.action_space) and a controller that runs on it by role alone."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import h5py
import jsonschema
import pytest
import tango
from fastmcp import Client

from asyncroscopy.contracts import validate
from asyncroscopy.examples import role_controller
from asyncroscopy.instruments.yaml_sim.spec import SpecError, action_hints, load_spec
from asyncroscopy.mcp import action_space
from asyncroscopy.mcp.agent_tools import agent_tool_patterns, for_model
from asyncroscopy.mcp.mcp_server import MCPServer
from asyncroscopy.core.names import ROLES
from asyncroscopy.mcp.action_roles import role_for

EXAMPLES = Path(__file__).resolve().parents[1] / "asyncroscopy" / "instruments" / "yaml_sim" / "examples"


class _H5Array:
    """A Tiled array node over an HDF5 dataset (view_data reads .read, .shape, .dtype, .metadata)."""

    def __init__(self, dataset):
        self.data, self.shape, self.dtype = dataset[()], dataset.shape, dataset.dtype
        self.metadata = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in dataset.attrs.items()}

    def read(self, slices=None):
        return self.data[slices] if slices else self.data


class _H5Group(dict):
    def __init__(self, group):
        super().__init__({k: _H5Group(v) if isinstance(v, h5py.Group) else _H5Array(v) for k, v in group.items()})
        self.metadata = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in group.attrs.items()}


@pytest.fixture
def serve(monkeypatch, tango_ctx, data_save_dir, job_dir):
    """An MCPServer over the named session devices, with Tiled replaced by the saved HDF5 files."""

    def node(key, hints=True):
        path = Path(key)
        if not path.exists():
            path = next((p for root in (data_save_dir, job_dir) for p in Path(root).rglob(Path(key).name)), None)
            if path is None:
                raise KeyError(key)
        with h5py.File(path, "r") as handle:
            return _H5Group(handle)

    def make(*names, role="agent"):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda name: tango.DeviceProxy(tango_ctx.get_device_access(name)))
        database = type("Db", (), {"get_device_exported": lambda self, p: type("E", (), {"value_string": list(names)})()})()
        server = MCPServer("as-test", "localhost", 1, blocked_classes=[], blocked_functions={"*": ["Init", "Kill", "RestartServer"]},
                           data_device_address="asyncroscopy/data/default", verbose=False, endpoint_role=role, database=database)
        server.setup(print_summary=False)
        monkeypatch.setattr(server, "_tiled_node", node)
        return server

    return make


def by_tool(space):
    return {a["tool"]: a for a in space["actions"]}


# ---------------------------------------------------------------------- the document
def test_stem_and_afm_name_the_stage_move_differently_and_share_its_role(serve):
    stem = action_space.build(serve("asyncroscopy/digitaltwin/default"))
    afm = action_space.build(serve("asyncroscopy/afm/default"))
    for space in (stem, afm):
        validate(space, "action_space")
        assert space["roles"] == ROLES and not space["warnings"]
    move_stage, move_xy = by_tool(stem)["DigitalTwin_move_stage"], by_tool(afm)["AFMTwin_move_xy"]
    assert move_stage["role"] == move_xy["role"] == "stage.move"
    assert move_stage["risk"] == move_xy["risk"] == "high" and move_stage["effect"] == move_xy["effect"] == "act"
    # The STEM's one array argument says what each element is; the AFM's two arguments say which axis they set.
    (position,) = move_stage["args"]
    assert position["elements"] == ["x", "y", "z", "alpha", "beta"] and position["units"] == ["m", "m", "m", "deg", "deg"]
    assert [(a["name"], a["axis"], a["units"], a["min"], a["max"], a["required"]) for a in move_xy["args"]] == [
        ("x_m", "x", "m", -0.002, 0.002, True), ("y_m", "y", "m", -0.002, 0.002, True)]
    assert move_xy["role_source"] == "rule" and by_tool(afm)["AFMTwin_set_stage_x_m"]["role_source"] == "annotation"
    assert stem["instruments"] == [{"device": "DigitalTwin", "class": "DigitalTwin", "address": "asyncroscopy/digitaltwin/default",
                                    "kind": {"stem": True, "probe": False, "envelope": False}}]
    assert afm["instruments"][0]["kind"] == {"stem": False, "probe": True, "envelope": False}


def test_roles_ranges_and_effects_of_the_stem(serve):
    actions = by_tool(action_space.build(serve("asyncroscopy/digitaltwin/default")))
    roles = {name: a["role"] for name, a in actions.items()}
    assert roles["DigitalTwin_acquire_scanned_image"] == "acquire.image" and roles["DigitalTwin_acquire_spectrum"] == "acquire.spectrum"
    assert roles["DigitalTwin_blank_beam"] == "beam.blank" and roles["DigitalTwin_unblank_beam"] == "beam.unblank"
    assert roles["DigitalTwin_place_beam"] == "beam.place" and roles["DigitalTwin_get_stage"] == "stage.read"
    assert roles["DigitalTwin_set_fov"] == "optics.fov" and roles["DigitalTwin_set_defocus"] == "optics.focus"
    assert roles["view_data"] == roles["list_data"] == "data.read" and roles["action_space"] == "other"
    fov = actions["DigitalTwin_set_fov"]["args"][0]
    assert (fov["units"], fov["min"], fov["max"]) == ("m", 1e-9, 1e-4)  # limits.py, at the twin's (default) properties
    assert actions["DigitalTwin_set_fov"]["risk"] == "low"  # a STEM_LOW setting: low, but it still acts
    assert actions["DigitalTwin_set_fov"]["effect"] == "act" and actions["DigitalTwin_get_fov"]["effect"] == "observe"
    assert actions["action_space"]["risk"] == "low" and actions["action_space"]["effect"] == "observe"
    assert actions["DigitalTwin_get_fov"]["returns"] == {"type": "number", "units": "m"}


def test_state_lists_reads_and_attributes_with_units(serve):
    space = action_space.build(serve("asyncroscopy/afm/default"))
    state = {(s["source"], s["name"]): s for s in space["state"]}
    assert state[("tool", "stage_x_m")]["role"] == "stage.read" and state[("tool", "stage_x_m")]["axis"] == "x"
    assert state[("tool", "stage_x_m")]["units"] == "m"
    assert state[("attribute", "scan_size_m")]["units"] == "m" and state[("attribute", "scan_size_m")]["writable"] is True
    assert ("attribute", "action_hints") not in state and ("attribute", "State") not in state


def test_jobs_carry_the_plan_schema_and_example(serve):
    space = action_space.build(serve("asyncroscopy/mapper/default"))
    (job,) = space["jobs"]
    assert job["device"] == "Mapper" and job["plan_schema"]["title"] == "Map plan"
    jsonschema.validate(job["plan_example"], job["plan_schema"])
    assert job["tools"]["reserve"] == "Mapper_reserve" and job["run_with"] == "run_job"
    roles = {a["tool"]: a["role"] for a in space["actions"]}
    assert roles["Mapper_start"] == "job.start" and roles["Mapper_abort"] == "job.abort" and roles["run_job"] == "job.run"
    assert space["instruments"][0]["kind"]["envelope"] is True


def test_action_space_is_a_native_tool_on_both_endpoints_but_not_for_models(serve):
    for role in ("agent", "operator"):
        server = serve("asyncroscopy/afm/default", role=role)
        assert "action_space" in server._native_names
        assert server.endpoint_info()["risks"]["action_space"] == "low"
        space = asyncio.run(_call(server, "action_space"))
        assert space["endpoint"]["role"] == role and by_tool(space)["AFMTwin_move_xy"]["role"] == "stage.move"
    patterns = agent_tool_patterns(server._all_tool_names(), False)
    assert not for_model("action_space", patterns) and for_model("AFMTwin_move_xy", patterns)


async def _call(server, tool, arguments=None):
    async with Client(server.mcp) as client:
        return action_space.structured(await client.call_tool(tool, arguments or {}))


def test_table_and_json_from_the_command_line(serve, monkeypatch, capsys):
    space = action_space.build(serve("asyncroscopy/afm/default"))
    monkeypatch.setattr(action_space, "fetch", lambda url: asyncio.sleep(0, result=space))
    assert action_space.main(["--url", "http://example.invalid/mcp"]) == 0
    table = capsys.readouterr().out
    assert "stage.move" in table and "AFMTwin_move_xy" in table and "x_m<x> m [-0.002..0.002]" in table
    assert action_space.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == action_space.SCHEMA_VERSION


# ---------------------------------------------------------------------- the controller
@pytest.mark.parametrize("device,expected", [
    ("asyncroscopy/digitaltwin/default", ["DigitalTwin_move_stage", "DigitalTwin_unblank_beam", "DigitalTwin_acquire_scanned_image", "DigitalTwin_blank_beam"]),
    ("asyncroscopy/afm/default", ["AFMTwin_retract_tip", "AFMTwin_move_xy", "AFMTwin_approach_tip", "AFMTwin_acquire_topography"]),
])
def test_one_role_controller_moves_acquires_and_reads_back_on_both_twins(serve, device, expected):
    server = serve(device)

    async def go():
        async with Client(server.mcp) as client:
            keeper = role_controller.RoleController(client, action_space.structured(await client.call_tool("action_space", {})))
            before = await keeper.position()
            try:
                return await role_controller.run(client, offset_m=(1e-7, -1e-7))
            finally:  # the session's twin goes back where it was for the other tests
                await keeper.each("probe.retract")
                await keeper.move(before, before)

    report = asyncio.run(go())
    called = [tool for _role, tool in report["calls"]]
    assert all(name in called for name in expected) and called.index(expected[0]) < called.index(expected[1]) < called.index(expected[2])
    assert report["reached"]["x"] == pytest.approx(report["target"]["x"], abs=2e-8)
    assert report["reached"]["y"] == pytest.approx(report["target"]["y"], abs=2e-8)
    assert report["readback"]["kind"] == "image"
    # Only roles: the controller never named a command (the tools it called came from the action space).
    assert {role for role, _tool in report["calls"]} <= set(ROLES)


# ---------------------------------------------------------------------- roles and the YAML role: key
@pytest.mark.parametrize("command,role", [
    ("move_stage", "stage.move"), ("move_xy", "stage.move"), ("get_stage", "stage.read"), ("approach_tip", "probe.approach"),
    ("retract_tip", "probe.retract"), ("laser_off", "beam.blank"), ("xray_off", "beam.blank"), ("blank_beam", "beam.blank"),
    ("acquire_force_curve", "acquire.spectrum"), ("acquire_topography", "acquire.image"), ("set_scan_size_m", "optics.fov"),
    ("auto_focus", "optics.focus"), ("set_exposure_s", "setting.set"), ("get_status", "setting.read"), ("stop_scan", "safe_stop"),
    ("laser_on", "other"), ("Disconnect", "other"),
])
def test_rules(command, role):
    assert role_for("AnyScope", command, probe=command.endswith("tip")) == role


def test_a_yaml_role_is_checked_and_reaches_the_hints(tmp_path):
    raman = action_hints(load_spec(EXAMPLES / "raman.yaml"))
    assert raman["laser_on"]["role"] == "beam.unblank"  # the file's annotation; the rules would say other
    assert raman["acquire_raman_spectrum"]["role"] == "acquire.spectrum"  # from the kind
    assert raman["set_stage_x_m"] == {"role": "stage.move", "args": [{"name": "stage_x_m", "type": "number", "units": "m", "min": -0.005, "max": 0.005, "doc": "Stage x position", "axis": "x"}]}
    bad = tmp_path / "bad.yaml"
    bad.write_text("settings: {power_w: {default: 1}}\ncommands:\n  fire:\n    doc: Fire.\n    role: beam.fire\n", encoding="utf-8")
    with pytest.raises(SpecError, match="commands.fire.role: 'beam.fire' is not a role"):
        load_spec(bad)
