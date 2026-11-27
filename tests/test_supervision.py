"""One queue for operator and agent; recipes; limits in the control layer; operator HTTP API."""

import threading
import time

import pytest

from asyncroscopy.atlas.synthetic import SyntheticMicroscope
from asyncroscopy.supervision.policy import ControlPolicy, PolicyViolation
from asyncroscopy.supervision.queue import ActionQueue, OperatorHasControl, load_history
from asyncroscopy.supervision.recipes import Capabilities, Recipe, Step, check
from asyncroscopy.supervision.service import SupervisionConfig, Supervisor, ToolEntry


def wait_for(cond, timeout=5.0):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise AssertionError("condition not met")
        time.sleep(0.01)


@pytest.fixture()
def queue(tmp_path):
    q = ActionQueue(history_path=tmp_path / "history.jsonl")
    yield q
    q.resume()


# -- queue ----------------------------------------------------------------------

def test_actions_record_who_asked_and_which_site(queue):
    assert queue.call("Twin_get_fov", lambda: 2e-9, origin="agent", site_id="s1") == 2e-9
    queue.call("Twin_set_fov", lambda: None, origin="operator", args={"arg": 4e-9})
    rows = queue.snapshot()["actions"]
    assert [(r["origin"], r["status"]) for r in rows] == [("agent", "done"), ("operator", "done")]
    assert rows[0]["site_id"] == "s1" and rows[1]["args"] == {"arg": 4e-9}


def test_pause_holds_between_actions_and_step_lets_one_through(queue):
    ran = []
    queue.pause()
    items = [queue.submit(f"t{i}", lambda i=i: ran.append(i)) for i in range(3)]
    time.sleep(0.1)
    assert ran == []
    queue.step()
    wait_for(lambda: ran == [0])
    time.sleep(0.05)
    assert ran == [0]
    queue.resume()
    for item in items:
        queue.wait(item, 2)
    assert ran == [0, 1, 2]


def test_skip_drops_a_pending_action(queue):
    queue.pause()
    a, b = queue.submit("a", lambda: "a"), queue.submit("b", lambda: "b")
    assert queue.skip(a.action.id)
    queue.resume()
    assert queue.wait(b, 2) == "b"
    with pytest.raises(RuntimeError, match="skipped"):
        queue.wait(a, 1)
    assert queue.get(a.action.id).status == "skipped"


def test_pause_applies_between_steps_of_a_composite_action(queue):
    steps = []

    def composite():
        for i in range(3):
            queue.child(f"step{i}", {}, lambda i=i: steps.append(i))
            if i == 0:
                queue.pause()          # the operator pauses mid-run
        return "ok"

    item = queue.submit("atlas.return_to_site", composite, site_id="s1")
    wait_for(lambda: steps == [0])
    time.sleep(0.1)
    assert steps == [0]
    queue.resume()
    assert queue.wait(item, 2) == "ok"
    parent = queue.get(item.action.id)
    assert len(parent.children) == 3
    assert all(queue.get(c).site_id == "s1" for c in parent.children)


def test_abort_stops_a_composite_at_the_next_step(queue):
    gate = threading.Event()

    def composite():
        queue.child("first", {}, gate.wait)
        queue.child("second", {}, lambda: None)

    item = queue.submit("long", composite)
    wait_for(lambda: queue.current == item.action.id)
    assert queue.abort()
    gate.set()
    with pytest.raises(Exception, match="aborted"):
        queue.wait(item, 2)
    assert queue.get(item.action.id).status == "aborted"


def test_take_over_rejects_agent_keeps_history_and_operator_runs(queue, tmp_path):
    queue.call("before", lambda: 1, origin="agent")
    queue.take_over("alice")
    with pytest.raises(OperatorHasControl):
        queue.call("agent_move", lambda: 2, origin="agent")
    queue.resume()
    assert queue.call("operator_move", lambda: 3, origin="operator") == 3
    queue.release()
    assert queue.call("after", lambda: 4, origin="agent") == 4
    statuses = {a["tool"]: a["status"] for a in load_history(tmp_path / "history.jsonl").values()}
    assert statuses == {"before": "done", "agent_move": "rejected", "operator_move": "done", "after": "done"}


def test_stop_does_not_wait_for_the_queue(tmp_path):
    stopped = threading.Event()
    q = ActionQueue(stop_handler=lambda: stopped.set() or "blanked")
    hold = threading.Event()
    q.submit("long acquisition", hold.wait)
    wait_for(lambda: q.current is not None)
    t0 = time.time()
    out = q.stop("bob")
    assert stopped.is_set() and time.time() - t0 < 0.5 and out["ok"]
    assert q.state()["paused"]                      # nothing new starts by itself afterwards
    hold.set()


def test_hard_limits_refuse_and_approval_waits_for_operator():
    policy = ControlPolicy(stage_limits={"x": [-1e-6, 1e-6]}, require_approval=["*set_column_valves"])
    q = ActionQueue(policy=policy)
    moved = []
    with pytest.raises(PolicyViolation, match="outside the policy limit"):
        q.call("Twin_move_stage", lambda: moved.append(1), args={"arg": [5e-6, 0, 0, 0, 0]})
    assert moved == []
    item = q.submit("Twin_set_column_valves", lambda: "open", args={"state": "open"})
    time.sleep(0.05)
    assert q.get(item.action.id).status == "awaiting_approval"
    later = q.submit("Twin_get_fov", lambda: 1)       # FIFO: waits behind the approval
    time.sleep(0.05)
    assert not later.done.is_set()
    assert q.approve(item.action.id, by="alice")
    assert q.wait(item, 2) == "open" and q.wait(later, 2) == 1
    assert q.get(item.action.id).decided_by == "alice"


def test_limits_apply_inside_composite_actions_too():
    q = ActionQueue(policy=ControlPolicy(stage_limits={"y": [-1e-6, 1e-6]}))

    def composite():
        q.child("move_stage", {"position": [0, 9e-6, 0, 0, 0]}, lambda: None, aliases=["instrument.move_stage"])

    with pytest.raises(PolicyViolation):
        q.call("atlas.return_to_site", composite)


# -- recipes ----------------------------------------------------------------------

def test_recipe_capability_and_twin_gate():
    r = Recipe(name="survey", steps=[Step("instrument.move_stage", {"arg": "${pos}"}),
                                     Step("instrument.acquire_scanned_image", {"detector_list": ["haadf"]})],
               params={"pos": {"default": [0, 0, 0, 0, 0]}})
    twin = Capabilities({"instrument.move_stage", "instrument.acquire_scanned_image"}, {"haadf"}, "twin")
    real = Capabilities({"instrument.move_stage", "instrument.acquire_scanned_image"}, {"haadf"}, "real")
    no_haadf = Capabilities({"instrument.move_stage", "instrument.acquire_scanned_image"}, {"bf"}, "twin")
    assert check(r, twin)["ok"]
    assert not check(r, no_haadf)["ok"] and check(r, no_haadf)["missing"] == {"detectors": ["haadf"]}
    assert "digital twin" in " ".join(check(r, real)["problems"])
    r.validations.append({"hash": r.content_hash(), "ok": True, "instrument_kind": "twin"})
    assert check(r, real)["ok"]
    r.steps[1].args["detector_list"] = ["haadf", "bf"]           # an edit is a new version
    assert not check(r, real)["ok"]


def test_recipe_rejects_undeclared_parameters():
    with pytest.raises(ValueError, match="undeclared"):
        Recipe.from_json({"name": "x", "steps": [{"tool": "instrument.set_fov", "args": {"arg": "${fov}"}}]})


# -- supervisor: save a good run as a recipe, rehearse on the twin, run for real -------------

def make_supervisor(tmp_path, kind="twin", policy=None):
    scope = SyntheticMicroscope(seed=3, fov_m=2e-6, imsize=96)
    cfg = SupervisionConfig(enabled=True, dir=str(tmp_path / kind), recipes_dir=str(tmp_path / "recipes"),
                            policy=policy or {}, agent_wait_s=10)
    sup = Supervisor(cfg, microscope=lambda: scope, instrument_name=f"{kind}-scope", instrument_kind=kind,
                     detectors=lambda: {"haadf"})
    calls = []
    sup.register_tool(ToolEntry("Scope_move_stage", "instrument.move_stage",
                                lambda arg: (calls.append(("move", list(arg))), scope.move_stage(arg))[1]))
    sup.register_tool(ToolEntry("Scope_blank_beam", "instrument.blank_beam",
                                lambda: calls.append(("blank",))))
    return sup, scope, calls


def test_good_run_becomes_recipe_and_replays_on_twin_before_real(tmp_path):
    twin, _, _ = make_supervisor(tmp_path, "twin")
    move = twin.tools["Scope_move_stage"]
    twin.run_tool(move.name, lambda: move.fn(arg=[1e-6, 0, 0, 0, 0]), {"arg": [1e-6, 0, 0, 0, 0]}, origin="agent")
    site = twin.save_site("lamella edge", origin="operator", fov_m=2e-6, imsize=96)
    ids = [a for a in twin.queue.order if twin.queue.actions[a].status == "done"]
    recipe = twin.recipe_from_actions("edge revisit", ids, "move, then save")
    assert [s["tool"] for s in recipe["steps"]] == ["instrument.move_stage", "atlas.save_site"]
    assert site["site_id"]

    real, _, real_calls = make_supervisor(tmp_path, "real")
    with pytest.raises(RuntimeError, match="digital twin"):
        real.run_recipe("edge revisit", origin="agent")
    assert real_calls == []                           # refused before anything moved

    twin.run_recipe("edge revisit", origin="operator")
    assert twin.recipes.load("edge revisit").validated_on_twin()
    out = real.run_recipe("edge revisit", origin="operator")
    assert out["steps"] == 2 and real_calls[0] == ("move", [1e-6, 0, 0, 0, 0])
    parent = real.queue.actions[real.queue.order[-1]]
    assert parent.tool == "recipe.run" and parent.run_id and len(parent.children) >= 2


def test_stop_runs_configured_command_out_of_band(tmp_path):
    sup, _, calls = make_supervisor(tmp_path)
    out = sup.queue.stop()
    assert out["ok"] and calls == [("blank",)]


def test_return_to_site_through_queue_records_children_and_visit(tmp_path):
    sup, scope, _ = make_supervisor(tmp_path)
    sup.calibrate_mapping(origin="operator", fov_m=2e-6, imsize=96)
    scope.move_stage([3e-6, 2e-6, 0, 0, 0])
    site = sup.save_site("pore", origin="operator", fov_m=2e-6, imsize=96)
    scope.move_stage([0, 0, 0, 0, 0])
    visit = sup.return_to_site(site["site_id"], origin="agent")
    assert visit["status"] == "match"
    top = sup.queue.actions[sup.queue.order[-1]]
    assert top.origin == "agent" and top.site_id == site["site_id"]
    kinds = [sup.queue.actions[c].tool for c in top.children]
    assert "acquire_survey" in kinds and "move_stage" in kinds
    assert sup.atlas.get(site["site_id"]).visits[-1]["action_id"] == top.id


# -- operator HTTP API ------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from asyncroscopy.supervision.http import OperatorAPI

    sup, scope, calls = make_supervisor(tmp_path)
    sup.config.operator_token = "s3cret"
    api = OperatorAPI(sup)
    app = Starlette(routes=[Route("/supervision/{path:path}", api.handle,
                                  methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])])
    with TestClient(app) as c:
        c.headers["Authorization"] = "Bearer s3cret"
        yield c, sup, calls


def test_http_requires_token(client):
    c, _, _ = client
    assert c.get("/supervision/state", headers={"Authorization": ""}).status_code == 401
    assert c.get("/supervision/ui", headers={"Authorization": ""}).status_code == 200


def test_http_click_site_queue_return_and_compare(client):
    c, sup, _ = client
    c.post("/supervision/atlas/calibrate", json={"fov_m": 2e-6, "imsize": 96})
    r = c.post("/supervision/atlas/sites", json={"name": "click me", "fov_m": 2e-6, "imsize": 96})
    assert r.status_code == 200
    wait_for(lambda: sup.atlas.sites, timeout=20)
    site_id = next(iter(sup.atlas.sites))
    queued = c.post(f"/supervision/atlas/sites/{site_id}/return", json={}).json()["queued"]
    assert queued["origin"] == "operator" and queued["site_id"] == site_id
    wait_for(lambda: sup.atlas.get(site_id).visits, timeout=20)
    visit = sup.atlas.get(site_id).visits[-1]
    detail = c.get(f"/supervision/atlas/sites/{site_id}/visits/{visit['id']}").json()
    assert detail["status"] == "match" and detail["requested_by"] == "operator"
    for path in (f"atlas/sites/{site_id}/reference.png", f"atlas/sites/{site_id}/visits/{visit['id']}.png"):
        png = c.get(f"/supervision/{path}")
        assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    events = c.get("/supervision/events?since=0").json()["events"]
    assert any(e["kind"] == "action" and e["data"]["tool"] == "atlas.return_to_site" for e in events)


def test_http_controls_and_stop(client):
    c, sup, calls = client
    assert c.post("/supervision/pause").json()["paused"]
    queued = c.post("/supervision/actions", json={"tool": "instrument.move_stage",
                                                  "args": {"arg": [0, 0, 0, 0, 0]}}).json()
    assert c.post(f"/supervision/actions/{queued['id']}/skip").json()["ok"]
    assert c.post("/supervision/take-over", json={"note": "manual alignment"}).json()["controller"] == "operator"
    assert c.post("/supervision/stop").json()["ok"] and ("blank",) in calls
    assert c.post("/supervision/release").json()["controller"] == "shared"
    assert c.get("/supervision/nope").status_code == 404


# -- MCP bridge -------------------------------------------------------------------------

def test_mcp_device_calls_go_through_the_queue_as_agent(monkeypatch, tmp_path):
    import asyncio

    import tango

    from asyncroscopy.mcp.mcp_server import MCPServer

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer("test", "localhost", 1234, blocked_classes=[], blocked_functions={},
                       data_device_address="", verbose=False,
                       supervision={"enabled": True, "dir": str(tmp_path),
                                    "policy": {"stage_limits": {"x": [-1e-6, 1e-6]}}})
    monkeypatch.setattr(server, "_find_tools", dict)
    server.setup(print_summary=False)
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"atlas_return_to_site", "supervision_queue_status", "recipe_run"} <= names
    assert not any(word in n for n in names for word in ("approve", "resume", "take_over", "policy"))

    info = type("CI", (), {"in_type": tango.CmdArgType.DevVarDoubleArray, "out_type": tango.CmdArgType.DevVoid,
                           "in_type_desc": "", "out_type_desc": ""})
    moved = []
    wrapper = server._create_wrapper(lambda pos: moved.append(list(pos)), info, "move_stage", "DigitalTwin")
    *_, direct = server._tool_meta["DigitalTwin_move_stage"]
    server.supervisor.register_tool(ToolEntry("DigitalTwin_move_stage", "instrument.move_stage", direct))

    wrapper(arg=[5e-7, 0, 0, 0, 0])
    with pytest.raises(PolicyViolation):
        wrapper(arg=[9e-6, 0, 0, 0, 0])
    assert moved == [[5e-7, 0, 0, 0, 0]]
    rows = server.supervisor.queue.snapshot()["actions"]
    assert [(r["tool"], r["origin"], r["status"]) for r in rows] == [
        ("DigitalTwin_move_stage", "agent", "done"), ("DigitalTwin_move_stage", "agent", "refused")]
