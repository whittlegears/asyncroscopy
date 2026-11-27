"""Health checks, failure diagnosis and the recovery ladder, against the twin corrector."""

import pytest
import tango

from asyncroscopy.health.monitor import HealthMonitor, diagnose, recover


@pytest.fixture()
def corrector(tango_ctx, corrector_proxy):
    access = tango_ctx.get_device_access("asyncroscopy/corrector/default")
    corrector_proxy.command_inout("reconnect")
    yield access, corrector_proxy
    corrector_proxy.command_inout("reconnect")


def test_healthy_device(corrector):
    access, _ = corrector
    h = HealthMonitor([access]).check(access)
    assert h.healthy and h.state == "ON" and h.has_reconnect and h.dev_class == "DigitalTwinCorrector"


def test_dropped_corrector_link_is_diagnosed_and_reconnected(corrector):
    access, proxy = corrector
    proxy.inject_link_drop()
    monitor = HealthMonitor([access])
    h = monitor.check(access)
    assert h.problem == "fault" and "lost" in h.detail

    with pytest.raises(tango.DevFailed) as err:
        proxy.measure_c1a1()
    assert diagnose(err.value) == "link_down"

    steps = []
    result = recover(monitor, access, step=lambda name, args, fn: (steps.append(name), fn())[1])
    assert result["status"] == "recovered" and result["by"] == "reconnect"
    assert steps == ["recovery.reconnect"]
    assert "aberrations" in proxy.measure_c1a1()


def test_disruptive_rung_waits_for_approval(corrector):
    access, proxy = corrector
    proxy.inject_link_drop()
    result = recover(HealthMonitor([access]), access, allowed=())
    assert result["status"] == "needs_approval" and result["next_rung"] == "reconnect"


def test_unreachable_device():
    monitor = HealthMonitor(["tango://127.0.0.1:1/no/such/device#dbase=no"], timeout_ms=300)
    h = monitor.check("tango://127.0.0.1:1/no/such/device#dbase=no")
    assert not h.reachable and h.problem == "unreachable"
    result = recover(monitor, "tango://127.0.0.1:1/no/such/device#dbase=no", allowed=("reconnect", "init"),
                     settle_s=0.1)
    assert result["status"] == "needs_approval" and result["next_rung"] == "restart_server"


class _Cmd:
    def __init__(self, name):
        self.cmd_name = name


class FakeServer:
    """A device whose server restart is the only thing that fixes it."""

    def __init__(self):
        self.restarted = False
        self.calls = []

    def proxy(self, name):
        fake = self

        class P:
            def set_timeout_millis(self, ms): pass

            def ping(self): return 1

            def info(self):
                return type("I", (), {"dev_class": "Vendor", "server_id": "Vendor/inst"})()

            def state(self): return "ON" if fake.restarted else "FAULT"

            def status(self): return "ok" if fake.restarted else "vendor server hung"

            def command_list_query(self): return [_Cmd("Init")]

            def adm_name(self): return "dserver/Vendor/inst"

            def command_inout(self, cmd):
                fake.calls.append((name, cmd))
                if cmd == "RestartServer":
                    fake.restarted = True
        return P()


def test_ladder_climbs_to_server_restart_when_allowed():
    fake = FakeServer()
    monitor = HealthMonitor(["vendor/dev/1"], proxy=fake.proxy)
    result = recover(monitor, "vendor/dev/1", allowed=("reconnect", "init", "restart_server"), settle_s=0.1)
    assert result["status"] == "recovered" and result["by"] == "restart_server"
    assert fake.calls == [("vendor/dev/1", "Init"), ("dserver/Vendor/inst", "RestartServer")]


# -- through the MCP bridge -------------------------------------------------------------

def _server(monkeypatch, tmp_path, **kw):
    from asyncroscopy.mcp.mcp_server import MCPServer

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    return MCPServer("test", "localhost", 1234, blocked_classes=[], blocked_functions={},
                     data_device_address="", verbose=False, **kw)


def test_timeout_reports_readback_and_next_step(monkeypatch, tmp_path, corrector):
    access, _ = corrector
    server = _server(monkeypatch, tmp_path)
    server._device_of[("DigitalTwinCorrector", "correct_aberration")] = access
    info = type("CI", (), {"in_type": tango.CmdArgType.DevString, "out_type": tango.CmdArgType.DevString,
                           "in_type_desc": "", "out_type_desc": ""})

    def times_out(arg):
        tango.Except.throw_exception("API_DeviceTimedOut", "Timeout (3000 mS) exceeded", "test")

    wrapper = server._create_wrapper(times_out, info, "correct_aberration", "DigitalTwinCorrector")
    with pytest.raises(RuntimeError) as err:
        wrapper(arg="C1 1e-9")
    text = str(err.value)
    assert "Readback after timeout: get_info =" in text and "DigitalTwinCorrector" in text
    assert "system_health" in text


def test_recover_device_queues_server_restart_for_operator(monkeypatch, tmp_path):
    from asyncroscopy.supervision.service import SupervisionConfig, Supervisor

    fake = FakeServer()
    server = _server(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "health_monitor",
                        lambda: HealthMonitor(["vendor/dev/1"], proxy=fake.proxy))
    server.supervision_config = SupervisionConfig(enabled=True, dir=str(tmp_path))
    server.supervisor = Supervisor(server.supervision_config)
    monkeypatch.setattr("asyncroscopy.health.monitor.HealthMonitor.wait_healthy",
                        lambda self, d, timeout_s=0, poll_s=0: self.check(d))

    result = server.recover_device("dev/1", origin="agent")
    assert result["status"] == "needs_approval" and result["next_rung"] == "restart_server"
    q = server.supervisor.queue
    pending = q.get(result["queued_for_approval"])
    assert pending.status == "awaiting_approval" and pending.origin == "agent"
    assert not fake.restarted                       # nothing disruptive without the operator
    assert q.approve(pending.id, by="operator")
    import time
    t0 = time.time()
    while q.get(pending.id).status != "done" and time.time() - t0 < 5:
        time.sleep(0.02)
    assert fake.restarted
