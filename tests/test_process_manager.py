import json
import os
import signal
import subprocess
from types import SimpleNamespace

from asyncroscopy.utils import process_manager
from asyncroscopy.utils.process_manager import ProcessManager, ManagedProcess


def test_process_manager_init(tmp_path):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    assert manager.name == "test_run"
    assert manager.state_dir == tmp_path
    assert manager.state_file == tmp_path / "test_run.json"


def test_start_process_tracks_process_group(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    calls = {}

    class FakePopen:
        stdout = None
        stderr = None
        pid = 9999

        def __init__(self, command, **kwargs):
            calls["command"] = command
            calls["kwargs"] = kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(process_manager.subprocess, "Popen", FakePopen)

    with manager:
        proc = manager.start_process("test_key", "Test Label", ["uv", "run", "dummy"])
        assert proc.pid == 9999
        assert proc.key == "test_key"
        assert proc.label == "Test Label"
        assert proc.command == ["uv", "run", "dummy"]
        assert proc in manager.active_processes

        # State file must contain active pids
        assert manager.state_file.exists()
        pids = json.loads(manager.state_file.read_text(encoding="utf-8"))
        assert pids == [9999]

        if os.name == "nt":
            assert "creationflags" in calls["kwargs"]
        else:
            assert calls["kwargs"]["start_new_session"] is True


def test_stop_process_graceful_and_force(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path, timeout=0.1)
    signals = []

    class FakeProcess:
        pid = 12345
        _poll_val = None
        wait_calls = 0

        def poll(self):
            return self._poll_val

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                # First wait times out to trigger force kill escalation
                raise subprocess.TimeoutExpired(["cmd"], timeout)
            self._poll_val = 0
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    def mock_killpg(pgid, sig):
        signals.append((pgid, sig))

    taskkill_calls = []

    if os.name != "nt":
        monkeypatch.setattr(process_manager.os, "killpg", mock_killpg)
        monkeypatch.setattr(process_manager.os, "getpgid", lambda pid: pid)
    else:
        # Capture taskkill invocations without spawning a real process
        class FakePopenResult:
            pass

        def mock_popen(cmd, **kwargs):
            taskkill_calls.append(cmd)
            return FakePopenResult()

        monkeypatch.setattr(process_manager.subprocess, "Popen", mock_popen)

    proc = ManagedProcess("test_key", "Test Label", FakeProcess(), ["cmd"])
    manager.active_processes.append(proc)
    manager.save()

    manager.stop_process(proc)

    assert proc not in manager.active_processes
    # The state file must have removed the PID
    pids = json.loads(manager.state_file.read_text(encoding="utf-8"))
    assert pids == []

    if os.name != "nt":
        assert (12345, signal.SIGTERM) in signals
        assert (12345, signal.SIGKILL) in signals
    else:
        assert taskkill_calls == [["taskkill", "/F", "/T", "/PID", "12345"]]


def test_cleanup_stale_state_on_enter(tmp_path, monkeypatch):
    state_file = tmp_path / "test_run.json"
    state_file.write_text("[12345, 67890]", encoding="utf-8")

    killed_pids = []

    def mock_kill(pid, sig):
        killed_pids.append((pid, sig))

    monkeypatch.setattr(process_manager.os, "kill", mock_kill)
    if os.name != "nt":
        monkeypatch.setattr(process_manager.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(process_manager.os, "killpg", mock_kill)

    time_vals = [100.0, 102.0, 100.0, 102.0]
    monkeypatch.setattr(process_manager.time, "time", lambda: time_vals.pop(0) if time_vals else 200.0)
    monkeypatch.setattr(process_manager.time, "sleep", lambda x: None)

    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    with manager:
        # State file should be cleaned up on entering context
        assert not state_file.exists()

    if os.name != "nt":
        # Check both SIGTERM and SIGKILL were sent due to mock time not incrementing
        assert (12345, signal.SIGTERM) in killed_pids
        assert (12345, signal.SIGKILL) in killed_pids
        assert (67890, signal.SIGTERM) in killed_pids
        assert (67890, signal.SIGKILL) in killed_pids


def test_wipe_databases(tmp_path, monkeypatch):
    monkeypatch.setattr(process_manager, "PROJECT_DIR", tmp_path)

    lowercase = tmp_path / "tango_database.db"
    uppercase = tmp_path / "Tango_database.db"
    lowercase.write_text("old db", encoding="utf-8")
    uppercase.write_text("old db", encoding="utf-8")

    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    manager.wipe_databases()

    assert not lowercase.exists()
    assert not uppercase.exists()


def _fake_killers(monkeypatch):
    """Record kills instead of performing them, on either OS."""
    killed = []

    class Done:
        returncode = 0
        stdout = ""

    def fake_run(cmd, **kwargs):
        if cmd[0] == "taskkill":
            killed.append(int(cmd[cmd.index("/PID") + 1]))
        return Done()

    monkeypatch.setattr(process_manager.subprocess, "run", fake_run)
    monkeypatch.setattr(ProcessManager, "_kill_stale_pid", lambda self, pid: killed.append(pid))
    return killed


def test_stop_processes_on_port_kills_only_listeners(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    killed = _fake_killers(monkeypatch)
    monkeypatch.setattr(process_manager, "listening_pids", lambda port: {5555, os.getpid(), 4} if port == 9094 else set())

    assert manager.stop_processes_on_port(9094) == 1
    assert killed == [5555]  # never this process or Windows' System PID
    assert manager.stop_processes_on_port(1234) == 0


def test_listening_pids_psutil_ignores_connected_clients(monkeypatch):
    class Conn:
        def __init__(self, pid, status, port):
            self.pid, self.status = pid, status
            self.laddr = SimpleNamespace(port=port)

    fake = SimpleNamespace(
        CONN_LISTEN="LISTEN",
        Error=Exception,
        net_connections=lambda kind: [
            Conn(10, "LISTEN", 9094),        # the server
            Conn(11, "ESTABLISHED", 9094),   # a client's accepted socket
            Conn(12, "LISTEN", 8000),        # another port
            Conn(None, "LISTEN", 9094),      # pid hidden by the OS
        ],
    )
    monkeypatch.setattr(process_manager, "psutil", fake)

    assert process_manager.listening_pids(9094) == {10}


def test_listening_pids_cli_fallback_asks_for_listeners_only(monkeypatch):
    calls = []

    class Done:
        stdout = "321\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr(process_manager, "psutil", None)
    monkeypatch.setattr(process_manager.subprocess, "run", fake_run)

    assert process_manager.listening_pids(9094) == {321}
    joined = " ".join(calls[0])
    if os.name == "nt":
        assert "Get-NetTCPConnection -State Listen -LocalPort 9094" in joined
        assert "wmic" not in joined and "netstat" not in joined
    else:
        assert calls[0][:1] == ["lsof"] and "tcp:9094" in calls[0] and "-sTCP:LISTEN" in calls[0]


def test_listening_pids_none_when_no_tool(monkeypatch):
    def missing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(process_manager, "psutil", None)
    monkeypatch.setattr(process_manager.subprocess, "run", missing)

    assert process_manager.listening_pids(9094) is None


def test_matches_device_server_needs_module_and_instance():
    tokens = ["uv", "run", "--no-sync", "python", "-u", "-m", "pkg.stage", "stage_instance"]
    assert process_manager.matches_device_server(tokens, "pkg.stage", "stage_instance")
    assert not process_manager.matches_device_server(tokens, "pkg.stage", "other_instance")
    assert not process_manager.matches_device_server(tokens, "pkg.scan", "stage_instance")
    # A launcher whose arguments merely mention the module is not a device server.
    assert not process_manager.matches_device_server(["python", "run_servers.py", "pkg.stage"], "pkg.stage", "stage_instance")


def test_stop_stale_device_servers_is_narrow(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    killed = _fake_killers(monkeypatch)
    table = {
        101: ["python", "-u", "-m", "pkg.stage", "stage_instance"],          # ours, same TANGO_HOST
        102: ["uv", "run", "python", "-u", "-m", "pkg.stage", "stage_instance"],  # its uv wrapper
        103: ["python", "-m", "pkg.stage", "stage_instance"],                # same server, other stack
        104: ["python", "-m", "asyncroscopy.instruments.other", "x_instance"],  # unrelated asyncroscopy
        os.getpid(): ["python", "-m", "pkg.stage", "stage_instance"],
    }
    hosts = {101: "localhost:9094", 102: None, 103: "localhost:9200"}
    monkeypatch.setattr(process_manager, "process_command_lines", lambda: table)
    monkeypatch.setattr(process_manager, "_process_tango_host", lambda pid: hosts.get(pid))

    count = manager.stop_stale_device_servers([("pkg.stage", "stage_instance")], tango_host="localhost:9094")

    assert count == 2
    assert sorted(killed) == [101, 102]


def test_stop_stale_device_servers_without_targets_kills_nothing(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)
    killed = _fake_killers(monkeypatch)
    monkeypatch.setattr(process_manager, "process_command_lines", lambda: {1: ["python", "-m", "asyncroscopy.data.data", "data_instance"]})

    assert manager.stop_stale_device_servers() == 0
    assert killed == []


def test_process_command_lines_windows_fallback_does_not_use_wmic(monkeypatch):
    if os.name != "nt":
        return
    calls = []

    class Done:
        stdout = '42\t"C:/tools/venv/python.exe" -u -m pkg.stage stage_instance\n7\t\n'

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr(process_manager, "psutil", None)
    monkeypatch.setattr(process_manager.subprocess, "run", fake_run)

    table = process_manager.process_command_lines()

    assert "Get-CimInstance Win32_Process" in " ".join(calls[0])
    assert table == {42: ["C:/tools/venv/python.exe", "-u", "-m", "pkg.stage", "stage_instance"]}
