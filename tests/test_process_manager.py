import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from asyncroscopy.utils import process_manager
from asyncroscopy.utils.process_manager import (
    PARENT_PID_ENV,
    ManagedProcess,
    ProcessManager,
    kill_process_tree,
    list_descendant_pids,
    pid_alive,
    watch_parent_from_environment,
    watch_parent_process,
)


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


def test_stop_processes_on_port(tmp_path, monkeypatch):
    manager = ProcessManager(name="test_run", state_dir=tmp_path)

    if os.name == "nt":
        calls = []

        class FakeCompletedProcess:

            def __init__(self, stdout="", returncode=0):
                self.stdout = stdout
                self.returncode = returncode

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            if "netstat" in cmd[0]:
                return FakeCompletedProcess(
                    stdout="  TCP    0.0.0.0:9094           0.0.0.0:0              LISTENING       5555\n"
                )
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", mock_run)

        count = manager.stop_processes_on_port(9094)
        assert count == 1
        assert any("taskkill" in cmd[0] and "5555" in cmd for cmd in calls)

    else:
        class FakeCompletedProcess:
            stdout = "1001\n1002\n"

        killed_pids = []

        def mock_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError()
            killed_pids.append(pid)

        monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: FakeCompletedProcess())
        monkeypatch.setattr(os, "kill", mock_kill)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", mock_kill)

        count = manager.stop_processes_on_port(9094)
        assert count == 2
        assert 1001 in killed_pids
        assert 1002 in killed_pids


# ---------------------------------------------------------------------------
# Whole-tree shutdown helpers
# ---------------------------------------------------------------------------

PROJECT_DIR = process_manager.PROJECT_DIR

# A stand-in for run_servers.py: it installs the same shutdown handler and
# parent watchdog, then starts two children through ProcessManager (each in
# its own session, exactly like the real device servers).
FAKE_LAUNCHER = textwrap.dedent(
    """
    import sys
    import time

    sys.path.insert(0, sys.argv[1])
    from asyncroscopy.utils.process_manager import (
        ProcessManager,
        install_shutdown_signal_handler,
        watch_parent_from_environment,
    )

    install_shutdown_signal_handler()
    watch_parent_from_environment()
    child = [sys.executable, "-c", "import time\\nwhile True:\\n    time.sleep(0.2)"]
    with ProcessManager(name="fake_launcher", state_dir=sys.argv[2]) as manager:
        for key in ("a", "b"):
            manager.start_process(key=key, label=key, command=child)
        print("READY", " ".join(str(p.pid) for p in manager.active_processes), flush=True)
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("UNWINDING", flush=True)
    """
)

# A launcher that refuses to die politely, so only the force-kill path can
# take its children down.
STUBBORN_LAUNCHER = textwrap.dedent(
    """
    import signal
    import subprocess
    import sys
    import time

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = [sys.executable, "-c", "import time\\nwhile True:\\n    time.sleep(0.2)"]
    children = [subprocess.Popen(child, start_new_session=True) for _ in range(2)]
    print("READY", " ".join(str(p.pid) for p in children), flush=True)
    while True:
        time.sleep(0.1)
    """
)

# A stand-in for a startup GUI: it launches the fake launcher with its own PID
# in the environment, then idles until it is killed.
FAKE_GUI = textwrap.dedent(
    """
    import os
    import subprocess
    import sys
    import time

    env = {**os.environ, "ASYNCROSCOPY_PARENT_PID": str(os.getpid())}
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    launcher = subprocess.Popen(
        [sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]], env=env, stdout=subprocess.PIPE, text=True, **kwargs
    )
    ready = launcher.stdout.readline().strip()
    print("LAUNCHER", launcher.pid, ready, flush=True)
    while True:
        time.sleep(0.1)
    """
)


def _new_group_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _read_ready_line(process: subprocess.Popen, timeout: float = 30.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line.startswith("READY"):
            return [int(pid) for pid in line.split()[1:]]
    raise AssertionError(f"launcher never reported READY (exit code {process.poll()})")


def _wait_until_dead(pids: list[int], timeout: float = 15.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.1)
    return [pid for pid in pids if pid_alive(pid)]


def _kill_leftovers(pids: list[int]) -> None:
    for pid in pids:
        if pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
            except OSError:
                pass


def test_list_descendant_pids_walks_nested_tree():
    table = [(1, 0, 1), (10, 1, 10), (11, 10, 11), (12, 11, 12), (13, 11, 13), (20, 1, 20)]

    assert list_descendant_pids(10, table) == [11, 12, 13]
    assert list_descendant_pids(11, table) == [12, 13]
    assert list_descendant_pids(20, table) == []
    assert list_descendant_pids(999, table) == []


def test_pid_alive_reports_this_process_and_a_dead_child():
    assert pid_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    assert not pid_alive(child.pid)
    assert not pid_alive(-1)


def test_watch_parent_process_fires_once_parent_is_gone():
    alive_calls = []
    triggered = threading.Event()

    def alive(pid: int) -> bool:
        alive_calls.append(pid)
        return len(alive_calls) < 3

    thread = watch_parent_process(4242, on_exit=triggered.set, interval=0.01, alive=alive)

    assert triggered.wait(timeout=5)
    thread.join(timeout=5)
    assert alive_calls == [4242, 4242, 4242]


def test_watch_parent_from_environment_ignores_missing_invalid_and_self():
    assert watch_parent_from_environment({}) is None
    assert watch_parent_from_environment({PARENT_PID_ENV: "not-a-pid"}) is None
    assert watch_parent_from_environment({PARENT_PID_ENV: "0"}) is None
    assert watch_parent_from_environment({PARENT_PID_ENV: str(os.getpid())}) is None


def test_kill_process_tree_takes_down_every_descendant(tmp_path):
    launcher_path = tmp_path / "fake_launcher.py"
    launcher_path.write_text(FAKE_LAUNCHER, encoding="utf-8")
    state_dir = tmp_path / "state"
    launcher = subprocess.Popen(
        [sys.executable, str(launcher_path), str(PROJECT_DIR), str(state_dir)],
        stdout=subprocess.PIPE,
        text=True,
        **_new_group_kwargs(),
    )
    children = []
    try:
        children = _read_ready_line(launcher)
        assert len(children) == 2 and all(pid_alive(pid) for pid in children)
        assert (state_dir / "fake_launcher.json").exists()

        messages = []
        survivors = kill_process_tree(launcher, grace=15.0, log=messages.append)

        assert launcher.poll() is not None
        assert _wait_until_dead(children) == []
        if os.name != "nt":
            # The launcher got the chance to unwind its own ProcessManager, so
            # nothing was left for the force-kill pass and it removed its state file.
            assert survivors == []
            assert not (state_dir / "fake_launcher.json").exists()
        assert messages[-1].startswith(("All processes stopped", "Force-killed"))
    finally:
        _kill_leftovers([launcher.pid, *children])


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM semantics are POSIX-only")
def test_kill_process_tree_force_kills_when_launcher_ignores_sigterm(tmp_path):
    launcher_path = tmp_path / "stubborn_launcher.py"
    launcher_path.write_text(STUBBORN_LAUNCHER, encoding="utf-8")
    launcher = subprocess.Popen([sys.executable, str(launcher_path)], stdout=subprocess.PIPE, text=True, start_new_session=True)
    children = []
    try:
        children = _read_ready_line(launcher)
        started = time.monotonic()

        survivors = kill_process_tree(launcher, grace=0.5)

        assert time.monotonic() - started < 10
        assert launcher.poll() is not None
        assert _wait_until_dead(children) == []
        assert set(survivors) == set(children)
    finally:
        _kill_leftovers([launcher.pid, *children])


def test_launcher_shuts_down_when_its_parent_disappears(tmp_path):
    launcher_path = tmp_path / "fake_launcher.py"
    launcher_path.write_text(FAKE_LAUNCHER, encoding="utf-8")
    gui_path = tmp_path / "fake_gui.py"
    gui_path.write_text(FAKE_GUI, encoding="utf-8")
    state_dir = tmp_path / "state"
    gui = subprocess.Popen(
        [sys.executable, str(gui_path), str(launcher_path), str(PROJECT_DIR), str(state_dir)],
        stdout=subprocess.PIPE,
        text=True,
        **_new_group_kwargs(),
    )
    launcher_pid = None
    children = []
    try:
        deadline = time.monotonic() + 30
        line = ""
        while time.monotonic() < deadline and not line.startswith("LAUNCHER"):
            line = gui.stdout.readline()
            if not line and gui.poll() is not None:
                break
        assert line.startswith("LAUNCHER"), f"fake GUI never reported its launcher (exit code {gui.poll()})"
        parts = line.split()
        launcher_pid = int(parts[1])
        children = [int(pid) for pid in parts[3:]]
        assert len(children) == 2 and pid_alive(launcher_pid) and all(pid_alive(pid) for pid in children)

        # The GUI dies hard, without any chance to signal anyone.
        gui.kill()
        gui.wait(timeout=30)

        assert _wait_until_dead([launcher_pid, *children], timeout=20) == []
    finally:
        _kill_leftovers([pid for pid in [gui.pid, launcher_pid, *children] if pid is not None])


def test_managed_command_stop_kills_the_launcher_tree(tmp_path, monkeypatch):
    """The startup GUIs' Stop button must take down the launcher and every server under it."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    shared = pytest.importorskip("startup_guis.shared")
    qt_core = pytest.importorskip(f"{shared.QT_API}.QtCore") if hasattr(shared, "QT_API") else None
    if qt_core is None:
        from startup_guis import qt_compat

        qt_core = pytest.importorskip(f"{qt_compat.QT_API}.QtCore")
    app = qt_core.QCoreApplication.instance() or qt_core.QCoreApplication([])

    launcher_path = tmp_path / "fake_launcher.py"
    launcher_path.write_text(FAKE_LAUNCHER, encoding="utf-8")
    state_dir = tmp_path / "state"
    lines: list[str] = []
    exits: list[int | None] = []
    command = shared.ManagedCommand(lines.append, exits.append)

    def pump(until, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not until():
            app.processEvents()
            time.sleep(0.02)
        assert until(), f"timed out; output so far: {lines}"

    children: list[int] = []
    try:
        command.start([sys.executable, str(launcher_path), str(PROJECT_DIR), str(state_dir)], state_name="fake_launcher")
        assert command.running
        pump(lambda: any(line.startswith("READY") for line in lines))
        ready = next(line for line in lines if line.startswith("READY"))
        children = [int(pid) for pid in ready.split()[1:]]
        assert len(children) == 2 and all(pid_alive(pid) for pid in children)

        command.stop(grace=15.0)

        assert not command.running
        assert _wait_until_dead(children) == []
        pump(lambda: bool(exits))
        assert any(line.startswith(("All processes stopped", "Force-killed")) for line in lines)
    finally:
        _kill_leftovers([pid for pid in [command.process.pid if command.process else None, *children] if pid is not None])
