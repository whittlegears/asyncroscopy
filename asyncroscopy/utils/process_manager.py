import _thread
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from collections.abc import Callable
from pathlib import Path

TANGO_DATABASE_FILES = ("tango_database.db", "Tango_database.db")
PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@dataclass
class ManagedProcess:
    key: str  # Unique identifier
    label: str  # The human-readable name of the process
    process: subprocess.Popen  # Actual handle of the process
    command: list[str] | None = None  # The command that was run
    stdout_lines: deque = field(default_factory=lambda: deque(maxlen=1000))  # Stdout buffer
    stderr_lines: deque = field(default_factory=lambda: deque(maxlen=1000))  # Stderr buffer

    @property
    def pid(self) -> int:
        """Returns the process ID."""
        return self.process.pid

    @property
    def running(self) -> bool:
        """Returns True if the process is running."""
        return self.process.poll() is None


class ProcessManager:
    """
    Manages child subprocesses for a single launcher script.

    Args:
        name: Unique identifier for this manager instance.
        state_dir: Directory to store process state files.
        timeout: Seconds to wait for a process to terminate before doing a forced kill.
        max_output_lines: Max number of lines to buffer for stdout/stderr.
    """

    def __init__(
        self,
        name: str = None,
        state_dir: str | Path = ".processes",
        timeout: float = 5.0,
        max_output_lines: int = 200,
    ):
        if name is None:
            name = Path(sys.argv[0]).stem
        if name.endswith(".json"):
            name = Path(name).stem

        self.name = name
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / f"{self.name}.json"
        self.timeout = timeout
        self.max_output_lines = max_output_lines

        self.active_processes: list[ManagedProcess] = []
        self.history: deque[ManagedProcess] = deque(maxlen=1000)

    def __enter__(self):
        self._cleanup_stale_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_all()

    def start_process(
        self,
        key: str,
        label: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | str | None = PROJECT_DIR,
        **kwargs,
    ) -> ManagedProcess:
        """Launches a child process, sets group/session execution, and records state."""
        popen_kwargs: dict = {"env": env, "cwd": cwd, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        popen_kwargs.update(kwargs)

        # Start child process in its own group for clean tree termination
        if os.name == "nt": # Windows
            popen_kwargs.setdefault(
                "creationflags", subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else: # Linux
            if "start_new_session" not in popen_kwargs and "preexec_fn" not in popen_kwargs:
                popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(command, **popen_kwargs)
        managed = ManagedProcess(
            key=key,
            label=label,
            process=proc,
            command=command,
            stdout_lines=deque(maxlen=self.max_output_lines),
            stderr_lines=deque(maxlen=self.max_output_lines),
        )
        self.active_processes.append(managed)
        self.history.append(managed)
        self._drain(proc.stdout, managed.stdout_lines)
        self._drain(proc.stderr, managed.stderr_lines)
        # Persisted immediately, not just on stop: if this process dies
        # ungracefully (crash, force-kill), the next launch's
        # _cleanup_stale_state() needs its PID on disk to find and reap it.
        self.save()
        return managed

    @staticmethod
    def _drain(stream, buf: deque) -> None:
        """Continuously reads lines and saves them into a queue, preventing the pipe from getting blocked"""
        if stream is None:
            return
        def _read() -> None:
            for line in iter(stream.readline, b""):
                buf.append(line.decode(errors="replace").rstrip())
            stream.close()
        threading.Thread(target=_read, daemon=True).start()

    def save(self):
        """Atomically saves live PIDs to this launcher's state file."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        active_pids = [mp.pid for mp in self.active_processes if mp.running]

        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(active_pids, f, indent=2)
        temp_file.replace(self.state_file)


    def stop_process(self, managed: ManagedProcess):
        """Stops a single managed process safely."""
        if not managed.running:
            if managed in self.active_processes:
                self.active_processes.remove(managed)
                self.save()
            return

        self._send_kill_request(managed.process)
        try:
            managed.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._force_kill(managed.process)
            managed.process.wait(timeout=1.0)

        if managed in self.active_processes:
            self.active_processes.remove(managed)
            self.save()

    def shutdown_all(self):
        """Performs a fast, concurrent shutdown of every tracked process."""
        old_handler = None
        # Temporarily ignore Ctrl+C so a second press doesn't abort the cleanup
        if threading.current_thread() is threading.main_thread():
            try:
                old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            except (ValueError, AttributeError):
                pass

        try:
            active = [mp for mp in self.active_processes if mp.running]

            # Broadcast the kill signal to every process
            for mp in reversed(active):
                self._send_kill_request(mp.process)

            # Wait for all of them concurrently with shared timeout
            start_time = time.time()
            for mp in reversed(active):
                proc = mp.process
                if proc.poll() is not None:
                    continue

                # Determine how much of the global timeout remains.
                # If we have spent more time than allowed, provide a small wait
                # to allow for immediate process exit before doing a forced kill.
                elapsed = time.time() - start_time
                rem = max(0.1, self.timeout - elapsed)

                try:
                    proc.wait(timeout=rem)
                except subprocess.TimeoutExpired:
                    self._force_kill(proc)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass

            self.active_processes.clear()
            self._remove_state_file()
        finally:
            # Restore normal Ctrl+C behavior
            if old_handler is not None:
                try:
                    signal.signal(signal.SIGINT, old_handler)
                except (ValueError, AttributeError):
                    pass

    def _send_kill_request(self, proc: subprocess.Popen):
        """Sends an initial termination request without blocking."""
        if os.name == "nt":
            subprocess.Popen(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            )
        else:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()

    def _force_kill(self, proc: subprocess.Popen):
        """Escalates to a hard kill if the graceful request timed out."""
        if os.name == "nt":
            pass # taskkill already does forced tree kill on Windows
        else:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()

    def _cleanup_stale_state(self):
        """Reads state file on startup, terminates surviving PIDs, and deletes the file."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                stale_pids = json.load(f)

            if isinstance(stale_pids, list):
                for pid in stale_pids:
                    if isinstance(pid, int):
                        self._kill_stale_pid(pid)
        except (json.JSONDecodeError, IOError):
            pass
        finally:
            self._remove_state_file()

    def _kill_stale_pid(self, pid: int):
        """Best-effort termination of orphan process IDs from a previous crash.

        Recorded PIDs are process-group leaders (start_process uses
        start_new_session=True), so signaling the whole group is required:
        a bare os.kill only reaches a wrapper like `uv run`, and once that
        wrapper is SIGKILLed it has no chance to relay the signal to the
        actual device-server child it spawned, leaving that child running
        and still holding its Tango server-instance name.
        """
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        else:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                return

            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return

            # Brief poll to see if SIGTERM was honored
            start = time.time()
            while time.time() - start < 1.0:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.05)
                except ProcessLookupError:
                    return

            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _remove_state_file(self):
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except OSError:
                pass

    def wipe_databases(self):
        """Deletes stale database files to prevent startup corruption."""
        for filename in TANGO_DATABASE_FILES:
            path = PROJECT_DIR / filename
            if path.exists():
                try:
                    path.unlink()
                    print(f"Deleted stale database: {filename}")
                except OSError as e:
                    print(f"Failed to delete {filename}: {e}")

    def scour_ports(self, ports: list[int]):
        """Finds and kills any process squatting on critical ports or stale device server processes."""
        for port in ports:
            count = self.stop_processes_on_port(port)
            if count > 0:
                print(f"Cleared {count} stale process(es) on port {port}")
        self.stop_stale_device_servers()

    def stop_stale_device_servers(self):
        """Finds and kills orphaned Python processes running asyncroscopy device servers."""
        current_pid = os.getpid()
        if os.name == "nt":
            try:
                cmd = ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                for line in res.stdout.splitlines():
                    if ("asyncroscopy.instruments" in line or "asyncroscopy.data" in line) and "run_servers.py" not in line:
                        parts = line.strip().split()
                        if parts and parts[-1].isdigit():
                            pid = int(parts[-1])
                            if pid != current_pid:
                                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            except Exception:
                pass

    def stop_processes_on_port(self, port: int) -> int:
        """Identifies and kills processes occupying a specific TCP port."""
        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["netstat", "-ano", "-p", "tcp"],
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                return 0

            pids = set()
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) < 5 or parts[0].upper() != "TCP":
                    continue
                if (
                    parts[3].upper() == "LISTENING"
                    and parts[1].endswith(f":{port}")
                    and parts[-1].isdigit()
                ):
                    pids.add(int(parts[-1]))

            stopped = 0
            for pid in pids:
                res = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                )
                if res.returncode == 0:
                    stopped += 1
            return stopped

        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True
            )
        except FileNotFoundError:
            return 0

        stopped = 0
        for pid_str in result.stdout.splitlines():
            if pid_str.strip().isdigit():
                self._kill_stale_pid(int(pid_str.strip()))
                stopped += 1
        return stopped

# ---------------------------------------------------------------------------
# Whole-tree lifecycle helpers shared by the startup GUIs and launcher scripts.
#
# A launcher such as run_servers.py spawns `uv run` -> python -> N device
# servers (each in its own session so they can be signalled individually) ->
# Tiled. Killing only the top of that tree orphans everything below it, which
# is exactly what the GUI's Stop button and window close must never do.
# ---------------------------------------------------------------------------

PARENT_PID_ENV = "ASYNCROSCOPY_PARENT_PID"
_WINDOWS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def pid_alive(pid: int) -> bool:
    """Best-effort liveness check that works for processes we did not spawn."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            process_query_limited_information = 0x1000
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                still_active = 259
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers os.kill(pid, 0) until it is reaped; treat one as gone
    # so a caller waiting on its own child does not wait forever.
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as handle:
                state = handle.read().rsplit(")", 1)[-1].split()[0]
            return state != "Z"
        except OSError:
            return True
    return True


def process_table() -> list[tuple[int, int, int | None]]:
    """(pid, parent pid, process group id or None) for every process on the machine."""
    if os.name == "nt":
        return [(pid, ppid, None) for pid, ppid in _windows_process_table()]
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "pgid="],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    table = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
            table.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return table


def _windows_process_table() -> list[tuple[int, int]]:
    commands = [
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }",
        ],
        ["wmic", "process", "get", "ProcessId,ParentProcessId"],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                creationflags=_WINDOWS_NO_WINDOW,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        table = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                first, second = int(parts[0]), int(parts[1])
                # PowerShell prints "pid ppid"; wmic prints "ParentProcessId ProcessId".
                table.append((first, second) if command[0] == "powershell" else (second, first))
        if table:
            return table
    return []


def list_descendant_pids(root_pid: int, table: list[tuple[int, int, int | None]] | None = None) -> list[int]:
    """Every process below `root_pid`, breadth first, however deeply nested."""
    if table is None:
        table = process_table()
    children: dict[int, list[int]] = {}
    for pid, ppid, _pgid in table:
        children.setdefault(ppid, []).append(pid)
    found: list[int] = []
    queue = [root_pid]
    seen = {root_pid}
    while queue:
        parent = queue.pop(0)
        for child in children.get(parent, []):
            if child not in seen:
                seen.add(child)
                found.append(child)
                queue.append(child)
    return found


def _signal_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _taskkill(pid: int, tree: bool = True) -> None:
    command = ["taskkill", "/F", "/PID", str(pid)]
    if tree:
        command.insert(1, "/T")
    try:
        subprocess.run(command, capture_output=True, timeout=30, check=False, creationflags=_WINDOWS_NO_WINDOW)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def _wait_for_exit(root: subprocess.Popen | int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if isinstance(root, subprocess.Popen):
            if root.poll() is not None:
                return True
        elif not pid_alive(root):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def kill_process_tree(
    root: subprocess.Popen | int,
    grace: float = 10.0,
    log: Callable[[str], None] | None = None,
) -> list[int]:
    """Stop a launcher and everything it spawned, however deeply nested.

    1. Snapshot every descendant first. Once the root exits, orphaned children
       are re-parented to init and the tree information is gone for good.
    2. Ask the root's process group to shut down (SIGTERM, or Ctrl+Break on
       Windows) so a launcher such as run_servers.py can unwind its own
       ProcessManager and stop Tiled cleanly.
    3. After `grace` seconds, force-kill every survivor: each descendant, the
       process group each descendant leads, and the root itself.

    Returns the PIDs that were still alive after the grace period and had to be
    force-killed. `log`, if given, receives one-line progress messages.
    """
    say = log or (lambda _message: None)
    root_pid = root.pid if isinstance(root, subprocess.Popen) else int(root)
    table = process_table()
    descendants = list_descendant_pids(root_pid, table)
    groups = {pgid for pid, _ppid, pgid in table if pgid is not None and (pid == root_pid or pid in descendants)}
    say(f"Stopping pid {root_pid} and {len(descendants)} descendant process(es)...")

    if os.name == "nt":
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        # A Ctrl+Break reaches the launcher and lets it unwind, but it also
        # reaches any wrapper (uv) which may exit first and drop the tree link.
        # Only attempt it when the descendant snapshot is available to fall
        # back on; otherwise go straight to the hard tree kill.
        if ctrl_break is not None and descendants and grace > 0:
            _signal_pid(root_pid, ctrl_break)
            _wait_for_exit(root, grace)
        _taskkill(root_pid, tree=True)
    else:
        if grace > 0:
            _signal_group(root_pid, signal.SIGTERM)
            _signal_pid(root_pid, signal.SIGTERM)
            if not _wait_for_exit(root, grace):
                say(f"pid {root_pid} did not exit within {grace:.0f}s; force-killing.")
        _signal_group(root_pid, signal.SIGKILL)
        _signal_pid(root_pid, signal.SIGKILL)

    # Anything spawned during the grace period is caught by a second snapshot.
    late = list_descendant_pids(root_pid)
    survivors = [pid for pid in dict.fromkeys([*descendants, *late]) if pid_alive(pid)]
    for pid in survivors:
        if os.name == "nt":
            _taskkill(pid, tree=True)
        else:
            _signal_pid(pid, signal.SIGKILL)
    if os.name != "nt":
        for pgid in groups:
            _signal_group(pgid, signal.SIGKILL)

    if isinstance(root, subprocess.Popen):
        try:
            root.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    if survivors:
        say(f"Force-killed {len(survivors)} process(es): {', '.join(str(pid) for pid in survivors)}")
    else:
        say("All processes stopped.")
    return survivors


def install_shutdown_signal_handler() -> None:
    """Route SIGTERM/SIGHUP/SIGBREAK to a single KeyboardInterrupt.

    Launcher scripts run their real work inside `with ProcessManager() as
    manager:` and rely on KeyboardInterrupt (normally Ctrl+C) to unwind that
    block so `manager.shutdown_all()` runs. This makes an external stop request
    (the GUI's Stop button, a closed terminal, `kill <pid>`) unwind the same
    way instead of the launcher dying instantly and leaving its children behind.

    Wrappers such as `uv run` can forward the same signal more than once for a
    single stop request; only the first raises, because re-raising while
    shutdown_all() is already unwinding would interrupt that cleanup mid-flight.
    """
    shutdown_requested = False

    def request_shutdown(_signum, _frame) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, request_shutdown)
        except (ValueError, OSError):
            pass


def watch_parent_process(
    parent_pid: int,
    on_exit: Callable[[], None] | None = None,
    interval: float = 0.5,
    alive: Callable[[int], bool] = pid_alive,
) -> threading.Thread:
    """Trigger a shutdown of this process as soon as `parent_pid` disappears.

    The default action raises KeyboardInterrupt in the main thread, which a
    launcher script handles exactly like Ctrl+C. This covers the cases no
    signal can: the GUI being force-killed, or its terminal window being
    closed on a platform that does not deliver SIGHUP to a new session.
    """
    action = on_exit or _thread.interrupt_main

    def watch() -> None:
        while alive(parent_pid):
            time.sleep(interval)
        action()

    thread = threading.Thread(target=watch, name=f"parent-watchdog-{parent_pid}", daemon=True)
    thread.start()
    return thread


def watch_parent_from_environment(environ: dict[str, str] | None = None) -> threading.Thread | None:
    """Start watch_parent_process() if a launcher GUI put its PID in the environment."""
    raw = (environ if environ is not None else os.environ).get(PARENT_PID_ENV, "")
    if not raw.strip().isdigit():
        return None
    parent_pid = int(raw)
    if parent_pid <= 0 or parent_pid == os.getpid():
        return None
    return watch_parent_process(parent_pid)


def assign_to_kill_on_close_job(pid: int):
    """Put `pid` (and everything it goes on to spawn) in a Windows job object.

    The job carries KILL_ON_JOB_CLOSE, so when the handle returned here is
    closed - including by the owning process dying for any reason - Windows
    terminates every process in the job. The caller must keep the returned
    handle referenced for as long as the tree should stay alive. Returns None
    on non-Windows platforms or if the assignment fails.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job_object_extended_limit_information = 9
        job_object_limit_kill_on_job_close = 0x2000
        process_set_quota = 0x0100
        process_terminate = 0x0001

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
        if not kernel32.SetInformationJobObject(
            job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(job)
            return None
        process = kernel32.OpenProcess(process_set_quota | process_terminate, False, pid)
        if not process:
            kernel32.CloseHandle(job)
            return None
        assigned = kernel32.AssignProcessToJobObject(job, process)
        kernel32.CloseHandle(process)
        if not assigned:
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None
