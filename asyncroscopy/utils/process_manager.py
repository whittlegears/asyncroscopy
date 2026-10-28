import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import threading

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a declared dependency
    psutil = None

TANGO_DATABASE_FILES = ("tango_database.db", "Tango_database.db")
PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

_warned_no_uv = False


def find_uv() -> str | None:
    """Return the path of the uv executable, or None if there is none.

    ``uv run`` exports ``UV`` (its own path) to the process it starts, so a
    launcher started as ``C:\\...\\uv.exe run ...`` finds uv even when the
    installer (winget, for one) left nothing on PATH.
    """
    uv = os.environ.get("UV")
    if uv and Path(uv).is_file():
        return uv
    return shutil.which("uv")


def running_in_project_venv() -> bool:
    """True when this interpreter is the project's own ``.venv``."""
    try:
        return Path(sys.prefix).resolve() == (PROJECT_DIR / ".venv").resolve()
    except OSError:
        return False


def python_command(*args: str, extras: Iterable[str] = (), sync: bool = False) -> list[str]:
    """Build the command a launcher uses to run ``python -u <args>`` in this project.

    With uv available the command is ``<uv> run --no-sync python -u <args>``.
    Children skip uv's sync by default: the launcher that starts them already
    runs in a synced environment, and several children syncing the same venv at
    once race (one can die mid-import while another reinstalls a package).

    Pass ``sync=True`` with ``extras`` when the child needs optional extras the
    environment may lack; the command is then
    ``<uv> run --extra X ... python -u <args>``, which installs them (``uv run``
    syncs inexactly, so other installed extras stay).

    Without uv (not on PATH and the launcher was not started by ``uv run``) the
    current interpreter runs the child: ``<sys.executable> -u <args>``. That is
    right when the launcher itself runs in the project venv; extras cannot be
    installed in that case.
    """
    uv = find_uv()
    if uv is not None:
        command = [uv, "run"]
        if sync:
            for extra in extras:
                command.extend(["--extra", extra])
        else:
            command.append("--no-sync")
        return [*command, "python", "-u", *args]

    global _warned_no_uv
    if not running_in_project_venv() and not _warned_no_uv:
        _warned_no_uv = True
        print(
            f"warning: uv not found; starting children with {sys.executable}, which is not "
            f"{PROJECT_DIR / '.venv'}. Set UV to uv's full path or put uv on PATH.",
            file=sys.stderr,
        )
    return [sys.executable, "-u", *args]


def _run_text(command: list[str], timeout: float = 20.0) -> str | None:
    """Run a read-only query command; return its stdout, or None if it could not run."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout or ""


def _powershell(script: str) -> list[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]


def listening_pids(port: int) -> set[int] | None:
    """PIDs with a TCP socket in LISTEN state on ``port`` (any address).

    Only listeners: a client connected to the port (an MCP bridge, a notebook)
    is not a squatter. Returns None if no method could list sockets.
    """
    port = int(port)
    if psutil is not None:
        try:
            return {
                conn.pid
                for conn in psutil.net_connections(kind="tcp")
                if conn.pid and conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port
            }
        except (psutil.Error, OSError):
            pass  # macOS needs root to see other processes' sockets; try the CLI tools.
    if os.name == "nt":
        out = _run_text(_powershell(
            f"Get-NetTCPConnection -State Listen -LocalPort {port} -ErrorAction SilentlyContinue"
            " | Select-Object -ExpandProperty OwningProcess"
        ))
    else:
        out = _run_text(["lsof", "-nP", "-t", "-i", f"tcp:{port}", "-sTCP:LISTEN"])
    if out is None:
        return None
    return {int(token) for token in out.split() if token.isdigit()}


def process_command_lines() -> dict[int, list[str]] | None:
    """Map PID -> command-line tokens for every visible process, or None on failure."""
    if psutil is not None:
        table = {}
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = proc.info.get("cmdline")
            if cmdline:
                table[proc.info["pid"]] = list(cmdline)
        return table
    if os.name == "nt":
        out = _run_text(_powershell(
            "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
        ))
        separator = "\t"
    else:
        out = _run_text(["ps", "-eo", "pid=,args="])
        separator = None
    if out is None:
        return None
    table = {}
    for line in out.splitlines():
        pid, _, cmdline = line.strip().partition(separator or " ")
        if pid.isdigit() and cmdline.strip():
            table[int(pid)] = [token.strip('"') for token in cmdline.split()]
    return table


def _process_tango_host(pid: int) -> str | None:
    """TANGO_HOST in another process's environment, when the OS lets us read it."""
    if psutil is None:
        return None
    try:
        return psutil.Process(pid).environ().get("TANGO_HOST")
    except (psutil.Error, OSError):
        return None


def matches_device_server(tokens: list[str], module_name: str, instance_name: str) -> bool:
    """True when a command line runs ``-m <module_name> <instance_name>``.

    That is the exact form run_servers.py uses, for the device server itself and
    for the ``uv run`` wrapper around it.
    """
    for index, token in enumerate(tokens[:-2]):
        if token == "-m" and tokens[index + 1] == module_name and tokens[index + 2] == instance_name:
            return True
    return False


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

    def _terminate_popen(self, proc: subprocess.Popen):
        """Terminates a process via SIGTERM, waits, and escalates to SIGKILL if necessary."""
        if proc.poll() is not None:
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                pass
        else:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()

            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait()

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

            if pgid == os.getpgid(0):
                # Never signal our own process group (that would stop this launcher).
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
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

    def scour_ports(
        self,
        ports: list[int],
        device_servers: Iterable[tuple[str, str]] = (),
        tango_host: str | None = None,
    ):
        """Frees the given ports and stops this config's own stale device servers.

        Args:
            ports: TCP ports whose listeners are stopped.
            device_servers: (module_name, instance_name) pairs this launcher is
                about to start. Only processes running exactly those are stopped,
                not every asyncroscopy process on the machine.
            tango_host: "host:port" of this launcher's database. When the OS lets
                us read a matching process's environment and it names a different
                TANGO_HOST, the process belongs to another stack and is left alone.
        """
        for port in ports:
            count = self.stop_processes_on_port(port)
            if count > 0:
                print(f"Cleared {count} stale process(es) on port {port}")
        count = self.stop_stale_device_servers(device_servers, tango_host=tango_host)
        if count > 0:
            print(f"Stopped {count} stale device server process(es)")

    def stop_stale_device_servers(
        self,
        device_servers: Iterable[tuple[str, str]] = (),
        tango_host: str | None = None,
    ) -> int:
        """Stops leftover processes running one of ``device_servers``; returns how many."""
        targets = [(str(module), str(instance)) for module, instance in device_servers]
        if not targets:
            return 0
        table = process_command_lines()
        if table is None:
            print("warning: could not list processes; stale device servers were not checked")
            return 0

        current_pid = os.getpid()
        stopped = 0
        for pid, tokens in table.items():
            if pid == current_pid:
                continue
            if not any(matches_device_server(tokens, module, instance) for module, instance in targets):
                continue
            if tango_host is not None:
                other_host = _process_tango_host(pid)
                if other_host is not None and other_host != tango_host:
                    continue
            self._stop_pid_tree(pid)
            stopped += 1
        return stopped

    def _stop_pid_tree(self, pid: int) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            self._kill_stale_pid(pid)

    def stop_processes_on_port(self, port: int) -> int:
        """Stops the processes LISTENING on a TCP port; returns how many."""
        pids = listening_pids(port)
        if not pids:
            return 0
        stopped = 0
        for pid in sorted(pids):
            if pid == os.getpid() or pid <= 4:  # never ourselves, or Windows' System/Idle
                continue
            if os.name == "nt":
                res = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                if res.returncode == 0:
                    stopped += 1
            else:
                self._kill_stale_pid(pid)
                stopped += 1
        return stopped
