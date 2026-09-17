import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import threading

PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

TANGO_DATABASE_FILES = ("tango_database.db", "Tango_database.db")
# Anchored to the project rather than the working directory: the launcher runs
# with cwd=PROJECT_DIR while a GUI or a terminal may not, and all three have to
# resolve the same state files for reaping to find anything.
DEFAULT_STATE_DIR = PROJECT_DIR / ".processes"
# A process running one of these modules is a device server, not a launcher.
DEVICE_SERVER_MODULES = ("asyncroscopy.instruments", "asyncroscopy.data", "asyncroscopy.mcp")
# ...unless its command line also names one of these, which are the processes
# that do the reaping and must never reap themselves.
LAUNCHER_COMMANDS = (
    "run_servers.py",
    "run_mcp.py",
    "run_llm.py",
    "run_segmentation.py",
    "server_gui.py",
    "mcp_gui.py",
)


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
        state_dir: str | Path | None = None,
        timeout: float = 5.0,
        max_output_lines: int = 200,
    ):
        if name is None:
            name = Path(sys.argv[0]).stem
        if name.endswith(".json"):
            name = Path(name).stem

        self.name = name
        self.state_dir = Path(DEFAULT_STATE_DIR if state_dir is None else state_dir)
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

    @classmethod
    def reap_recorded_processes(cls, state_dir: str | Path | None = None) -> int:
        """Kills every PID any launcher recorded under state_dir, then clears the files.

        start_process() persists its children's PIDs immediately, so this reaps
        them without owning them: a GUI that only ever held the launcher handle
        can still stop the device servers the launcher spawned.
        """
        directory = Path(DEFAULT_STATE_DIR if state_dir is None else state_dir)
        if not directory.is_dir():
            return 0
        stopped = 0
        for state_file in sorted(directory.glob("*.json")):
            try:
                recorded = json.loads(state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                recorded = []
            if isinstance(recorded, list):
                for pid in recorded:
                    if isinstance(pid, int) and pid != os.getpid():
                        cls._kill_stale_pid(pid)
                        stopped += 1
            try:
                state_file.unlink()
            except OSError:
                pass
        return stopped

    @classmethod
    def reap_all(cls, state_dir: str | Path | None = None, ports: tuple[int, ...] = ()) -> int:
        """Stops everything this project may have left running, from any caller.

        This is the whole shutdown contract in one call, so a GUI, a launcher and
        a terminal all reap identically: recorded child PIDs first (the precise
        list), then anything squatting the ports the stack needs, then any
        device server still running without a launcher.
        """
        stopped = cls.reap_recorded_processes(state_dir)
        for port in ports:
            stopped += cls.stop_processes_on_port(port)
        stopped += cls.stop_stale_device_servers()
        return stopped

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

    @staticmethod
    def _kill_stale_pid(pid: int):
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
        stale = self.stop_stale_device_servers()
        if stale > 0:
            print(f"Cleared {stale} orphaned device server process(es)")

    @classmethod
    def stop_stale_device_servers(cls) -> int:
        """Kills orphaned device servers this project left running, on any OS.

        A device server that outlived its launcher still owns its Tango
        server-instance name and its port, so the next start fails. Match on the
        module a device server is launched with, and never on a launcher or GUI:
        those are the processes doing the reaping.
        """
        skip_pids = {os.getpid(), os.getppid()}
        if os.name == "nt":
            command = ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"]
        else:
            command = ["ps", "-eo", "pid=,args="]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except (FileNotFoundError, OSError):
            return 0

        stopped = 0
        for line in result.stdout.splitlines():
            pid = cls._pid_of_stale_device_server(line, skip_pids)
            if pid is None:
                continue
            cls._kill_stale_pid(pid)
            stopped += 1
        return stopped

    @staticmethod
    def _pid_of_stale_device_server(line: str, skip_pids: set[int]) -> int | None:
        """Returns the PID on a `ps`/`wmic` line when it is an orphaned device server."""
        line = line.strip()
        if not line:
            return None
        if os.name == "nt":
            # wmic prints "CommandLine  ProcessId", so the PID is the last field.
            parts = line.split()
            pid_text, command = (parts[-1], line) if parts else ("", line)
        else:
            # ps -eo "pid=,args=" prints "<pid> <command line>".
            pid_text, _, command = line.partition(" ")
        if not pid_text.isdigit():
            return None
        pid = int(pid_text)
        if pid in skip_pids:
            return None
        if not any(module in command for module in DEVICE_SERVER_MODULES):
            return None
        if any(owner in command for owner in LAUNCHER_COMMANDS):
            return None
        return pid

    def stop_processes_on_port(cls, port: int) -> int:
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
                cls._kill_stale_pid(int(pid_str.strip()))
                stopped += 1
        return stopped