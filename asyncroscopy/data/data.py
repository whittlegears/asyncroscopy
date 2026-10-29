"""DATA Tango device.

This device is the Tango bridge to the Tiled HTTP data server. It stores the
server URI and acquisition save path used by notebooks and microscope devices.

Acquisitions are registered with Tiled explicitly through ``register_path``.
The DATA device intentionally does not start a Tiled filesystem watcher:
in-situ experiments register each image as it is written and avoid the
overhead of monitoring the full acquisition directory. Write a file fully,
then register it: Tiled opens the file to read its structure.

A Tiled server started by ``start_tiled_server`` is supervised: if it exits or
stops answering it is restarted (with backoff, giving up after repeated
failures), and ``reconcile`` then registers any file written while it was down.
Its output goes to a log file named in ``get_config()["tiled_log"]``.

Starting or restarting a managed Tiled server can take longer than Tango's
default client timeout. Callers that invoke ``start_tiled_server`` directly
should set an extended timeout on their DATA ``DeviceProxy``. Changing
``host``, ``port`` or ``save_path`` while a managed server runs restarts it in
the background; ``get_config`` shows when it is serving the new settings.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlparse

from asyncroscopy.data import tiled_client  # before tiled: it limits the Tiled client's retries
from asyncroscopy.data.tiled_client import CONNECTION_ERRORS, TiledUnavailable

import h5py
import psutil
from SciFiReaders import MRCReader
from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command
from tiled.client.register import identity, register

DEFAULT_TILED_URI = "http://127.0.0.1:9091"
DEFAULT_ACQUISITION_DIR = "outputs/tiled_acquisitions"
ONE_NODE_PER_FILE_WALKER = "tiled.client.register:one_node_per_item"
# Suffixes Tiled 0.2 registers from a file (tiled.mimetypes plus the stdlib's image/tiff); Tiled skips others silently.
REGISTRABLE_SUFFIXES = (".h5", ".hdf5", ".hdf", ".nxs", ".tif", ".tiff")
# register_path answers within about REGISTER_WAIT_SECONDS plus a 2 s check that Tiled
# answers, well inside the 12 s the instrument devices give DATA. A registration that
# takes longer carries on in the background, and reconcile catches it if it fails.
REGISTER_WAIT_SECONDS = 8.0
# How long a background registration may run before it is reported as failed.
REGISTER_TIMEOUT_SECONDS = 20
RECENT_WRITE_SECONDS = 10.0  # reconcile leaves files this new alone: they may still be being written
FUTURE_MTIME_SLACK_SECONDS = 60.0  # a file "modified" further in the future than this has a wrong clock, not a writer
CATALOG_INIT_TIMEOUT_SECONDS = 60
TILED_LOG_DIR = Path(os.environ.get("ASYNCROSCOPY_LOG_DIR", "outputs/logs")) / "tiled"
TILED_LOGS_KEPT = 5
LOG_TAIL_BYTES = 64 * 1024
SUPERVISE_INTERVAL_SECONDS = 5.0
FAILED_CHECKS_BEFORE_RESTART = 3
MAX_RESTARTS = 5
RESTART_WINDOW_SECONDS = 600.0


class DATA(Device):
    """Tango bridge to the Tiled HTTP data server."""

    host = attribute(label="Tiled Host", dtype=str, access=AttrWriteType.READ_WRITE, doc="Hostname or IP address for the Tiled HTTP data server.")
    port = attribute(label="Tiled Port", dtype=int, access=AttrWriteType.READ_WRITE, doc="TCP port for the Tiled HTTP data server.")
    save_path = attribute(label="Acquisition Save Path", dtype=str, access=AttrWriteType.READ_WRITE, doc="Directory where acquisition files are written and served by Tiled.")
    tiled_server = attribute(label="Tiled Server", dtype=str, access=AttrWriteType.READ, doc="yes if the configured Tiled HTTP data server responds, otherwise no.")

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)
        # Tango's Init command runs delete_device then init_device on this same object.
        # A Tiled the operator had asked for is started again, and the locks carry over,
        # so nothing from before Init can start a second Tiled alongside the new one.
        restart_tiled = getattr(self, "_restart_tiled_after_init", False)
        self._tiled_lock = getattr(self, "_tiled_lock", None) or threading.RLock()
        self._register_lock = getattr(self, "_register_lock", None) or threading.Lock()
        if not hasattr(self, "_host"):  # values set through configure() survive Init
            self._host, self._port = self._parse_uri(os.environ.get("ASYNCROSCOPY_TILED_URI", DEFAULT_TILED_URI))
            self._save_path = os.environ.get("ASYNCROSCOPY_ACQUISITION_DIR", DEFAULT_ACQUISITION_DIR)
        self._api_key = tiled_client.api_key()
        self._tiled_process = None
        self._tiled_serve_path = None
        self._tiled_log = None
        self._want_tiled = False
        self._gave_up = False
        self._failed_checks = 0
        self._restart_times: deque[float] = deque()
        self._next_restart_at = 0.0
        self._last_restart_reason = ""
        self._reconcile_due: float | None = None
        self._supervisor = None
        self._supervisor_stop = threading.Event()
        self._tiled_server = "yes" if self._tiled_alive() else "no"
        self._tiled_server_status = ""
        if restart_tiled:
            threading.Thread(target=self.start_tiled_server, name="tiled-restart-after-init", daemon=True).start()
        self.info_stream("DATA device initialised")

    def delete_device(self) -> None:
        self._restart_tiled_after_init = getattr(self, "_want_tiled", False)
        with self._tiled_lock:  # waits for a restart in progress to finish
            supervisor = self._detach_supervisor()
            self._stop_tiled_processes()
        _join(supervisor)
        super().delete_device()

    def read_host(self) -> str:
        return self._host

    def write_host(self, value: str) -> None:
        value = value.strip()
        if value == self._host:
            return
        self._host = value
        self._synchronize_tiled_processes()

    def read_port(self) -> int:
        return self._port

    def write_port(self, value: int) -> None:
        value = int(value)
        if value == self._port:
            return
        self._port = value
        self._synchronize_tiled_processes()

    def read_save_path(self) -> str:
        return self._save_path

    def write_save_path(self, value: str) -> None:
        value = value.strip()
        if value == self._save_path:
            return

        _ensure_directory(value)
        self._save_path = value
        self._synchronize_tiled_processes()

    def read_tiled_server(self) -> str:
        self._tiled_server = "yes" if self._tiled_alive() else "no"
        return self._tiled_server

    @command(dtype_out=str)
    def get_config(self) -> str:
        config = {
            "host": self._host,
            "port": self._port,
            "uri": self._uri(),
            "save_path": self._save_path,
            "tiled_server": self._tiled_server,
            "tiled_server_status": self._tiled_server_status,
            "tiled_server_serving": self._tiled_serve_path,
            "tiled_log": str(self._tiled_log) if self._tiled_log else None,
            "restarts": len(self._restart_times),
            "last_restart_reason": self._last_restart_reason,
        }
        return json.dumps(config)

    @command(dtype_in=str, dtype_out=str)
    def configure(self, config_json: str) -> str:
        config = json.loads(config_json) if config_json else {}
        for key, writer in {
            "host": self.write_host,
            "port": self.write_port,
            "save_path": self.write_save_path,
        }.items():
            if key in config:
                writer(config[key])
        return self.get_config()

    @command(dtype_out=str)
    def start_tiled_server(self, timeout=30) -> str:
        """Start the catalog HTTP server without a filesystem watcher, and supervise it."""
        with self._tiled_lock:
            recovering = self._gave_up or bool(self._restart_times)
            self._gave_up = False
            self._restart_times.clear()
            self._next_restart_at = 0.0
            self._failed_checks = 0
            result = self._start_tiled(timeout)
            if self._tiled_server == "yes" and self._tiled_process is not None:
                self._want_tiled = True
                if recovering:
                    self._reconcile_due = time.monotonic()
                self._start_supervisor()
            return result

    def _start_tiled(self, timeout: float = 30) -> str:
        if self._tiled_process is not None:
            if self._tiled_process.poll() is None and self._tiled_alive():
                self._tiled_server = "yes"
                return self.get_config()
            self._stop_tiled_processes()  # ours, but exited or hung: never mistake it for a foreign server
        note = ""
        if self._tiled_alive():
            refusal, note = self._replace_leftover_tiled()
            if refusal:
                self._tiled_server = "no"
                self._tiled_server_status = refusal
                return self.get_config()
        listener = self._port_listener()
        if listener is not None:
            self._tiled_server = "no"
            self._tiled_server_status = (
                f"port {self._port} is used by {listener}, which is not a Tiled server; "
                "stop it or change tiled.port in the YAML"
            )
            return self.get_config()

        save_path = PureWindowsPath(self._save_path) if _is_windows_drive_path(self._save_path) else Path(self._save_path).expanduser()
        catalog = save_path / ".asyncroscopy_tiled_catalog.db"
        catalog_database = _catalog_database_uri(catalog)

        try:
            _ensure_directory(self._save_path)
            command = [*self._tiled_command(), "catalog", "init", "--if-not-exists", catalog_database]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=CATALOG_INIT_TIMEOUT_SECONDS)
        except subprocess.CalledProcessError as exc:
            self._tiled_server = "no"
            output = (exc.stdout or "").strip()
            self._tiled_server_status = f"{exc}; output: {output}" if output else str(exc)
            return self.get_config()
        except Exception as exc:
            self._tiled_server = "no"
            self._tiled_server_status = str(exc)
            return self.get_config()

        command = [
            *self._tiled_command(), "serve", "catalog", catalog_database,
            "--read", self._save_path, "--write", self._save_path,
            "--public", "--api-key", self._api_key,
            "--host", self._host, "--port", str(self._port),
        ]
        self._tiled_log = _new_log_file(self._port)
        log_config = _quiet_log_config()
        if log_config is not None:
            command += ["--log-config", str(log_config)]
        with open(self._tiled_log, "w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        self._tiled_process = process
        self._tiled_serve_path = self._save_path
        tiled_client.forget(self._uri())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._tiled_alive():
            if process.poll() is not None:
                break
            time.sleep(0.5)
        self._tiled_server = "yes" if self._tiled_alive() else "no"
        if self._tiled_server == "yes":
            self._tiled_server_status = f"{note}running; serving path; files register manually"
        else:
            self._tiled_server_status = f"not running; exit_code={process.poll()}{self._log_tail()}"
        return self.get_config()

    @command(dtype_out=str)
    def stop_tiled_server(self) -> str:
        with self._tiled_lock:
            self._want_tiled = False
            supervisor = self._detach_supervisor()
            self._stop_tiled_processes()
            self._tiled_server = "yes" if self._tiled_alive() else "no"
            self._tiled_server_status = "stopped managed Tiled processes"
        _join(supervisor)
        return self.get_config()

    @command(dtype_in=str, dtype_out=str)
    def register_path(self, path: str) -> str:
        """Register one acquisition file explicitly; no filesystem watcher is used.

        Registering a file that is already registered returns its key unchanged
        (Tiled itself would delete the existing entry on a second registration).
        Answers within about 10 s: a registration that takes longer carries on in
        the background. If the supervised Tiled is down, the key is returned and
        the file is registered by ``reconcile`` once Tiled is back; otherwise a
        Tiled outage is an error, as nothing would register the file later.
        """
        path = path.strip()
        key = PureWindowsPath(path).name if _is_windows_drive_path(path) else Path(path).name
        suffix = Path(key).suffix.lower()
        if suffix not in REGISTRABLE_SUFFIXES:
            raise ValueError(
                f"Tiled cannot register {key!r}: no reader for {suffix or 'a file without a suffix'!r}; "
                f"supported: {' '.join(REGISTRABLE_SUFFIXES)}"
            )

        uri = self._uri()
        try:
            if tiled_client.about(uri) is None:
                raise TiledUnavailable(f"Tiled at {uri} is not answering (no Tiled about-document at {uri}/api/v1/)")
            finished = _run_bounded(
                lambda: self._register_one(tiled_client.connect(uri), path, key),
                REGISTER_WAIT_SECONDS,
                on_late_error=lambda exc: self.warn_stream(f"Background registration of {key} failed: {exc}"),
            )
        except Exception as exc:
            if isinstance(exc, TiledUnavailable) and self._will_reconcile(path):
                self._tiled_server_status = f"deferred: Tiled down; {key} registers when it is back ({exc})"
                self._reconcile_due = time.monotonic()
                self.warn_stream(self._tiled_server_status)
                return key
            message = (
                f"File registration failed: {exc}\n\n"
                f"Requested file:\n    {path}\n\n"
                f"Data save path:\n    {self._save_path}\n\n"
                f"Tiled server serving:\n    {self._tiled_serve_path or '(external server; path not managed by DATA)'}"
            )
            self._tiled_server_status = message
            raise RuntimeError(message) from exc
        if not finished:
            self._tiled_server_status = f"{key} is still registering in the background; it appears in Tiled when done"
            self._reconcile_due = time.monotonic() + REGISTER_TIMEOUT_SECONDS
            return key
        self._tiled_server_status = "running; registered path"
        return key

    def _will_reconcile(self, path: str) -> bool:
        """Whether the supervisor will register ``path`` once Tiled is back."""
        supervising = self._supervisor is not None and self._supervisor.is_alive()
        return self._want_tiled and not self._gave_up and supervising and _reconcilable(path, self._save_path)

    def _register_one(self, client, path: str, key: str) -> None:
        """Register ``path`` under ``key`` unless that same file is already there; raise for a different one.

        Holds a lock so the supervisor's reconcile and an acquisition cannot register
        the same new file at once (Tiled deletes an entry on a colliding registration).
        Raises TiledUnavailable only when Tiled cannot be reached.
        """
        uri = self._uri()
        with self._register_lock:
            existing = _reach(uri, lambda: _lookup(client, key))
            if existing is not None:
                registered = _registered_file(existing)
                if registered is not None and _same_file(registered, path):
                    return
                raise KeyCollision(
                    f"Tiled already has a different entry {key!r} ({registered or 'not a registered file'}); "
                    "rename the new file, since registering it would delete the existing entry"
                )
            try:
                _reach(uri, lambda: asyncio.run(asyncio.wait_for(
                    register(client, path, walkers=[ONE_NODE_PER_FILE_WALKER], key_from_filename=identity),
                    REGISTER_TIMEOUT_SECONDS,
                )))
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise RegistrationTimeout(f"Tiled did not finish registering {key} within {REGISTER_TIMEOUT_SECONDS} s") from exc
            if _reach(uri, lambda: _lookup(client, key)) is None:
                raise FileNotFoundError(
                    f"Tiled skipped {path}: it could not be read as {Path(key).suffix} (or it is not under the served save path)"
                )

    @command(dtype_in=str, dtype_out=str)
    def copy_and_register_remote_file(self, request_json: str) -> str:
        """Convert a remote AutoScript MRC file to HDF5 and register it.

        The source MRC is left untouched. The HDF5 file is written under the
        configured data save path and becomes visible only after an atomic
        rename, so Tiled can never observe a partial conversion.
        """
        request = json.loads(request_json)
        source = Path(request["source_path"]).expanduser()
        if source.suffix.lower() not in {".mrc", ".mrcs"}:
            raise ValueError(f"Expected an MRC source file, received: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"Remote MRC file is not readable: {source}")

        destination_directory = Path(self._save_path).expanduser()
        destination_directory.mkdir(parents=True, exist_ok=True)
        detector = str(request.get("detector", "BM-Ceta"))
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        destination = destination_directory / f"stem_data_{detector}_{stamp}.h5"
        partial = destination.with_suffix(".h5.partial")

        try:
            channels = MRCReader(str(source)).read()
            if "Channel_000" not in channels:
                raise ValueError(f"SciFiReaders did not return Channel_000 for {source}")
            source_data = channels["Channel_000"]
            if len(source_data.shape) != 4:
                raise ValueError(f"Expected SciFiReaders to return 4D-STEM data, received shape {source_data.shape}")

            requested_scan_shape = tuple(int(value) for value in request.get("scan_shape", []))
            if requested_scan_shape and tuple(source_data.shape[:2]) != requested_scan_shape:
                raise ValueError(f"MRC scan shape {source_data.shape[:2]} does not match requested scan shape {requested_scan_shape}")

            with h5py.File(partial, "w", track_order=True) as h5:
                dataset = h5.create_dataset("stem_data", shape=source_data.shape, dtype=source_data.dtype, chunks=True)
                for row in range(source_data.shape[0]):
                    for column in range(source_data.shape[1]):
                        frame = source_data[row, column]
                        dataset[row, column] = frame.compute() if hasattr(frame, "compute") else frame

                dataset.attrs["acquisition_type"] = "stem_data"
                dataset.attrs["detector"] = str(request.get("detector", "BM-Ceta"))
                dataset.attrs["source_format"] = "MRC"
                dataset.attrs["source_file"] = str(source)
                dataset.attrs["data_type"] = str(getattr(source_data, "data_type", "image_4d"))
                for name in ("dwell_time", "scan_region", "scan_shape"):
                    if name in request:
                        value = request[name]
                        dataset.attrs[name] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)

                h5.attrs["source_mrc_metadata_json"] = json.dumps(getattr(source_data, "original_metadata", {}), default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))

            os.replace(partial, destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

        return self.register_path(str(destination))

    @command(dtype_out=str)
    def reconcile(self) -> str:
        """Register every acquisition file under the save path that Tiled does not have yet.

        Files are registered one at a time, never the directory: registering a
        directory makes Tiled delete everything under the catalog root first,
        including stitched maps and other data written through the Tiled API.
        Files changed in the last few seconds are left for the next reconcile,
        since they may still be being written.
        """
        save_path = Path(self._save_path).expanduser()
        result = {"save_path": str(save_path), "registered": [], "already": 0, "pending": [], "failed": []}
        try:
            client = tiled_client.connect(self._uri())
            existing = set(_reach(self._uri(), client.keys))
        except Exception as exc:
            self._reconcile_due = time.monotonic()  # the supervisor tries again on its next check
            self._tiled_server_status = f"reconcile failed: {exc}"
            raise RuntimeError(self._tiled_server_status) from exc
        seen: set[str] = set()
        now = time.time()
        for file in _acquisition_files(save_path):
            if file.name in existing:
                result["already"] += 1
                continue
            try:
                age = now - file.stat().st_mtime
            except OSError:
                continue
            if -FUTURE_MTIME_SLACK_SECONDS < age < RECENT_WRITE_SECONDS:
                result["pending"].append(file.name)
                continue
            if file.name in seen:
                result["failed"].append({"path": str(file), "error": "another file with this name was registered from the save path"})
                continue
            seen.add(file.name)
            try:
                self._register_one(client, str(file), file.name)
                result["registered"].append(file.name)
            except TiledUnavailable as exc:
                result["failed"].append({"path": str(file), "error": f"{type(exc).__name__}: {exc}"})
                result["unreachable"] = True
                break
            except Exception as exc:
                result["failed"].append({"path": str(file), "error": f"{type(exc).__name__}: {exc}"})
        if result.get("unreachable"):
            self._reconcile_due = time.monotonic()
        elif result["pending"]:
            self._reconcile_due = time.monotonic() + RECENT_WRITE_SECONDS
        else:
            self._reconcile_due = None
        self._tiled_server_status = (
            f"running; reconciled save path ({len(result['registered'])} registered, "
            f"{len(result['pending'])} still being written, {len(result['failed'])} failed)"
        )
        result["tiled_server"] = self._tiled_server
        result["tiled_server_status"] = self._tiled_server_status
        return json.dumps(result)

    def _uri(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _tiled_alive(self) -> bool:
        """True when a Tiled server (not just any HTTP server) answers at the configured address."""
        return tiled_client.about(self._uri()) is not None

    def _tiled_health(self) -> tuple[bool, str]:
        """Whether the Tiled this device manages is running, answering and holding the port; the reason when not."""
        if self._tiled_process is None:
            return False, "no managed Tiled process"
        code = self._tiled_process.poll()
        if code is not None:
            return False, f"Tiled exited with code {code}{self._log_tail()}"
        if not self._tiled_alive():
            return False, f"Tiled is running but {self._uri()}/api/v1/ does not answer"
        owner = self._listener_pid()
        if isinstance(owner, int) and not _is_ours(owner, self._tiled_process.pid):
            return False, f"port {self._port} is answered by pid {owner}, not by this device's Tiled (pid {self._tiled_process.pid})"
        return True, ""

    def _listener_pid(self) -> int | None | str:
        """Pid listening on the Tiled port; None if nothing listens; "unknown" if the OS will not say."""
        try:
            for connection in psutil.net_connections(kind="tcp"):
                if connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == self._port:
                    return connection.pid if connection.pid is not None else "unknown"
        except (psutil.Error, OSError):
            return "unknown"
        return None

    def _port_listener(self) -> str | None:
        """'<name> pid <N>' of whatever listens on the Tiled port, or None."""
        pid = self._listener_pid()
        if pid is None:
            return None
        if pid == "unknown":
            try:
                with socket.create_connection((self._host, self._port), timeout=0.5):
                    return "another process (the OS does not say which)"
            except OSError:
                return None
        try:
            return f"{psutil.Process(pid).name()} pid {pid}"
        except psutil.Error:
            return f"pid {pid}"

    def _replace_leftover_tiled(self) -> tuple[str | None, str]:
        """Deal with a Tiled already answering on our port: (refusal, note).

        A Tiled orphaned by an earlier run of this stack (``tiled serve`` reading
        this save path, whose starting process is gone) is stopped, so a supervised
        one can replace it. Anything else is refused: one serving another
        directory or with another API key would fail every registration later,
        and one whose starter still runs belongs to another stack or an operator.
        """
        pid = self._listener_pid()
        if not isinstance(pid, int):
            return (
                f"a Tiled already answers on port {self._port}, and the OS does not say which process it is; "
                "stop it or change tiled.port in the YAML"
            ), ""
        try:
            process = psutil.Process(pid)
            name, command_line, cwd = process.name(), process.cmdline(), process.cwd()
        except psutil.Error as exc:
            return (
                f"a Tiled already answers on port {self._port} (pid {pid}) and it cannot be inspected ({type(exc).__name__}); "
                "stop it or change tiled.port in the YAML"
            ), ""
        served = _read_directory(command_line, cwd)
        if "serve" not in command_line or served is None or not _same_file(served, self._save_path):
            return (
                f"port {self._port} is used by a Tiled ({name} pid {pid}) serving {served or 'an unknown directory'}, "
                f"not {self._save_path}; stop it or change tiled.port in the YAML"
            ), ""
        owner = _live_owner(process)
        if owner is not None:
            return (
                f"another Tiled serving this save path is running (pid {pid}, started by {owner}); "
                "stop it or use another tiled.port"
            ), ""
        try:
            process.terminate()
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        deadline = time.monotonic() + 10
        while self._listener_pid() is not None and time.monotonic() < deadline:
            time.sleep(0.2)
        tiled_client.forget(self._uri())
        return None, f"replaced a Tiled left by an earlier run (pid {pid}); "

    def _log_tail(self, lines: int = 20) -> str:
        if not self._tiled_log:
            return ""
        try:
            with open(self._tiled_log, "rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - LOG_TAIL_BYTES))
                text = log.read().decode("utf-8", errors="replace").splitlines()[-lines:]
        except OSError:
            return ""
        if not text:
            return f"; log {self._tiled_log} is empty"
        return f"; log {self._tiled_log}:\n" + "\n".join(text)

    def _start_supervisor(self) -> None:
        if self._supervisor is not None and self._supervisor.is_alive() and not self._supervisor_stop.is_set():
            return
        self._supervisor_stop = threading.Event()
        self._supervisor = threading.Thread(target=self._supervise, args=(self._supervisor_stop,), name="tiled-supervisor", daemon=True)
        self._supervisor.start()

    def _detach_supervisor(self):
        """Signal the current supervisor to stop (call with the lock held); join the returned thread after releasing it."""
        self._supervisor_stop.set()
        supervisor, self._supervisor = self._supervisor, None
        return supervisor

    def _supervise(self, stop: threading.Event) -> None:
        while not stop.wait(SUPERVISE_INTERVAL_SECONDS):
            try:
                self._supervise_once(stop)
            except Exception as exc:  # the supervisor must outlive any single failure
                self.error_stream(f"Tiled supervisor: {exc}")

    def _supervise_once(self, stop: threading.Event | None = None) -> None:
        """One supervision step: restart the managed Tiled if it died, stopped answering or lost its port; then reconcile."""
        reconcile = False
        with self._tiled_lock:
            if (stop is not None and stop.is_set()) or not self._want_tiled or self._gave_up:
                return
            healthy, reason = self._tiled_health()
            if healthy:
                self._failed_checks = 0
                self._tiled_server = "yes"
                reconcile = self._reconcile_due is not None and time.monotonic() >= self._reconcile_due
            else:
                reconcile = self._restart(reason)
        if reconcile:
            try:
                self.reconcile()
            except Exception as exc:
                self.warn_stream(f"Tiled is back, but reconcile failed: {exc}")

    def _restart(self, reason: str) -> bool:
        """Restart the managed Tiled (called with the lock held); True when it is serving again."""
        self._failed_checks += 1
        exited = self._tiled_process is None or self._tiled_process.poll() is not None
        if not exited and self._failed_checks < FAILED_CHECKS_BEFORE_RESTART:
            return False
        now = time.monotonic()
        while self._restart_times and now - self._restart_times[0] > RESTART_WINDOW_SECONDS:
            self._restart_times.popleft()
        if len(self._restart_times) >= MAX_RESTARTS:
            self._gave_up = True
            self._stop_tiled_processes()
            self._tiled_server = "no"
            self._tiled_server_status = (
                f"not running; gave up after {MAX_RESTARTS} restarts in {RESTART_WINDOW_SECONDS / 60:.0f} min "
                f"(last: {reason}); fix the cause, then call start_tiled_server"
            )
            self.error_stream(self._tiled_server_status)
            return False
        if now < self._next_restart_at:
            return False
        self._restart_times.append(now)
        self._next_restart_at = now + min(60.0, 2.0 ** len(self._restart_times))
        self._last_restart_reason = reason
        self.warn_stream(f"Restarting Tiled: {reason}")
        self._stop_tiled_processes()
        self._start_tiled()
        self._failed_checks = 0
        return self._tiled_server == "yes"

    def _synchronize_tiled_processes(self) -> None:
        """Follow a host, port or save path change: restart a managed Tiled in the background."""
        with self._tiled_lock:
            managed = self._tiled_process is not None and self._tiled_process.poll() is None
            if managed:
                self.info_stream(f"Restarting managed Tiled server for save path: {self._save_path}")
                self._tiled_server_status = "restarting for the new settings"
            else:
                self._tiled_process = None
                self._tiled_serve_path = None
        if managed:
            threading.Thread(target=self._restart_for_new_settings, name="tiled-restart-settings", daemon=True).start()
        elif self._tiled_alive():
            self._tiled_server_status = "running externally; files register manually"
        else:
            self._tiled_server_status = "not running"

    def _restart_for_new_settings(self) -> None:
        with self._tiled_lock:
            self._stop_tiled_processes()
            self._start_tiled()

    def _stop_tiled_processes(self) -> None:
        process = self._tiled_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._tiled_process = None
        self._tiled_serve_path = None
        tiled_client.forget(self._uri())

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, int]:
        without_scheme = uri.split("://", 1)[-1].strip("/")
        host, _, port = without_scheme.partition(":")
        return host or "127.0.0.1", int(port or 9091)

    @staticmethod
    def _tiled_command() -> list[str]:
        return [sys.executable, "-m", "tiled"]


def _is_windows_drive_path(path: str | Path | PureWindowsPath) -> bool:
    windows_path = PureWindowsPath(path)
    return bool(windows_path.drive)


def _catalog_database_uri(path: str | Path | PureWindowsPath) -> str:
    if _is_windows_drive_path(path):
        return f"sqlite:///{PureWindowsPath(path).as_posix()}"
    return str(Path(path).expanduser())


def _ensure_directory(path: str | Path) -> None:
    if not (_is_windows_drive_path(path) and os.name != "nt"):
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)


def _new_log_file(port: int) -> Path:
    """A fresh log file for a Tiled server; only the newest few for this port are kept."""
    TILED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(TILED_LOG_DIR.glob(f"tiled_{port}_*.log"))[: -(TILED_LOGS_KEPT - 1)]:
        try:
            old.unlink()
        except OSError:
            pass
    return TILED_LOG_DIR / f"tiled_{port}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"


def _quiet_log_config() -> Path | None:
    """Tiled's own logging configuration without the per-request access lines, which grow the log without bound."""
    try:
        from tiled.server.logging_config import LOGGING_CONFIG
    except ImportError:
        return None
    config = json.loads(json.dumps(LOGGING_CONFIG))
    for formatter in config.get("formatters", {}).values():
        formatter["use_colors"] = False
    config.setdefault("loggers", {}).setdefault("uvicorn.access", {})["level"] = "WARNING"
    path = TILED_LOG_DIR / "tiled_logging.json"
    try:
        TILED_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    except OSError:
        return None
    return path


class KeyCollision(RuntimeError):
    """The key is taken by a different file; registering would delete the existing entry."""


class RegistrationTimeout(RuntimeError):
    """Tiled did not finish a registration in time."""


def _reach(uri: str, call):
    """Run a call to Tiled, turning a failure to reach it (and only that) into TiledUnavailable."""
    try:
        return call()
    except CONNECTION_ERRORS as exc:
        raise tiled_client.unavailable(uri, exc) from exc


def _run_bounded(call, wait_s: float, on_late_error) -> bool:
    """Run ``call`` in a worker thread; True if it finished within ``wait_s`` (re-raising its error), False if it is still running."""
    outcome: dict = {}

    def work() -> None:
        try:
            call()
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=work, name="tiled-register", daemon=True)
    worker.start()
    worker.join(wait_s)
    if worker.is_alive():
        def report() -> None:
            worker.join()
            if "error" in outcome:
                on_late_error(outcome["error"])

        threading.Thread(target=report, name="tiled-register-report", daemon=True).start()
        return False
    if "error" in outcome:
        raise outcome["error"]
    return True


def _join(thread) -> None:
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=SUPERVISE_INTERVAL_SECONDS + 1)


def _lookup(client, key: str):
    try:
        return client[key]
    except KeyError:
        return None


def _registered_file(node) -> str | None:
    """Local path of the file a Tiled node was registered from, or None (e.g. data written through the API)."""
    try:
        uri = node.data_sources()[0].assets[0].data_uri
    except Exception:
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    local = unquote(parsed.path)
    if len(local) > 2 and local[0] == "/" and local[2] == ":":  # /C:/... on Windows
        local = local[1:]
    return local


def _same_file(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _is_under(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(directory))]) == os.path.normcase(os.path.abspath(directory))
    except ValueError:  # different drives
        return False


def _is_ours(pid: int, our_pid: int) -> bool:
    """Whether ``pid`` is our Tiled process or a child of it (a venv's python.exe starts the real interpreter as a child)."""
    if pid == our_pid:
        return True
    try:
        return any(parent.pid == our_pid for parent in psutil.Process(pid).parents())
    except psutil.Error:
        return True  # cannot tell: do not restart a Tiled that answers


def _live_owner(process) -> str | None:
    """'<name> pid <N>' of the live process that started this Tiled, or None if it is an orphan.

    Steps over parents that are the same ``tiled serve`` command (a venv's python.exe
    starts the real interpreter as a child). When this cannot be told, the Tiled is
    treated as owned, so it is never stopped by mistake.
    """
    try:
        current = process
        while True:
            parent = current.parent()  # psutil returns None when the parent is gone (it also checks pid reuse)
            if parent is None:
                return None
            command_line = parent.cmdline()
            if "serve" in command_line and any("tiled" in part for part in command_line):
                current = parent
                continue
            return f"{parent.name()} pid {parent.pid}"
    except psutil.NoSuchProcess:
        return None
    except psutil.Error as exc:
        return f"a process that cannot be inspected ({type(exc).__name__})"


def _reconcilable(path: str, save_path: str) -> bool:
    """Whether reconcile would find ``path``: under the save path, outside dot-directories and Tiled's storage."""
    if not _is_under(path, save_path):
        return False
    relative = Path(os.path.relpath(os.path.abspath(path), os.path.abspath(save_path)))
    *directories, name = relative.parts
    return not name.startswith(".") and not any(d.startswith(".") or d.endswith("_map") for d in directories)


def _read_directory(command_line: list[str], cwd: str) -> str | None:
    """The ``--read`` directory of a ``tiled serve`` command line, resolved against that process's working directory."""
    for flag, value in zip(command_line, command_line[1:]):
        if flag == "--read":
            return value if os.path.isabs(value) else os.path.join(cwd, value)
    return None


def _acquisition_files(save_path: Path):
    """Registrable files under the save path, skipping dot-directories and Tiled's own writable storage."""
    if not save_path.is_dir():
        return
    for directory, subdirectories, files in os.walk(save_path):
        subdirectories[:] = sorted(d for d in subdirectories if not d.startswith(".") and not d.endswith("_map"))
        for name in sorted(files):
            if not name.startswith(".") and Path(name).suffix.lower() in REGISTRABLE_SUFFIXES:
                yield Path(directory) / name


if __name__ == "__main__":
    DATA.run_server()
