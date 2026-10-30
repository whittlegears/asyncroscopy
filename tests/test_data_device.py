import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import h5py
import httpx
import numpy as np
import pytest
import sidpy
import tango

from asyncroscopy.data import data as data_module
from asyncroscopy.data.data import (
    DATA,
    KeyCollision,
    RegistrationTimeout,
    _acquisition_files,
    _catalog_database_uri,
    _live_owner,
    _read_directory,
    _reconcilable,
    _registered_file,
)
from asyncroscopy.data.tiled_client import TiledUnavailable

# A local port nothing listens on, so no test reaches a running Tiled.
CLOSED_PORT = 9


@pytest.fixture(autouse=True)
def no_supervisor_left_running(monkeypatch, request):
    """Stop any supervisor a test started, before its monkeypatches are undone (it could otherwise start a real Tiled)."""
    yield
    if "data_proxy" in request.fixturenames:
        request.getfixturevalue("data_proxy").stop_tiled_server()


def tiled_answers(monkeypatch):
    monkeypatch.setattr("asyncroscopy.data.tiled_client.about", lambda uri, timeout_s=2.0: {"api_version": 0, "library_version": "test"})


class FakeProcess:
    pid = 4321

    def __init__(self):
        self.running = True
        self.terminated = False

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        self.running = False

    def wait(self, timeout):
        return 0

    def kill(self):
        self.running = False


class FakeNode:
    def __init__(self, path):
        self.path = path

    def data_sources(self):
        asset = SimpleNamespace(data_uri=Path(self.path).resolve().as_uri())
        return [SimpleNamespace(assets=[asset])]


class FakeClient:
    """Stands in for a Tiled root: register() adds a node for the file, like Tiled does."""

    def __init__(self, files=()):
        self.nodes = {Path(f).name: FakeNode(f) for f in files}
        self.registered = []

    def __getitem__(self, key):
        return self.nodes[key]

    def keys(self):
        return list(self.nodes)


def use_client(monkeypatch, client, skip=False, delay=0.0):
    tiled_answers(monkeypatch)

    async def fake_register(_client, path, **kwargs):
        client.registered.append(path)
        time.sleep(delay)
        if not skip:
            client.nodes[Path(path).name] = FakeNode(path)

    monkeypatch.setattr("asyncroscopy.data.tiled_client.connect", lambda uri: client)
    monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)


def fake_tiled_commands(monkeypatch, popen_calls, run_commands):
    def fake_popen(command, **kwargs):
        process = FakeProcess()
        popen_calls.append({"command": command, "kwargs": kwargs, "process": process})
        return process

    monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
    monkeypatch.setattr(DATA, "_port_listener", lambda self: None)
    monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "asyncroscopy.data.data.subprocess.run",
        lambda command, **_: run_commands.append(command) or type("Result", (), {"returncode": 0, "stdout": ""})(),
    )


class TestDataDevice:
    def test_state_is_on(self, data_proxy: tango.DeviceProxy) -> None:
        assert data_proxy.state() == tango.DevState.ON

    def test_config_round_trip(self, data_proxy: tango.DeviceProxy, tmp_path) -> None:
        config = {
            "host": "127.0.0.1",
            "port": CLOSED_PORT,
            "save_path": str(tmp_path),
        }

        returned = json.loads(data_proxy.configure(json.dumps(config)))

        assert returned["host"] == config["host"]
        assert returned["port"] == config["port"]
        assert returned["save_path"] == config["save_path"]
        assert returned["uri"] == f"http://127.0.0.1:{CLOSED_PORT}"
        assert returned["tiled_server_status"] == "not running"

    def test_default_uri_is_this_machine(self) -> None:
        assert DATA._parse_uri("") == ("127.0.0.1", 9091)
        assert data_module.DEFAULT_TILED_URI == "http://127.0.0.1:9091"

    def test_save_path_creates_missing_directory(self, data_proxy: tango.DeviceProxy, tmp_path) -> None:
        save_path = tmp_path / "new" / "acquisitions"

        data_proxy.save_path = str(save_path)

        assert save_path.is_dir()

    def test_start_tiled_server_uses_catalog_server_command_and_logs_to_a_file(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        calls = []
        popen_calls = []
        run_commands = []

        def fake_alive(self):
            calls.append(None)
            return len(calls) > 1

        data_proxy.host = "127.0.0.1"
        data_proxy.port = CLOSED_PORT
        data_proxy.save_path = str(tmp_path)
        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(data_module, "TILED_LOG_DIR", tmp_path / "logs")
        fake_tiled_commands(monkeypatch, popen_calls, run_commands)

        returned = json.loads(data_proxy.start_tiled_server())

        assert returned["tiled_server"] == "yes"
        expected_catalog = _catalog_database_uri(tmp_path / ".asyncroscopy_tiled_catalog.db")
        assert len(popen_calls) == 1
        actual_command = popen_calls[0]["command"]
        assert actual_command[:5] == ["python", "-m", "tiled", "serve", "catalog"]
        assert actual_command[5] == expected_catalog
        assert actual_command[6] == "--read"
        assert Path(actual_command[7]) == tmp_path
        assert actual_command[8:11] == ["--write", str(tmp_path), "--public"]
        assert actual_command[13:17] == ["--host", "127.0.0.1", "--port", str(CLOSED_PORT)]
        assert actual_command[17] == "--log-config"
        logging_config = json.loads(Path(actual_command[18]).read_text())
        assert logging_config["loggers"]["uvicorn.access"]["level"] == "WARNING"

        kwargs = popen_calls[0]["kwargs"]
        assert kwargs["stderr"] == subprocess.STDOUT
        assert Path(kwargs["stdout"].name) == Path(returned["tiled_log"])
        assert Path(returned["tiled_log"]).parent == tmp_path / "logs"

        assert len(run_commands) == 1
        assert run_commands[0][:6] == ["python", "-m", "tiled", "catalog", "init", "--if-not-exists"]
        assert run_commands[0][6] == expected_catalog
        data_proxy.stop_tiled_server()

    def test_start_refuses_a_port_held_by_something_that_is_not_tiled(self, data_proxy, monkeypatch, tmp_path) -> None:
        popen_calls, run_commands = [], []
        data_proxy.port = CLOSED_PORT
        data_proxy.save_path = str(tmp_path)
        fake_tiled_commands(monkeypatch, popen_calls, run_commands)
        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: False)
        monkeypatch.setattr(DATA, "_port_listener", lambda self: "nginx.exe pid 4242")

        returned = json.loads(data_proxy.start_tiled_server())

        assert returned["tiled_server"] == "no"
        assert "nginx.exe pid 4242, which is not a Tiled server" in returned["tiled_server_status"]
        assert popen_calls == []

    def test_start_refuses_a_tiled_serving_another_directory(self, data_proxy, monkeypatch, tmp_path) -> None:
        data_proxy.port = CLOSED_PORT
        data_proxy.save_path = str(tmp_path / "ours")
        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        leftover = LeftoverTiled(monkeypatch, ["python", "-m", "tiled", "serve", "catalog", "x.db", "--read", str(tmp_path / "theirs")], str(tmp_path))

        returned = json.loads(data_proxy.start_tiled_server())

        assert returned["tiled_server"] == "no"
        assert "python.exe pid 77) serving" in returned["tiled_server_status"]
        assert "theirs" in returned["tiled_server_status"]
        assert not leftover.terminated

    def test_a_tiled_whose_process_cannot_be_identified_is_refused(self, data_proxy, monkeypatch, tmp_path) -> None:
        data_proxy.port = CLOSED_PORT
        data_proxy.save_path = str(tmp_path)
        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)

        def denied(kind):
            raise data_module.psutil.AccessDenied()

        monkeypatch.setattr(data_module.psutil, "net_connections", denied)

        returned = json.loads(data_proxy.start_tiled_server())

        assert returned["tiled_server"] == "no"
        assert "the OS does not say which process" in returned["tiled_server_status"]

    def test_a_tiled_another_live_stack_runs_on_the_same_path_is_refused_not_killed(self, data_proxy, monkeypatch, tmp_path) -> None:
        data_proxy.port = CLOSED_PORT
        data_proxy.save_path = str(tmp_path)
        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        starter = SimpleNamespace(pid=900, name=lambda: "python.exe", cmdline=lambda: ["python", "startup_scripts/run_servers.py"], parent=lambda: None)
        leftover = LeftoverTiled(monkeypatch, ["python", "-m", "tiled", "serve", "catalog", "c.db", "--read", str(tmp_path)], str(tmp_path), parent=starter)

        returned = json.loads(data_proxy.start_tiled_server())

        assert returned["tiled_server"] == "no"
        assert "another Tiled serving this save path is running (pid 77, started by python.exe pid 900)" in returned["tiled_server_status"]
        assert not leftover.terminated

    def test_a_tiled_left_from_an_earlier_run_on_the_same_path_is_replaced_by_a_supervised_one(self, data_proxy, monkeypatch, tmp_path) -> None:
        # The leftover's --read is relative: it is resolved against its own working directory, not DATA's.
        (tmp_path / "acq").mkdir()
        data_proxy.port = CLOSED_PORT
        data_proxy.save_path = str(tmp_path / "acq")
        popen_calls, run_commands = [], []
        fake_tiled_commands(monkeypatch, popen_calls, run_commands)
        monkeypatch.setattr(data_module, "TILED_LOG_DIR", tmp_path / "logs")
        leftover = LeftoverTiled(monkeypatch, ["python", "-m", "tiled", "serve", "catalog", "c.db", "--read", "acq"], str(tmp_path))
        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: (not leftover.terminated) or bool(popen_calls))

        returned = json.loads(data_proxy.start_tiled_server())

        assert leftover.terminated
        assert len(popen_calls) == 1
        assert returned["tiled_server"] == "yes"
        assert returned["tiled_server_status"].startswith("replaced a Tiled left by an earlier run (pid 77); running")
        data_proxy.stop_tiled_server()

    def test_catalog_database_uri_uses_sqlite_uri_for_windows_drive_path(self) -> None:
        assert (
            _catalog_database_uri("C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db")
            == "sqlite:///C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db"
        )
        assert (
            _catalog_database_uri("C:\\tiled_catalog_test\\.asyncroscopy_tiled_catalog.db")
            == "sqlite:///C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db"
        )

    def test_start_tiled_server_uses_sqlite_uri_for_windows_catalog(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        calls = []
        popen_calls = []
        run_commands = []

        def fake_alive(self):
            calls.append(None)
            return len(calls) > 1

        fake_tiled_commands(monkeypatch, popen_calls, run_commands)
        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(data_module, "TILED_LOG_DIR", tmp_path / "logs")
        # The Windows drive path below is fed only to exercise the
        # drive-path -> sqlite-URI branch; neutralize the mkdir side effect so
        # the test does not depend on the path being creatable (a non-admin user
        # cannot create directories under C:\, and the path can't exist on Linux).
        monkeypatch.setattr("asyncroscopy.data.data._ensure_directory", lambda path: None)

        data_proxy.host = "127.0.0.1"
        data_proxy.port = CLOSED_PORT
        data_proxy.save_path = "C:/tiled_catalog_test"
        calls.clear()

        returned = json.loads(data_proxy.start_tiled_server())

        expected_catalog = "sqlite:///C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db"
        assert returned["tiled_server"] == "yes"
        assert run_commands[0][6] == expected_catalog
        assert popen_calls[0]["command"][5] == expected_catalog
        data_proxy.stop_tiled_server()

    def test_register_path_registers_single_file(self, data_proxy, monkeypatch, tmp_path) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")
        client = FakeClient()
        use_client(monkeypatch, client)

        result = data_proxy.register_path(str(saved))

        assert result == "frame.h5"
        assert client.registered == [str(saved)]

    def test_registering_the_same_file_again_keeps_it(self, data_proxy, monkeypatch, tmp_path) -> None:
        # Tiled deletes an existing entry when a second registration collides with it.
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")
        client = FakeClient([saved])
        use_client(monkeypatch, client)

        assert data_proxy.register_path(str(saved)) == "frame.h5"
        assert client.registered == []
        assert "frame.h5" in client.nodes

    def test_a_different_file_with_a_registered_name_is_refused_not_deferred(self, data_proxy, monkeypatch, tmp_path) -> None:
        # A collision must never pass for "Tiled is down": the key would point at another file.
        (tmp_path / "old").mkdir()
        (tmp_path / "new").mkdir()
        client = FakeClient([tmp_path / "old" / "frame.h5"])
        use_client(monkeypatch, client)
        monkeypatch.setattr(DATA, "_will_reconcile", lambda self, path: True)

        with pytest.raises(tango.DevFailed, match="different entry 'frame.h5'"):
            data_proxy.register_path(str(tmp_path / "new" / "frame.h5"))
        assert client.registered == []
        assert not json.loads(data_proxy.get_config())["tiled_server_status"].startswith("deferred")

    def test_a_slow_registration_carries_on_in_the_background(self, data_proxy, monkeypatch, tmp_path) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")
        client = FakeClient()
        use_client(monkeypatch, client, delay=1.0)
        monkeypatch.setattr(data_module, "REGISTER_WAIT_SECONDS", 0.2)

        started = time.monotonic()
        assert data_proxy.register_path(str(saved)) == "frame.h5"
        assert time.monotonic() - started < 1.0
        assert "still registering in the background" in json.loads(data_proxy.get_config())["tiled_server_status"]
        deadline = time.monotonic() + 5
        while "frame.h5" not in client.nodes:
            assert time.monotonic() < deadline
            time.sleep(0.05)

    def test_an_unsupported_file_type_fails_at_once_naming_the_supported_ones(self, data_proxy, monkeypatch, tmp_path) -> None:
        client = FakeClient()
        use_client(monkeypatch, client)

        with pytest.raises(tango.DevFailed, match=r"no reader for '\.dm4'; supported: \.h5"):
            data_proxy.register_path(str(tmp_path / "frame.dm4"))
        assert client.registered == []

    def test_a_file_tiled_skips_is_reported(self, data_proxy, monkeypatch, tmp_path) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"not really hdf5")
        use_client(monkeypatch, FakeClient(), skip=True)

        with pytest.raises(tango.DevFailed, match="Tiled skipped"):
            data_proxy.register_path(str(saved))

    def test_with_the_supervised_tiled_down_the_key_is_returned_and_registration_deferred(self, data_proxy, monkeypatch, tmp_path) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")

        def down(uri):
            raise TiledUnavailable(f"Tiled at {uri} is not answering")

        monkeypatch.setattr("asyncroscopy.data.tiled_client.connect", down)
        tiled_answers(monkeypatch)
        monkeypatch.setattr(DATA, "_will_reconcile", lambda self, path: True)

        assert data_proxy.register_path(str(saved)) == "frame.h5"
        assert json.loads(data_proxy.get_config())["tiled_server_status"].startswith("deferred: Tiled down; frame.h5")

    def test_with_no_supervised_tiled_an_outage_is_an_error(self, data_proxy, monkeypatch, tmp_path) -> None:
        # An external Tiled, a stopped one, or one the supervisor gave up on: nothing would register the file later.
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")

        def down(uri):
            raise TiledUnavailable(f"Tiled at {uri} is not answering")

        monkeypatch.setattr("asyncroscopy.data.tiled_client.connect", down)
        tiled_answers(monkeypatch)

        with pytest.raises(tango.DevFailed, match="File registration failed: Tiled at .* is not answering"):
            data_proxy.register_path(str(saved))

    def test_a_connection_dropped_during_registration_is_reported_as_tiled_down(self, data_proxy, monkeypatch, tmp_path) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")

        class Dropping(FakeClient):
            def __getitem__(self, key):
                raise httpx.ConnectError("connection refused")

        use_client(monkeypatch, Dropping())

        with pytest.raises(tango.DevFailed, match="is not answering .*ConnectError"):
            data_proxy.register_path(str(saved))

    def test_copy_and_register_remote_mrc_writes_atomic_hdf5(self, data_proxy: tango.DeviceProxy, monkeypatch, tmp_path) -> None:
        source_directory = tmp_path / "autoscript"
        destination_directory = tmp_path / "acquisitions"
        source_directory.mkdir()
        source = source_directory / "scan.mrc"
        source.write_bytes(b"AutoScript MRC fixture")
        source_data = np.arange(24, dtype=np.uint16).reshape(2, 2, 3, 2)
        source_dataset = sidpy.Dataset.from_array(source_data, name="MRC_000", chunks=(1, 1, 3, 2))
        source_dataset.data_type = "image_4d"
        source_dataset.original_metadata = {
            "Pixel size X": np.array([0.5]),
            "Pixel size Y": np.array([0.5]),
        }

        class FakeMRCReader:
            def __init__(self, path):
                assert path == str(source)

            def read(self):
                return {"Channel_000": source_dataset}

        client = FakeClient()
        use_client(monkeypatch, client)
        monkeypatch.setattr("asyncroscopy.data.data.MRCReader", FakeMRCReader)
        data_proxy.save_path = str(destination_directory)

        request = {
            "source_path": str(source),
            "detector": "BM-Ceta",
            "scan_shape": [2, 2],
            "dwell_time": 1e-3,
            "scan_region": [0.0, 0.0, 1.0, 1.0],
        }
        key = data_proxy.copy_and_register_remote_file(json.dumps(request))

        destination = destination_directory / key
        assert source.is_file()
        assert destination.is_file()
        assert client.registered == [str(destination)]
        assert not list(destination_directory.glob("*.partial"))
        with h5py.File(destination, "r") as h5:
            assert h5["stem_data"].shape == (2, 2, 3, 2)
            np.testing.assert_array_equal(h5["stem_data"][()], source_data)
            assert h5["stem_data"].attrs["detector"] == "BM-Ceta"
            assert json.loads(h5["stem_data"].attrs["scan_shape"]) == [2, 2]
            metadata = json.loads(h5.attrs["source_mrc_metadata_json"])
            assert metadata["Pixel size X"] == [0.5]

    def test_copy_and_register_remote_file_rejects_non_mrc(self, data_proxy: tango.DeviceProxy, tmp_path) -> None:
        source = tmp_path / "scan.txt"
        source.write_text("not an MRC")

        with pytest.raises(tango.DevFailed, match="Expected an MRC source file"):
            data_proxy.copy_and_register_remote_file(json.dumps({"source_path": str(source)}))

    def test_reconcile_registers_only_missing_files_one_at_a_time(self, data_proxy, monkeypatch, tmp_path) -> None:
        # Registering the directory would make Tiled delete everything under the root first,
        # stitched maps written through the API included; reconcile must register files only.
        for name in ("present.h5", "missing.h5", "notes.txt", ".hidden.h5"):
            written_a_while_ago(tmp_path / name)
        (tmp_path / "maps").mkdir()
        written_a_while_ago(tmp_path / "maps" / "map_run.h5")
        (tmp_path / "maps" / "map_running.h5").write_bytes(b"x")  # still being written
        (tmp_path / "map_abc_map" / "level_0").mkdir(parents=True)
        (tmp_path / "map_abc_map" / "level_0" / "0.0").write_bytes(b"chunk")
        data_proxy.save_path = str(tmp_path)
        client = FakeClient([tmp_path / "present.h5"])
        client.nodes["map_abc_map"] = SimpleNamespace(data_sources=lambda: [])
        use_client(monkeypatch, client)

        result = json.loads(data_proxy.reconcile())

        assert sorted(result["registered"]) == ["map_run.h5", "missing.h5"]
        assert result["pending"] == ["map_running.h5"]
        assert result["already"] == 1
        assert result["failed"] == []
        assert all(Path(p).is_file() for p in client.registered)
        assert "map_abc_map" in client.nodes

    def test_reconcile_registers_a_file_with_a_clock_far_in_the_future(self, data_proxy, monkeypatch, tmp_path) -> None:
        future = tmp_path / "future.h5"
        future.write_bytes(b"x")
        later = time.time() + 3600
        os.utime(future, (later, later))
        data_proxy.save_path = str(tmp_path)
        use_client(monkeypatch, FakeClient())

        result = json.loads(data_proxy.reconcile())

        assert result["registered"] == ["future.h5"] and result["pending"] == []

    def test_configured_settings_survive_init(self, data_proxy, tmp_path) -> None:
        data_proxy.configure(json.dumps({"save_path": str(tmp_path / "kept")}))
        data_proxy.Init()
        assert json.loads(data_proxy.get_config())["save_path"] == str(tmp_path / "kept")

    def test_reconcile_reports_tiled_down(self, data_proxy, monkeypatch, tmp_path) -> None:
        data_proxy.save_path = str(tmp_path)

        def down(uri):
            raise TiledUnavailable("not answering")

        monkeypatch.setattr("asyncroscopy.data.tiled_client.connect", down)

        with pytest.raises(tango.DevFailed, match="reconcile failed: not answering"):
            data_proxy.reconcile()

    def test_save_path_change_restarts_managed_server(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        popen_calls = []
        run_commands = []

        def fake_alive(self):
            return self._tiled_process is not None and self._tiled_process.poll() is None

        fake_tiled_commands(monkeypatch, popen_calls, run_commands)
        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(data_module, "TILED_LOG_DIR", tmp_path / "logs")

        first_path = tmp_path / "first"
        second_path = tmp_path / "second"

        data_proxy.save_path = str(first_path)
        data_proxy.start_tiled_server()
        data_proxy.save_path = str(second_path)  # returns at once; the restart runs in the background
        deadline = time.monotonic() + 10
        config = json.loads(data_proxy.get_config())
        while len(popen_calls) < 2 or config["tiled_server_status"] == "restarting for the new settings":
            assert time.monotonic() < deadline, config
            time.sleep(0.05)
            config = json.loads(data_proxy.get_config())

        assert popen_calls[0]["process"].terminated is True
        assert [Path(call["command"][7]) for call in popen_calls] == [Path(first_path), Path(second_path)]
        assert Path(config["tiled_server_serving"]) == Path(second_path)
        assert config["tiled_server_status"] == "running; serving path; files register manually"

        data_proxy.stop_tiled_server()

    def test_register_path_error_reports_save_and_serving_paths(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        save_path = tmp_path / "current"
        requested_path = save_path / "missing.h5"
        data_proxy.save_path = str(save_path)

        async def fake_register(*args, **kwargs):
            raise FileNotFoundError(requested_path)

        monkeypatch.setattr("asyncroscopy.data.tiled_client.connect", lambda uri: FakeClient())
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)
        tiled_answers(monkeypatch)

        with pytest.raises(tango.DevFailed) as exc_info:
            data_proxy.register_path(str(requested_path))

        message = str(exc_info.value)
        status = json.loads(data_proxy.get_config())["tiled_server_status"]
        assert "File registration failed:" in message
        assert f"Requested file:\n    {requested_path}" in status
        assert f"Data save path:\n    {save_path}" in status
        assert "Tiled server serving:\n    " in status


def written_a_while_ago(path):
    path.write_bytes(b"x")
    old = time.time() - 60
    os.utime(path, (old, old))


class LeftoverTiled:
    """A Tiled process found listening on the port, as psutil reports it; ``parent`` is its starter (None: gone)."""

    def __init__(self, monkeypatch, command_line, cwd, parent=None):
        self.terminated = False
        listener = SimpleNamespace(status=data_module.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=CLOSED_PORT), pid=77)
        process = SimpleNamespace(
            name=lambda: "python.exe",
            cmdline=lambda: command_line,
            cwd=lambda: cwd,
            parent=lambda: parent,
            terminate=lambda: setattr(self, "terminated", True),
            wait=lambda timeout: 0,
            kill=lambda: None,
        )
        monkeypatch.setattr(data_module.psutil, "net_connections", lambda kind: [] if self.terminated else [listener])
        monkeypatch.setattr(data_module.psutil, "Process", lambda pid: process)


class Supervised:
    """The DATA supervision methods on a plain object (a Tango device cannot be built outside a server)."""

    _supervise_once = DATA._supervise_once
    _restart = DATA._restart
    _tiled_health = DATA._tiled_health
    _stop_tiled_processes = DATA._stop_tiled_processes
    _log_tail = DATA._log_tail
    _uri = DATA._uri
    _register_one = DATA._register_one
    _will_reconcile = DATA._will_reconcile

    def __init__(self, process):
        self._host, self._port = "127.0.0.1", CLOSED_PORT
        self._save_path = "acq"
        self._tiled_lock = threading.RLock()
        self._register_lock = threading.Lock()
        self._tiled_process = process
        self._tiled_serve_path = "x"
        self._tiled_log = None
        self._want_tiled = True
        self._gave_up = False
        self._reconcile_due = None
        self._supervisor = None
        self.listener = None
        self._failed_checks = 0
        self._restart_times = deque()
        self._next_restart_at = 0.0
        self._last_restart_reason = ""
        self._tiled_server = "yes"
        self._tiled_server_status = ""
        self.alive = True
        self.starts = 0
        self.reconciles = 0
        self.messages = []

    def _tiled_alive(self):
        return self.alive

    def _listener_pid(self):
        return self.listener

    def _start_tiled(self, timeout=30):
        self.starts += 1
        self._tiled_process = FakeProcess()
        self._tiled_server = "yes"

    def reconcile(self):
        self.reconciles += 1

    def warn_stream(self, message):
        self.messages.append(message)

    error_stream = warn_stream


class TestTiledSupervisor:
    def test_a_dead_tiled_is_restarted_and_then_reconciled(self) -> None:
        dead = FakeProcess()
        dead.running = False
        device = Supervised(dead)

        device._supervise_once()

        assert device.starts == 1 and device.reconciles == 1
        assert device._last_restart_reason.startswith("Tiled exited with code 0")

    def test_a_healthy_tiled_is_left_alone(self) -> None:
        device = Supervised(FakeProcess())

        device._supervise_once()

        assert device.starts == 0 and device.reconciles == 0

    def test_a_running_tiled_that_stops_answering_is_restarted_after_three_checks(self) -> None:
        process = FakeProcess()
        device = Supervised(process)
        device.alive = False

        device._supervise_once()
        device._supervise_once()
        assert device.starts == 0
        device._supervise_once()

        assert device.starts == 1 and process.terminated

    def test_it_gives_up_after_repeated_restarts(self, monkeypatch) -> None:
        clock = [1000.0]
        monkeypatch.setattr(data_module.time, "monotonic", lambda: clock[0])
        device = Supervised(FakeProcess())
        for _ in range(data_module.MAX_RESTARTS + 1):
            device._tiled_process.running = False
            clock[0] += 61
            device._supervise_once()

        assert device.starts == data_module.MAX_RESTARTS
        assert device._gave_up is True
        assert device._tiled_process is None  # a hung Tiled is stopped, not left holding the port
        assert device._tiled_server_status.startswith(f"not running; gave up after {data_module.MAX_RESTARTS} restarts")
        device._tiled_process = FakeProcess()
        device._tiled_process.running = False
        device._supervise_once()
        assert device.starts == data_module.MAX_RESTARTS

    def test_restarts_back_off(self, monkeypatch) -> None:
        clock = [1000.0]
        monkeypatch.setattr(data_module.time, "monotonic", lambda: clock[0])
        device = Supervised(FakeProcess())
        device._tiled_process.running = False
        device._supervise_once()
        device._tiled_process.running = False
        clock[0] += 1
        device._supervise_once()

        assert device.starts == 1

    def test_nothing_happens_once_stopped(self) -> None:
        dead = FakeProcess()
        dead.running = False
        device = Supervised(dead)
        device._want_tiled = False

        device._supervise_once()

        assert device.starts == 0

    def test_an_old_supervisor_does_nothing_after_its_stop_signal(self) -> None:
        dead = FakeProcess()
        dead.running = False
        device = Supervised(dead)
        stop = threading.Event()
        stop.set()

        device._supervise_once(stop)

        assert device.starts == 0

    def test_another_process_answering_on_the_port_counts_as_unhealthy(self, monkeypatch) -> None:
        unrelated = SimpleNamespace(parents=lambda: [SimpleNamespace(pid=1)])
        child = SimpleNamespace(parents=lambda: [SimpleNamespace(pid=FakeProcess.pid)])
        device = Supervised(FakeProcess())

        device.listener = 555
        monkeypatch.setattr(data_module.psutil, "Process", lambda pid: unrelated)
        healthy, reason = device._tiled_health()
        assert not healthy and "answered by pid 555, not by this device's Tiled (pid 4321)" in reason

        monkeypatch.setattr(data_module.psutil, "Process", lambda pid: child)  # a venv python.exe starts Tiled as a child
        assert device._tiled_health() == (True, "")

    def test_files_left_as_still_being_written_are_reconciled_later(self, monkeypatch) -> None:
        clock = [1000.0]
        monkeypatch.setattr(data_module.time, "monotonic", lambda: clock[0])
        device = Supervised(FakeProcess())
        device._reconcile_due = 1005.0

        device._supervise_once()
        assert device.reconciles == 0
        clock[0] = 1006.0
        device._supervise_once()
        assert device.reconciles == 1

    def test_deferral_needs_a_running_supervisor_and_a_file_under_the_save_path(self, tmp_path) -> None:
        device = Supervised(FakeProcess())
        device._save_path = str(tmp_path)
        inside = str(tmp_path / "a.h5")
        assert not device._will_reconcile(inside)  # no supervisor thread
        device._supervisor = SimpleNamespace(is_alive=lambda: True)
        assert device._will_reconcile(inside)
        assert not device._will_reconcile(str(tmp_path.parent / "elsewhere.h5"))
        device._gave_up = True
        assert not device._will_reconcile(inside)

    def test_a_collision_raises_its_own_error_even_when_tiled_answers(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "old").mkdir()
        client = FakeClient([tmp_path / "old" / "frame.h5"])
        use_client(monkeypatch, client)
        with pytest.raises(KeyCollision):
            Supervised(FakeProcess())._register_one(client, str(tmp_path / "frame.h5"), "frame.h5")

    def test_a_registration_tiled_never_finishes_raises_its_own_error(self, monkeypatch, tmp_path) -> None:
        async def never(*args, **kwargs):
            import asyncio

            await asyncio.sleep(10)

        monkeypatch.setattr("asyncroscopy.data.data.register", never)
        monkeypatch.setattr(data_module, "REGISTER_TIMEOUT_SECONDS", 0.1)
        with pytest.raises(RegistrationTimeout):
            Supervised(FakeProcess())._register_one(FakeClient(), str(tmp_path / "frame.h5"), "frame.h5")

    def test_reconcile_and_an_acquisition_never_register_the_same_file_twice(self, monkeypatch, tmp_path) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"x")
        client = FakeClient()
        use_client(monkeypatch, client, delay=0.2)
        device = Supervised(FakeProcess())
        threads = [threading.Thread(target=device._register_one, args=(client, str(saved), "frame.h5")) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert client.registered == [str(saved)]


def test_the_log_tail_reads_only_the_end_of_a_long_log(tmp_path) -> None:
    log = tmp_path / "tiled.log"
    log.write_text("x" * (data_module.LOG_TAIL_BYTES * 4) + "\nlast line\n")
    device = SimpleNamespace(_tiled_log=log)
    tail = DATA._log_tail(device)
    assert tail.endswith("last line")
    assert len(tail) < data_module.LOG_TAIL_BYTES + 200 + len(str(log))  # the tail names the log, whose path varies by clone


def test_a_tiled_bound_to_every_interface_is_reached_locally_and_advertised_by_its_public_uri(monkeypatch) -> None:
    """In a container Tiled binds 0.0.0.0: DATA talks to it on loopback and tells clients the service address."""
    device = SimpleNamespace(_host="0.0.0.0", _port=9091)
    device._uri = lambda: DATA._uri(device)
    monkeypatch.delenv("ASYNCROSCOPY_TILED_PUBLIC_URI", raising=False)
    assert DATA._uri(device) == "http://127.0.0.1:9091"
    assert DATA._public_uri(device) == "http://127.0.0.1:9091"
    monkeypatch.setenv("ASYNCROSCOPY_TILED_PUBLIC_URI", "http://twin-tiled:9091/")
    assert DATA._public_uri(device) == "http://twin-tiled:9091"
    assert DATA._uri(SimpleNamespace(_host="localhost", _port=9191)) == "http://localhost:9191"


def test_registered_file_reads_a_windows_file_uri() -> None:
    node = SimpleNamespace(data_sources=lambda: [SimpleNamespace(assets=[SimpleNamespace(data_uri="file://localhost/C:/data/frame%20a.h5")])])
    assert _registered_file(node) == "C:/data/frame a.h5"
    assert _registered_file(SimpleNamespace(data_sources=lambda: [])) is None


def test_a_failed_reconcile_is_retried_on_the_next_check(monkeypatch, tmp_path) -> None:
    def down(uri):
        raise TiledUnavailable("not answering")

    monkeypatch.setattr("asyncroscopy.data.tiled_client.connect", down)
    device = SimpleNamespace(_save_path=str(tmp_path), _host="127.0.0.1", _port=CLOSED_PORT, _reconcile_due=None, _tiled_server="yes", _tiled_server_status="")
    device._uri = lambda: DATA._uri(device)
    with pytest.raises(RuntimeError, match="reconcile failed"):
        DATA.reconcile(device)
    assert device._reconcile_due is not None


def test_only_a_tiled_whose_starter_is_gone_counts_as_orphaned() -> None:
    gone = SimpleNamespace(parent=lambda: None)
    assert _live_owner(gone) is None
    launcher = SimpleNamespace(pid=5, name=lambda: "python.exe", cmdline=lambda: ["python.exe", "-m", "tiled", "serve", "catalog"], parent=lambda: None)
    assert _live_owner(SimpleNamespace(parent=lambda: launcher)) is None  # a venv launcher of the same command is stepped over
    shell = SimpleNamespace(pid=6, name=lambda: "pwsh.exe", cmdline=lambda: ["pwsh.exe"], parent=lambda: None)
    assert _live_owner(SimpleNamespace(parent=lambda: shell)) == "pwsh.exe pid 6"

    def denied():
        raise data_module.psutil.AccessDenied()

    assert "cannot be inspected" in _live_owner(SimpleNamespace(parent=denied))


def test_deferral_uses_the_same_rule_as_reconcile(tmp_path) -> None:
    save = str(tmp_path)
    assert _reconcilable(str(tmp_path / "maps" / "map_1.h5"), save)
    assert not _reconcilable(str(tmp_path / ".cache" / "a.h5"), save)
    assert not _reconcilable(str(tmp_path / "map_x_map" / "a.h5"), save)
    assert not _reconcilable(str(tmp_path / "map_x_map_20261029T023056_33d6" / "a.h5"), save)
    assert not _reconcilable(str(tmp_path / "map_x_map_20261030T174128543Z_b1be" / "a.h5"), save)
    assert not _reconcilable(str(tmp_path / "map_x_points" / "a.h5"), save)
    assert _reconcilable(str(tmp_path / "map_x_mapping" / "a.h5"), save)
    assert not _reconcilable(str(tmp_path / ".a.h5"), save)
    assert not _reconcilable(str(tmp_path.parent / "elsewhere.h5"), save)


def test_the_read_directory_is_resolved_against_the_process_directory(tmp_path) -> None:
    assert _read_directory(["tiled", "serve", "--read", "acq"], str(tmp_path)) == os.path.join(str(tmp_path), "acq")
    assert _read_directory(["tiled", "serve", "--read", str(tmp_path)], "elsewhere") == str(tmp_path)
    assert _read_directory(["tiled", "serve"], str(tmp_path)) is None


def test_acquisition_files_skip_hidden_files_and_tiled_storage(tmp_path) -> None:
    (tmp_path / "a.h5").write_bytes(b"x")
    (tmp_path / "b.tiff").write_bytes(b"x")
    (tmp_path / ".c.h5").write_bytes(b"x")
    (tmp_path / "d.partial").write_bytes(b"x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "e.h5").write_bytes(b"x")

    assert [p.name for p in _acquisition_files(tmp_path)] == ["a.h5", "b.tiff"]
