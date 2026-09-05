import json
import subprocess
from pathlib import Path


import pytest
import tango

from asyncroscopy.data.data import DATA, _catalog_database_uri


class TestDataDevice:
    def test_state_is_on(self, data_proxy: tango.DeviceProxy) -> None:
        assert data_proxy.state() == tango.DevState.ON

    def test_config_round_trip(self, data_proxy: tango.DeviceProxy, tmp_path) -> None:
        config = {
            "host": "127.0.0.1",
            "port": 9091,
            "save_path": str(tmp_path),
        }

        returned = json.loads(data_proxy.configure(json.dumps(config)))

        assert returned["host"] == config["host"]
        assert returned["port"] == config["port"]
        assert returned["save_path"] == config["save_path"]
        assert returned["uri"] == "http://127.0.0.1:9091"

    def test_save_path_creates_missing_directory(self, data_proxy: tango.DeviceProxy, tmp_path) -> None:
        save_path = tmp_path / "new" / "acquisitions"

        data_proxy.save_path = str(save_path)

        assert save_path.is_dir()

    def test_start_tiled_server_uses_catalog_server_command(
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

        class FakeProcess:
            def __init__(self):
                self.running = True

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.running = False

            def wait(self, timeout):
                return 0

            def kill(self):
                self.running = False

        def fake_popen(command, **kwargs):
            popen_calls.append({"command": command, "kwargs": kwargs})
            return FakeProcess()

        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091
        data_proxy.save_path = str(tmp_path)
        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "asyncroscopy.data.data.subprocess.run",
            lambda command, **_: (
                run_commands.append(command)
                or type("Result", (), {"returncode": 0, "stdout": ""})()
            ),
        )

        returned = json.loads(data_proxy.start_tiled_server())

        assert returned["tiled_server"] == "yes"
        command_prefix = ["python", "-m"]
        key_value = popen_calls[0]["command"][10]
        expected_command = [
            *command_prefix,
            "tiled",
            "serve",
            "catalog",
            str(tmp_path / ".asyncroscopy_tiled_catalog.db"),
            "--read",
            str(tmp_path),
            "--public",
            "--api-key",
            key_value,
            "--host",
            "127.0.0.1",
            "--port",
            "9091",
        ]

        assert len(popen_calls) == 1
        actual_command = popen_calls[0]["command"]

        expected_catalog = _catalog_database_uri(tmp_path / ".asyncroscopy_tiled_catalog.db")

        assert actual_command[:5] == expected_command[:5]
        assert actual_command[5] == expected_catalog
        assert actual_command[6] == expected_command[6]
        assert Path(actual_command[7]) == Path(expected_command[7])
        assert actual_command[8:] == expected_command[8:]

        kwargs = dict(popen_calls[0]["kwargs"])
        assert "TILED_ALLOW_ORIGINS" in kwargs.pop("env")
        assert kwargs == {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.STDOUT,
            "text": True,
        }

        assert len(run_commands) == 1
        assert run_commands[0][:6] == [
            *command_prefix,
            "tiled",
            "catalog",
            "init",
            "--if-not-exists",
        ]
        assert run_commands[0][6] == expected_catalog
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
    ) -> None:
        calls = []
        popen_calls = []
        run_commands = []

        def fake_alive(self):
            calls.append(None)
            return len(calls) > 1

        class FakeProcess:
            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", lambda command, **kwargs: popen_calls.append(command) or FakeProcess())
        monkeypatch.setattr(
            "asyncroscopy.data.data.subprocess.run",
            lambda command, **_: (
                run_commands.append(command)
                or type("Result", (), {"returncode": 0, "stdout": ""})()
            ),
        )
        # The Windows drive path below is fed only to exercise the
        # drive-path -> sqlite-URI branch; neutralize the mkdir side effect so
        # the test does not depend on the path being creatable (a non-admin user
        # cannot create directories under C:\, and the path can't exist on Linux).
        monkeypatch.setattr("asyncroscopy.data.data._ensure_directory", lambda path: None)

        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091
        data_proxy.save_path = "C:/tiled_catalog_test"
        calls.clear()

        returned = json.loads(data_proxy.start_tiled_server())

        expected_catalog = "sqlite:///C:/tiled_catalog_test/.asyncroscopy_tiled_catalog.db"
        assert returned["tiled_server"] == "yes"
        assert run_commands[0][6] == expected_catalog
        assert popen_calls[0][5] == expected_catalog
        data_proxy.stop_tiled_server()

    def test_register_path_registers_single_file(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        registrations = []
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091

        def fake_from_uri(*args, **kwargs):
            return object()

        async def fake_register(client, path, **kwargs):
            registrations.append(path)

        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        monkeypatch.setattr("asyncroscopy.data.data.open_client", fake_from_uri)
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        result = data_proxy.register_path(str(saved))

        assert result == "frame.h5"
        assert registrations == [str(saved)]

    def test_register_save_path_registers_only_missing_files(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        """register_save_path registers file-by-file and skips existing catalog
        entries — it must never hand tiled the whole directory, because
        directory-level register() deletes every existing catalog entry first."""
        registrations = []
        (tmp_path / "already.h5").write_bytes(b"fake-h5")
        (tmp_path / "missing_a.h5").write_bytes(b"fake-h5")
        (tmp_path / "missing_b.h5").write_bytes(b"fake-h5")
        # The managed catalog database is a dotfile inside the save path and
        # must never be registered as data.
        (tmp_path / ".asyncroscopy_tiled_catalog.db").write_bytes(b"sqlite")
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091
        data_proxy.save_path = str(tmp_path)

        class FakeClient:
            def keys(self):
                return ["already.h5", "unrelated_older_entry.h5"]

        async def fake_register(client, path, **kwargs):
            registrations.append(path)

        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        monkeypatch.setattr("asyncroscopy.data.data.open_client", lambda *args, **kwargs: FakeClient())
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        result = json.loads(data_proxy.register_save_path())

        assert result["registered_path"] == str(tmp_path)
        assert result["registered_files"] == 2
        assert result["already_registered"] == 1
        assert result["tiled_server_status"] == "running; registered save path"
        assert registrations == [
            str(tmp_path / "missing_a.h5"),
            str(tmp_path / "missing_b.h5"),
        ]

    def test_register_save_path_reports_failure_and_keeps_status(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        (tmp_path / "frame.h5").write_bytes(b"fake-h5")
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091
        data_proxy.save_path = str(tmp_path)

        class FakeClient:
            def keys(self):
                return []

        async def failing_register(client, path, **kwargs):
            raise ConnectionError("tiled is down")

        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        monkeypatch.setattr("asyncroscopy.data.data.open_client", lambda *args, **kwargs: FakeClient())
        monkeypatch.setattr("asyncroscopy.data.data.register", failing_register)

        with pytest.raises(tango.DevFailed):
            data_proxy.register_save_path()

        status = json.loads(data_proxy.get_config())["tiled_server_status"]
        assert "Save path registration failed:" in status
        assert "tiled is down" in status

    def test_register_path_returns_windows_tiled_key(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
    ) -> None:
        windows_path = "D:/microscopedata/tiled/ahoust17/frame.h5"
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091

        def fake_from_uri(*args, **kwargs):
            return object()

        async def fake_register(*args, **kwargs):
            return None

        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        monkeypatch.setattr("asyncroscopy.data.data.open_client", fake_from_uri)
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        assert data_proxy.register_path(windows_path) == "frame.h5"

    def test_register_path_waits_until_tiled_key_is_readable(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        saved = tmp_path / "frame.h5"
        saved.write_bytes(b"fake-h5")
        data_proxy.host = "127.0.0.1"
        data_proxy.port = 9091

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def __getitem__(self, key):
                self.calls += 1
                if self.calls < 3:
                    raise KeyError(key)
                return object()

        fake_client = FakeClient()
        sleeps = []

        def fake_from_uri(*args, **kwargs):
            return fake_client

        async def fake_register(*args, **kwargs):
            return None

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        monkeypatch.setattr("asyncroscopy.data.data.open_client", fake_from_uri)
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)
        monkeypatch.setattr("asyncroscopy.data.data.asyncio.sleep", fake_sleep)

        assert data_proxy.register_path(str(saved)) == "frame.h5"
        assert fake_client.calls == 3
        assert sleeps == [0.25, 0.25]

    def test_save_path_change_restarts_managed_server(
        self,
        data_proxy: tango.DeviceProxy,
        monkeypatch,
        tmp_path,
    ) -> None:
        popen_calls = []
        processes = []

        class FakeProcess:
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

        def fake_popen(command, **kwargs):
            process = FakeProcess()
            processes.append(process)
            popen_calls.append({"command": command, "kwargs": kwargs})
            return process

        def fake_alive(self):
            return self._tiled_process is not None and self._tiled_process.poll() is None

        run_commands = []
        monkeypatch.setattr(DATA, "_tiled_alive", fake_alive)
        monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "asyncroscopy.data.data.subprocess.run",
            lambda command, **_: (
                run_commands.append(command)
                or type("Result", (), {"returncode": 0, "stdout": ""})()
            ),
        )

        first_path = tmp_path / "first"
        second_path = tmp_path / "second"

        data_proxy.save_path = str(first_path)
        data_proxy.start_tiled_server()
        data_proxy.save_path = str(second_path)
        config = json.loads(data_proxy.get_config())


        assert processes[0].terminated is True
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

        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: True)
        monkeypatch.setattr("asyncroscopy.data.data.open_client", lambda *args, **kwargs: object())
        monkeypatch.setattr("asyncroscopy.data.data.register", fake_register)

        with pytest.raises(tango.DevFailed) as exc_info:
            data_proxy.register_path(str(requested_path))

        message = str(exc_info.value)
        status = json.loads(data_proxy.get_config())["tiled_server_status"]
        assert "File registration failed:" in message
        assert f"Requested file:\n    {requested_path}" in status
        assert f"Data save path:\n    {save_path}" in status
        assert "Tiled server serving:\n    " in status


class TestTiledBrowserOrigins:
    def test_managed_tiled_server_gets_allowed_origins(self, data_proxy: tango.DeviceProxy, monkeypatch, tmp_path) -> None:
        from asyncroscopy.data import tiled_client

        popen_calls = []

        class FakeProcess:
            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout):
                return 0

            def kill(self):
                pass

        monkeypatch.setenv(tiled_client.TILED_ALLOW_ORIGINS_ENV, "http://localhost:1420, http://gui.lab:5173")
        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: bool(popen_calls))
        monkeypatch.setattr(DATA, "_tiled_command", lambda self: ["python", "-m", "tiled"])
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.Popen", lambda command, **kwargs: popen_calls.append(kwargs) or FakeProcess())
        monkeypatch.setattr("asyncroscopy.data.data.subprocess.run", lambda command, **_: type("R", (), {"returncode": 0, "stdout": ""})())

        data_proxy.save_path = str(tmp_path)
        config = json.loads(data_proxy.start_tiled_server())
        data_proxy.stop_tiled_server()

        assert json.loads(popen_calls[0]["env"]["TILED_ALLOW_ORIGINS"]) == ["http://localhost:1420", "http://gui.lab:5173"]
        assert config["allow_origins"] == ["http://localhost:1420", "http://gui.lab:5173"]

    def test_default_origins_are_loopback_only(self, monkeypatch) -> None:
        from asyncroscopy.data import tiled_client

        monkeypatch.delenv(tiled_client.TILED_ALLOW_ORIGINS_ENV, raising=False)
        assert tiled_client.allowed_origins() == tiled_client.DEFAULT_BROWSER_ORIGINS
        assert all(origin.startswith(("http://localhost", "http://127.0.0.1")) for origin in tiled_client.allowed_origins())


class TestRegisterWithoutTiled:
    def test_register_path_returns_key_when_tiled_is_down(self, data_proxy: tango.DeviceProxy, monkeypatch, tmp_path) -> None:
        """The digital twin must acquire with no Tiled server at all."""
        monkeypatch.setattr(DATA, "_tiled_alive", lambda self: False)
        opened = []
        monkeypatch.setattr("asyncroscopy.data.data.open_client", lambda *a, **k: opened.append(a))

        acquisition = tmp_path / "stem_image_HAADF_20260904T120000000000.h5"
        acquisition.write_bytes(b"h5")
        data_proxy.set_timeout_millis(10_000)

        key = data_proxy.register_path(str(acquisition))

        assert key == acquisition.name
        assert opened == [], "no Tiled client may be opened while the server is down"
        config = json.loads(data_proxy.get_config())
        assert config["tiled_server"] == "no"
        assert "1 acquisition(s) saved but not registered" in config["tiled_server_status"]
