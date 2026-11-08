"""asyncroscopy.data.tiled_client: Tiled detection, retries and errors an agent can act on."""

import io
import json
from types import SimpleNamespace

import httpx
import pytest

from asyncroscopy.data import tiled_client
from asyncroscopy.data.tiled_client import DataKeyNotFound, TiledUnavailable


@pytest.fixture(autouse=True)
def fresh_caches():
    tiled_client._clients.clear()
    tiled_client._uris.clear()
    yield
    tiled_client._clients.clear()
    tiled_client._uris.clear()


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def answer(monkeypatch, body: bytes):
    monkeypatch.setattr(tiled_client, "_nothing_listening", lambda uri: False)
    monkeypatch.setattr(tiled_client, "urlopen", lambda request, timeout: Response(body))


def test_about_accepts_a_tiled_about_document(monkeypatch):
    answer(monkeypatch, json.dumps({"api_version": 0, "library_version": "0.2.20", "links": {}}).encode())
    assert tiled_client.about("http://tiled.example:9091")["library_version"] == "0.2.20"


@pytest.mark.parametrize("body", [b"<html>a web page</html>", b'{"status": "ok"}', b"[]"])
def test_about_rejects_a_server_that_is_not_tiled(monkeypatch, body):
    answer(monkeypatch, body)
    assert tiled_client.about("http://tiled.example:9091") is None


def listening_on(monkeypatch, ports):
    import psutil

    connections = [SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=p)) for p in ports]
    monkeypatch.setattr(psutil, "net_connections", lambda kind: connections)


def test_about_answers_at_once_when_nothing_listens_locally(monkeypatch):
    listening_on(monkeypatch, [22])
    monkeypatch.setattr(tiled_client, "urlopen", lambda request, timeout: pytest.fail("should not connect"))
    assert tiled_client.about("http://127.0.0.1:9") is None


def test_only_local_addresses_are_checked_for_a_listener(monkeypatch):
    listening_on(monkeypatch, [9091])
    assert tiled_client._nothing_listening("http://127.0.0.1:9")
    assert not tiled_client._nothing_listening("http://localhost:9091")
    assert not tiled_client._nothing_listening("http://tiled.example:9")


def test_when_the_os_will_not_list_listeners_it_falls_back_to_connecting(monkeypatch):
    import psutil

    def denied(kind):
        raise psutil.AccessDenied()  # macOS without root

    monkeypatch.setattr(psutil, "net_connections", denied)
    assert not tiled_client._nothing_listening("http://127.0.0.1:9")


def test_only_failures_to_reach_tiled_count_as_unavailable():
    assert issubclass(httpx.ConnectError, tiled_client.CONNECTION_ERRORS)
    assert issubclass(ConnectionRefusedError, tiled_client.CONNECTION_ERRORS)
    for other in (FileExistsError, FileNotFoundError, TimeoutError, PermissionError):
        assert not issubclass(other, tiled_client.CONNECTION_ERRORS)


def test_the_tiled_client_retries_only_briefly():
    from tiled.client import utils

    assert utils.TILED_RETRY_ATTEMPTS <= 2 and utils.TILED_RETRY_TIMEOUT <= 3


def tiled_http():
    """The HTTP library the installed Tiled client sends requests with (httpx, or httpx2 from tiled 0.2.21)."""
    from tiled.client import context

    return context.httpx


def test_connect_retries_once_then_names_the_uri(monkeypatch):
    attempts, sleeps = [], []

    def refuse(uri, **kwargs):
        attempts.append(kwargs)
        raise tiled_http().ConnectError("connection refused")

    monkeypatch.setattr("tiled.client.from_uri", refuse)
    monkeypatch.setattr(tiled_client.time, "sleep", sleeps.append)

    with pytest.raises(TiledUnavailable, match=r"Tiled at http://tiled.example:9091 is not answering"):
        tiled_client.connect("http://tiled.example:9091")
    assert len(attempts) == 2 and sleeps == [0.5]
    assert attempts[0]["api_key"] == "secret"


def test_the_timeout_is_one_the_tiled_client_can_use():
    """Tiled 0.2.21 moved from httpx to httpx2, and an httpx.Timeout then failed every connect with a TypeError."""
    timeout = tiled_client.tiled_timeout()
    assert isinstance(timeout, tiled_http().Timeout)
    assert (timeout.connect, timeout.read) == (tiled_client.TIMEOUT_S["connect"], tiled_client.TIMEOUT_S["read"])
    with tiled_http().Client(timeout=timeout) as client:
        with pytest.raises(tiled_http().TransportError):
            client.get("http://127.0.0.1:9")  # nothing listens on the discard port: a transport error, not a TypeError


def test_connect_fails_at_once_when_nothing_listens_locally(monkeypatch):
    listening_on(monkeypatch, [22])
    monkeypatch.setattr("tiled.client.from_uri", lambda uri, **kwargs: pytest.fail("should not connect"))
    with pytest.raises(TiledUnavailable, match="nothing is listening"):
        tiled_client.connect("http://127.0.0.1:9")


def test_connect_caches_until_forgotten(monkeypatch):
    made = []
    monkeypatch.setattr("tiled.client.from_uri", lambda uri, **kwargs: made.append(uri) or object())

    first = tiled_client.connect("http://tiled.example:9091")
    assert tiled_client.connect("http://tiled.example:9091") is first
    tiled_client.forget("http://tiled.example:9091")
    assert tiled_client.connect("http://tiled.example:9091") is not first
    assert len(made) == 2


def test_the_api_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_TILED_API_KEY", "lab-key")
    assert tiled_client.api_key() == "lab-key"


class Root(dict):
    def keys(self):
        return list(super().keys())


def test_node_for_key_names_keys_that_start_with_a_stem(monkeypatch):
    # TIFF acquisitions return a stem; the registered keys are <stem>_<detector>.tiff.
    root = Root({"stem_image_1_HAADF.tiff": "haadf", "stem_image_1_BF.tiff": "bf", "other.h5": "x"})
    monkeypatch.setattr(tiled_client, "connect", lambda uri: root)

    assert tiled_client.node_for_key("other.h5", "http://tiled.example") == "x"
    with pytest.raises(DataKeyNotFound) as error:
        tiled_client.node_for_key("stem_image_1", "http://tiled.example")
    message = str(error.value)
    assert "stem_image_1_HAADF.tiff" in message and "stem_image_1_BF.tiff" in message and "other.h5" not in message


def test_node_for_key_turns_a_dropped_connection_into_tiled_unavailable(monkeypatch):
    class Dropping:
        def __getitem__(self, key):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(tiled_client, "connect", lambda uri: Dropping())
    tiled_client._clients["http://tiled.example"] = Dropping()

    with pytest.raises(TiledUnavailable, match="ReadTimeout"):
        tiled_client.node_for_key("a.h5", "http://tiled.example")
    assert "http://tiled.example" not in tiled_client._clients


def test_uri_from_a_busy_data_device_falls_back_to_the_cached_one(monkeypatch):
    class Proxy:
        busy = False

        def __init__(self, address):
            pass

        def set_timeout_millis(self, millis):
            pass

        def get_config(self):
            if Proxy.busy:
                raise RuntimeError("API_DeviceTimedOut")
            return json.dumps({"uri": "http://127.0.0.1:9091"})

    monkeypatch.setattr("tango.DeviceProxy", Proxy)
    assert tiled_client.uri_from_data_device("asyncroscopy/data/default") == "http://127.0.0.1:9091"
    Proxy.busy = True
    tiled_client._uris["asyncroscopy/data/default"] = ("http://127.0.0.1:9091", 0.0)  # expired, so it asks again
    assert tiled_client.uri_from_data_device("asyncroscopy/data/default") == "http://127.0.0.1:9091"


def test_uri_without_a_data_device_uses_the_environment(monkeypatch):
    class Proxy:
        def __init__(self, address):
            raise RuntimeError("device not exported")

    monkeypatch.setattr("tango.DeviceProxy", Proxy)
    monkeypatch.setenv("ASYNCROSCOPY_TILED_URI", "http://127.0.0.1:9191")
    assert tiled_client.uri_from_data_device("asyncroscopy/data/default") == "http://127.0.0.1:9191"
    monkeypatch.delenv("ASYNCROSCOPY_TILED_URI")
    with pytest.raises(TiledUnavailable, match="device not exported"):
        tiled_client.uri_from_data_device("asyncroscopy/data/default")
