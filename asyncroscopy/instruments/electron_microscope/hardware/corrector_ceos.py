"""CEOS JSON-RPC/TCP implementation of the generic corrector interface."""

from __future__ import annotations

import json
import socket
from typing import Optional

import tango
from tango import DevState
from tango.server import device_property

from asyncroscopy.instruments.electron_microscope.hardware.corrector import CORRECTOR, FIRST_ORDER


class CEOSCorrector(CORRECTOR):
    """Forward the common corrector API to a physical CEOS server."""

    ceos_host = device_property(
        dtype=str,
        default_value="10.46.217.241",
        doc="Hostname or IP address of the CEOS hardware server",
    )
    ceos_port = device_property(
        dtype=int,
        default_value=9092,
        doc="TCP port of the CEOS hardware server",
    )
    socket_timeout = device_property(
        dtype=float,
        default_value=30.0,
        doc="Socket timeout in seconds for CEOS communication: a CEOS server that does not answer in this time "
        "fails the call instead of holding the corrector device.",
    )

    def init_device(self) -> None:
        self._message_id = 1
        super().init_device()

    def _connect_backend(self) -> None:
        try:
            with socket.create_connection(
                (self.ceos_host, int(self.ceos_port)),
                timeout=float(self.socket_timeout),
            ):
                pass
            self._last_status = "Connected"
            self.set_state(DevState.ON)
        except OSError as exc:
            self._last_status = f"Connection failed: {exc}"
            self.set_state(DevState.FAULT)
            self.error_stream(f"Cannot reach CEOS server: {exc}")

    def _get_info(self) -> str:
        return self._call("getInfo")

    def _acquire_tableau(self, tab_type: str, angle: float) -> str:
        return self._call("acquireTableau", {"tabType": tab_type, "angle": angle})

    def _measure_c1a1(self) -> str:
        return self._call("measureC1A1")

    def _correct_aberration(self, name: str, values: list[float]) -> str:
        return self._call("correctAberration", {"name": name, "value": values})

    def _measure_aberrations(self) -> dict[str, list[float]]:
        result = json.loads(self._measure_c1a1()).get("result") or {}
        aberrations = result.get("aberrations", result)
        return {name: [float(v) for v in (aberrations[name] if isinstance(aberrations[name], list) else [aberrations[name]])] for name in FIRST_ORDER}

    def _correct(self, name: str, values: list[float]) -> None:
        self._correct_aberration(name, values)  # a CEOS refusal raises in _call

    def _call(self, method: str, params: Optional[dict] = None) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": self._message_id,
            "method": method,
            "params": params or {},
        }
        self._message_id += 1
        netstring = self._encode_netstring(payload)
        try:
            with socket.create_connection(
                (self.ceos_host, int(self.ceos_port)),
                timeout=float(self.socket_timeout),
            ) as sock:
                sock.sendall(netstring)
                raw_response = self._recv_netstring(sock)
            result = self._decode_netstring(raw_response)
        except OSError as exc:
            self._last_status = f"Error: {exc}"
            self.set_state(DevState.FAULT)
            tango.Except.throw_exception("CeosCommError", str(exc), "CEOSCorrector._call()")
            raise RuntimeError("unreachable")
        # The server answered: a refusal comes back as the JSON-RPC error member, and must not read as success.
        error = self._rpc_error(result)
        self.set_state(DevState.ON)
        if error is not None:
            self._last_status = f"CEOS refused {method}: {error}"
            tango.Except.throw_exception("CeosError", f"CEOS refused {method}: {error}", "CEOSCorrector._call()")
        self._last_status = "OK"
        return result

    @staticmethod
    def _rpc_error(reply: str) -> str | None:
        """What is wrong with a CEOS JSON-RPC reply (its error member, or a reply that is not JSON-RPC); None when it succeeded."""
        try:
            body = json.loads(reply)
        except ValueError:
            return f"the reply is not JSON: {reply[:200]!r}"
        if not isinstance(body, dict):
            return f"the reply is not a JSON-RPC object: {reply[:200]!r}"
        if body.get("error") is not None:
            return json.dumps(body["error"])[:500]
        return None

    @staticmethod
    def _encode_netstring(payload: dict) -> bytes:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return f"{len(body)}:".encode("ascii") + body + b","

    @staticmethod
    def _decode_netstring(raw: bytes) -> str:
        value = raw.decode("utf-8").strip()
        if ":" in value and value.split(":", 1)[0].isdigit():
            _, value = value.split(":", 1)
        return value.rstrip(",")

    @staticmethod
    def _recv_netstring(sock: socket.socket, bufsize: int = 4096) -> bytes:
        header = b""
        while b":" not in header:
            chunk = sock.recv(bufsize)
            if not chunk:
                raise OSError("CEOS connection closed before netstring header")
            header += chunk
        length_bytes, payload = header.split(b":", 1)
        if not length_bytes.isdigit():
            raise OSError("Invalid CEOS netstring length")
        expected = int(length_bytes)
        while len(payload) < expected + 1:
            chunk = sock.recv(bufsize)
            if not chunk:
                raise OSError("CEOS connection closed before complete netstring")
            payload += chunk
        if payload[expected : expected + 1] != b",":
            raise OSError("Invalid CEOS netstring terminator")
        return length_bytes + b":" + payload[: expected + 1]


# Compatibility aliases for older imports.
CeosCorrector = CEOSCorrector


if __name__ == "__main__":
    CEOSCorrector.run_server()
