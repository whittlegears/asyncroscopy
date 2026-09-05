"""Tests for the simulated TestCorrector device.

TestCorrector inherits every Tango command from CORRECTOR unchanged (that
parity is asserted in test_digital_twin_parity.py); these tests verify the
simulated CEOS responses keep the real server's JSON-RPC 2.0 formatting.
"""

import json

import pytest
import tango


def rpc_result(raw: str) -> dict:
    """Decode a corrector command reply and assert the JSON-RPC envelope."""
    payload = json.loads(raw)
    assert payload["jsonrpc"] == "2.0"
    assert isinstance(payload["id"], int)
    return payload["result"]


class TestCorrectorDevice:
    def test_state_is_on_without_hardware(self, corrector_proxy: tango.DeviceProxy) -> None:
        assert corrector_proxy.state() == tango.DevState.ON
        assert "simulated" in corrector_proxy.status_message

    def test_get_info_reports_simulated_corrector(self, corrector_proxy: tango.DeviceProxy) -> None:
        info = rpc_result(corrector_proxy.get_info())

        assert "simulated" in info["type"]
        assert "C1" in info["aberrations"] and "A1" in info["aberrations"]

    def test_measure_c1a1_returns_coefficients(self, corrector_proxy: tango.DeviceProxy) -> None:
        measured = rpc_result(corrector_proxy.measure_c1a1())

        assert len(measured["C1"]) == 1
        assert len(measured["A1"]) == 2

    def test_acquire_tableau_echoes_arguments(self, corrector_proxy: tango.DeviceProxy) -> None:
        tableau = rpc_result(corrector_proxy.acquire_tableau("Fast 18"))

        assert tableau["tabType"] == "Fast"
        assert tableau["angle"] == pytest.approx(18.0)
        assert set(tableau["coefficients"]) >= {"C1", "A1", "C3"}

    def test_correct_aberration_updates_state(self, corrector_proxy: tango.DeviceProxy) -> None:
        original = rpc_result(corrector_proxy.measure_c1a1())["A1"]

        corrected = rpc_result(corrector_proxy.correct_aberration("A1 1e-9 -2e-9"))
        assert corrected["value"] == pytest.approx([1e-9, -2e-9])

        measured = rpc_result(corrector_proxy.measure_c1a1())["A1"]
        assert measured == pytest.approx([1e-9, -2e-9])

        corrector_proxy.correct_aberration(f"A1 {original[0]} {original[1]}")

    def test_correct_aberration_validates_input(self, corrector_proxy: tango.DeviceProxy) -> None:
        with pytest.raises(tango.DevFailed, match="Unknown aberration"):
            corrector_proxy.correct_aberration("Z9 1e-9")
        with pytest.raises(tango.DevFailed, match="value"):
            # C1 takes one value, not two
            corrector_proxy.correct_aberration("C1 1e-9 2e-9")

    def test_reconnect_is_a_noop_simulated_connect(self, corrector_proxy: tango.DeviceProxy) -> None:
        # command_inout, not proxy.reconnect(): DeviceProxy has a client-side
        # reconnect() method that shadows the Tango command of the same name.
        # (The MCP bridge invokes commands via command_inout for this reason.)
        corrector_proxy.command_inout("reconnect")
        assert corrector_proxy.state() == tango.DevState.ON

    def test_sim_coefficient_round_trip(self, corrector_proxy: tango.DeviceProxy) -> None:
        # Unset returns JSON null (a DevString command cannot return None)
        assert json.loads(corrector_proxy.get_aberrations_coeff_sim()) is None

        coefficients = {"C1": [1e-9], "A1": [2e-9, 3e-9]}
        corrector_proxy.set_aberrations_coeff_sim(json.dumps(coefficients))
        assert json.loads(corrector_proxy.get_aberrations_coeff_sim()) == coefficients


class FakeDataServer:
    def __init__(self, save_path):
        self.save_path = str(save_path)
        self.registered = []

    def register_path(self, path: str) -> str:
        self.registered.append(path)
        return path


class TestCorrectorArchiving:
    """Tableau and C1/A1 results are archived to DATA/Tiled when a DATA device is configured."""

    def make_corrector(self, data_server):
        from asyncroscopy.instruments.electron_microscope.hardware.TestCorrector import TestCorrector, _ABERRATION_DEFAULTS

        corrector = TestCorrector.__new__(TestCorrector)
        corrector._message_id = 1
        corrector._last_status = "Connected (simulated)"
        corrector._aberrations = {name: list(values) for name, values in _ABERRATION_DEFAULTS.items()}
        corrector._data_proxy = data_server
        corrector._last_archived_key = ""
        return corrector

    def test_measure_c1a1_is_archived_and_return_value_unchanged(self, tmp_path) -> None:
        import h5py

        data_server = FakeDataServer(tmp_path)
        corrector = self.make_corrector(data_server)

        raw = corrector.measure_c1a1()
        measured = rpc_result(raw)
        assert set(measured) == {"C1", "A1"}

        key = corrector.get_last_archived_key()
        assert key and key == data_server.registered[-1]
        assert "c1a1_corrector_" in key
        with h5py.File(key, "r") as h5:
            assert h5["coefficients"][()].tolist() == measured["C1"] + measured["A1"]
            assert json.loads(h5["coefficients"].attrs["result"]) == measured
            assert h5.attrs["acquisition_type"] == "c1a1"

    def test_tableau_archives_coefficients(self, tmp_path) -> None:
        data_server = FakeDataServer(tmp_path)
        corrector = self.make_corrector(data_server)

        tableau = rpc_result(corrector.acquire_tableau("Fast 18"))
        assert tableau["tabType"] == "Fast"
        assert corrector.get_last_archived_key().split("\\")[-1].split("/")[-1].startswith("tableau_corrector_")

    def test_without_data_device_nothing_is_archived(self) -> None:
        corrector = self.make_corrector(None)
        corrector._tango_properties = {"data_device_address": ""}

        rpc_result(corrector.measure_c1a1())
        assert corrector.get_last_archived_key() == ""
