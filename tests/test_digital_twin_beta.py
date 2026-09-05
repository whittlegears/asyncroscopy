"""Tests for the DigitalTwinBeta (double-tilt) Tango device."""

import json
from pathlib import Path

import h5py
import pytest
import tango


class TestDigitalTwinBeta:

    def test_state_is_on(self, beta_twin_proxy: tango.DeviceProxy):
        assert beta_twin_proxy.state() == tango.DevState.ON

    def test_defocus_commands_round_trip(self, beta_twin_proxy: tango.DeviceProxy):
        beta_twin_proxy.set_defocus(8e-9)

        assert beta_twin_proxy.get_defocus() == pytest.approx(8e-9)

    def test_column_valves_round_trip(self, beta_twin_proxy: tango.DeviceProxy):
        beta_twin_proxy.set_column_valves("open")
        assert json.loads(beta_twin_proxy.get_parameters())["column_valves_state"] == "open"

        beta_twin_proxy.set_column_valves("close")
        assert json.loads(beta_twin_proxy.get_parameters())["column_valves_state"] == "close"

    def test_get_parameters_returns_status_json(self, beta_twin_proxy: tango.DeviceProxy):
        parameters = json.loads(beta_twin_proxy.get_parameters())

        assert parameters["manufacturer"] == "UTKTeam"
        assert "defocus_m" in parameters
        assert "stage_position" in parameters

    def test_beta_tilt_round_trips_through_stage(
        self, beta_twin_proxy: tango.DeviceProxy, scan_proxy: tango.DeviceProxy,
    ):
        beta_twin_proxy.move_stage([0.0, 0.0, 0.0, 5.0, 12.0])

        position = beta_twin_proxy.get_stage()

        assert position[3] == pytest.approx(5.0)
        assert position[4] == pytest.approx(12.0)

    def test_get_image_returns_saved_hdf5(
        self, beta_twin_proxy: tango.DeviceProxy, scan_proxy: tango.DeviceProxy,
    ):
        scan_proxy.imsize = 16
        scan_proxy.dwell_time = 1e-6

        saved_path = Path(beta_twin_proxy.acquire_scanned_image(["haadf"]))

        assert saved_path.suffix == ".h5"
        assert saved_path.exists()
        with h5py.File(saved_path, "r") as h5:
            image = h5["image/HAADF"][()]
            assert image.shape == (16, 16)


class TestBetaTwinSpectrumGoesThroughDataWriter:
    def test_acquire_spectrum_returns_saved_spectrum_key(self, beta_twin_proxy: tango.DeviceProxy) -> None:
        import h5py

        key = beta_twin_proxy.acquire_spectrum("eds")

        # No DATA device is configured in the test context, so the key is the local file path.
        assert key.endswith(".h5") and "spectrum_eds_" in key
        with h5py.File(key, "r") as h5:
            elements = json.loads(h5["spectrum"].attrs["elements"])
            assert len(elements) == h5["spectrum"].shape[0] > 0
            assert h5.attrs["instrument_class"] == "DigitalTwinBeta"
            assert "acquisition_id" in h5.attrs
