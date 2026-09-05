"""
Tests for the DigitalTwin Tango device.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

class TestDigitalTwin:

    def test_state_is_on(self, twin_proxy: tango.DeviceProxy):
        assert twin_proxy.state() == tango.DevState.ON

    def test_manufacturer_is_digital_twin(self, twin_proxy: tango.DeviceProxy):
        assert twin_proxy.manufacturer == "UTKTeam"

    def test_defocus_commands_round_trip(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_defocus(8e-9)

        assert twin_proxy.get_defocus() == pytest.approx(8e-9)

    def test_column_valves_round_trip(self, twin_proxy: tango.DeviceProxy):
        import json

        twin_proxy.set_column_valves("open")
        assert json.loads(twin_proxy.get_parameters())["column_valves_state"] == "open"

        twin_proxy.set_column_valves("close")
        assert json.loads(twin_proxy.get_parameters())["column_valves_state"] == "close"

    def test_column_valves_rejects_invalid_state(self, twin_proxy: tango.DeviceProxy):
        with pytest.raises(tango.DevFailed):
            twin_proxy.set_column_valves("sideways")

    def test_fov_commands_round_trip(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_fov(5e-8)

        assert twin_proxy.get_fov() == pytest.approx(5e-8)

    def test_image_shift_commands_round_trip(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_image_shift([1e-9, -2e-9])

        assert twin_proxy.get_image_shift() == pytest.approx([1e-9, -2e-9])

    def test_beam_tilt_commands_round_trip(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_beam_tilt([0.01, -0.02])

        assert twin_proxy.get_beam_tilt() == pytest.approx([0.01, -0.02])

    def test_diffraction_shift_commands_round_trip(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_diffraction_shift([0.03, -0.04])

        assert twin_proxy.get_diffraction_shift() == pytest.approx([0.03, -0.04])

    def test_screen_commands_round_trip(self, twin_proxy: tango.DeviceProxy):
        import json

        twin_proxy.set_screen("in")
        twin_proxy.set_screen_current(75.0)
        twin_proxy.calibrate_screen_current()

        assert twin_proxy.get_screen_current() == pytest.approx(75.0)
        assert json.loads(twin_proxy.get_parameters())["screen_position"] == "in"

    def test_auto_focus_zeroes_defocus(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_defocus(1e-7)

        twin_proxy.auto_focus()

        assert twin_proxy.get_defocus() == pytest.approx(0.0)

    def test_get_parameters_returns_status_json(self, twin_proxy: tango.DeviceProxy):
        import json

        parameters = json.loads(twin_proxy.get_parameters())

        assert parameters["manufacturer"] == "UTKTeam"
        assert parameters["stem_mode"] is True
        assert "defocus_m" in parameters
        assert parameters["column_valves_state"] == "close"
        assert "stage_position" in parameters
        assert "fov_m" in parameters
        assert parameters["scan_detectors"] == ["haadf"]
        assert parameters["spectrum_detectors"] == ["eds"]
        assert "BM-Ceta" in parameters["camera_detectors"]
        assert "scan" in parameters["device_proxies"]
        assert "corrector" in parameters["device_proxies"]
        assert "detectors" not in parameters, "device roles must not be published as detectors"

    def test_get_image_returns_saved_hdf5(self, twin_proxy: tango.DeviceProxy, scan_proxy: tango.DeviceProxy):
        scan_proxy.imsize = 32
        scan_proxy.dwell_time = 1e-6

        saved_path = Path(twin_proxy.acquire_scanned_image(["haadf"]))

        assert saved_path.suffix == ".h5"
        assert saved_path.exists()
        with h5py.File(saved_path, "r") as h5:
            image = h5["image/HAADF"][()]
            assert image.shape == (32, 32)
            assert h5["image/HAADF"].attrs["acquisition_type"] == "stem_image"
            assert h5["image/HAADF"].attrs["detector"] == "HAADF"

    def test_acquire_scanned_image_rejects_unknown_detector(
        self, twin_proxy: tango.DeviceProxy, scan_proxy: tango.DeviceProxy
    ):
        # Observed live: an agent passed 'STEM' (a mode, not a detector) and got
        # an opaque AutoScript server error. The device must reject unknown
        # names itself with the valid vocabulary in the message.
        scan_proxy.imsize = 32
        scan_proxy.dwell_time = 1e-6

        with pytest.raises(tango.DevFailed) as excinfo:
            twin_proxy.acquire_scanned_image(["STEM"])
        message = str(excinfo.value)
        assert "STEM" in message
        assert "HAADF" in message

    def test_acquire_scanned_image_accepts_underscore_alias(
        self, twin_proxy: tango.DeviceProxy, scan_proxy: tango.DeviceProxy
    ):
        scan_proxy.imsize = 32
        scan_proxy.dwell_time = 1e-6

        with h5py.File(twin_proxy.acquire_scanned_image(["bf_s"]), "r") as h5:
            assert h5["image/BF-S"][()].shape == (32, 32)

    def test_stage_navigation_changes_and_restores_view(
        self,
        twin_proxy: tango.DeviceProxy,
        scan_proxy: tango.DeviceProxy,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def fake_stage_render(self, imsize: int, dwell_time: float, detector_list: list):
            self._sync_stage_from_proxy()
            stage_signal = int(round((self._stage_position[0] - self._stage_position[1]) * 1e10))
            return np.full((imsize, imsize), stage_signal, dtype=np.int16)

        from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin

        monkeypatch.setattr(DigitalTwin, "_render_stem_image", fake_stage_render)

        scan_proxy.imsize = 64
        scan_proxy.dwell_time = 1e-6

        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            image_a = h5["image/HAADF"][()]

        twin_proxy.move_stage([8e-9, -7e-9, 0.0, 0.0, 0.0])
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            image_b = h5["image/HAADF"][()]
        assert not np.array_equal(image_a, image_b)

        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            image_a_again = h5["image/HAADF"][()]
        assert np.array_equal(image_a, image_a_again)

    def test_spectrum_is_repeatable_at_same_pose_and_beam(
        self,
        twin_proxy: tango.DeviceProxy,
        eds_proxy: tango.DeviceProxy,
    ):
        eds_proxy.exposure_time = 0.05
        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        twin_proxy.place_beam([0.45, 0.55])

        with h5py.File(twin_proxy.acquire_spectrum("eds"), "r") as h5:
            spec_1 = h5["spectrum"][()]
        with h5py.File(twin_proxy.acquire_spectrum("eds"), "r") as h5:
            spec_2 = h5["spectrum"][()]
        assert spec_1.tolist() == spec_2.tolist()
