import json
import numpy as np
import pytest
import tango


class TestThermoMicroscope:
    def test_startup_state_is_on(self, thermo_proxy: tango.DeviceProxy) -> None:
        assert thermo_proxy.state() == tango.DevState.ON

    def test_haadf_defaults_are_visible_through_proxy(self, haadf_proxy: tango.DeviceProxy) -> None:
        haadf_proxy.dwell_time = 1e-6
        haadf_proxy.image_width = 1024
        haadf_proxy.image_height = 1024
        assert haadf_proxy.state() == tango.DevState.ON
        assert haadf_proxy.dwell_time == pytest.approx(1e-6)
        assert haadf_proxy.image_width == 1024
        assert haadf_proxy.image_height == 1024

    def test_get_image_returns_valid_encoded_data(
        self,
        thermo_proxy: tango.DeviceProxy,
        patched_single_image: pytest.MonkeyPatch,
    ) -> None:
        json_meta, raw_bytes = thermo_proxy.get_image("haadf")
        meta = json.loads(json_meta)

        assert meta["detector"] == "haadf"
        assert meta["shape"] == [1024, 1024]
        assert meta["dtype"] == "uint16"
        assert meta["dwell_time"] == pytest.approx(1e-6)
        assert "timestamp" in meta

        image = np.frombuffer(raw_bytes, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
        assert image.shape == (1024, 1024)
        assert image.dtype == np.uint16

    def test_detector_settings_propagate_into_get_image(
        self,
        thermo_proxy: tango.DeviceProxy,
        haadf_proxy: tango.DeviceProxy,
        patched_single_image: pytest.MonkeyPatch,
    ) -> None:
        haadf_proxy.dwell_time = 2e-6
        haadf_proxy.image_width = 256
        haadf_proxy.image_height = 512

        json_meta, raw_bytes = thermo_proxy.get_image("haadf")
        meta = json.loads(json_meta)

        assert meta["detector"] == "haadf"
        assert meta["shape"] == [512, 256]
        assert meta["dtype"] == "uint16"
        assert meta["dwell_time"] == pytest.approx(2e-6)

        image = np.frombuffer(raw_bytes, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
        assert image.shape == (512, 256)
        assert image.dtype == np.uint16

    def test_unknown_detector_raises(self, thermo_proxy: tango.DeviceProxy, patched_single_image: pytest.MonkeyPatch) -> None:
        with pytest.raises(tango.DevFailed) as exc:
            thermo_proxy.get_image("void")

        err_text = str(exc.value)

        # If detector is not configured (or name is misspelled),
        # proxy lookup returns None and attribute access fails.
        assert "NoneType" in err_text
        assert "dwell_time" in err_text

    def test_disconnect_sets_state_off(self, thermo_proxy: tango.DeviceProxy) -> None:
        thermo_proxy.Disconnect()
        assert thermo_proxy.state() == tango.DevState.OFF

    def test_connect_restores_state_on(self, thermo_proxy: tango.DeviceProxy) -> None:
        thermo_proxy.Disconnect()
        assert thermo_proxy.state() == tango.DevState.OFF

        thermo_proxy.Connect()
        assert thermo_proxy.state() == tango.DevState.ON