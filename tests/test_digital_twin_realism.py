"""
Tests for the DigitalTwin realism layer: imperfect stage moves, travel limits,
thermal drift, and beam-state gating of acquisitions.

These run against the `asyncroscopy/digitaltwin/realistic` device from
conftest, which enables the same knobs configs/DigitalTwin.yaml turns on for
deployments (with values scaled up so they are observable in fast tests).
"""

import time

import h5py
import numpy as np
import pytest
import tango


class TestDigitalTwinRealism:

    def test_move_lands_near_but_not_exactly_on_target(self, realistic_twin_proxy: tango.DeviceProxy):
        realistic_twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        realistic_twin_proxy.move_stage([5e-6, 0.0, 0.0, 0.0, 0.0])

        position = np.asarray(realistic_twin_proxy.get_stage(), dtype=np.float64)

        # get_stage travels over a float32 Tango wire, so compare against what
        # an ideal stage would return, not the float64 target.
        ideal = float(np.float32(5e-6))
        assert position[0] != ideal, "a real stage never lands exactly on target"
        assert abs(position[0] - 5e-6) < 1e-6
        assert abs(position[1]) < 1e-6
        # Tilts are not subject to the x/y/z noise model
        assert position[3] == pytest.approx(0.0)
        assert position[4] == pytest.approx(0.0)

    def test_travel_limits_are_enforced(self, realistic_twin_proxy: tango.DeviceProxy):
        with pytest.raises(tango.DevFailed):
            realistic_twin_proxy.move_stage([2e-3, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(tango.DevFailed):
            realistic_twin_proxy.move_stage([0.0, 0.0, 5e-4, 0.0, 0.0])

    def test_stage_drifts_between_reads(self, realistic_twin_proxy: tango.DeviceProxy):
        realistic_twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])

        first = np.asarray(realistic_twin_proxy.get_stage(), dtype=np.float64)
        time.sleep(0.3)
        second = np.asarray(realistic_twin_proxy.get_stage(), dtype=np.float64)

        assert not np.array_equal(first[:3], second[:3])

    def test_closed_valves_give_dark_frame(
        self,
        realistic_twin_proxy: tango.DeviceProxy,
        scan_proxy: tango.DeviceProxy,
    ):
        scan_proxy.imsize = 32
        scan_proxy.dwell_time = 1e-6

        realistic_twin_proxy.unblank_beam()
        realistic_twin_proxy.set_column_valves("close")
        with h5py.File(realistic_twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            dark = h5["image/HAADF"][()]

        realistic_twin_proxy.set_column_valves("open")
        with h5py.File(realistic_twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            lit = h5["image/HAADF"][()]

        assert float(dark.max()) < 0.1, "closed valves must yield detector noise only"
        assert float(lit.max()) > 0.5, "open valves must yield a real (normalized) image"

    def test_blanked_beam_gives_noise_spectrum(
        self,
        realistic_twin_proxy: tango.DeviceProxy,
        eds_proxy: tango.DeviceProxy,
    ):
        eds_proxy.exposure_time = 0.05
        realistic_twin_proxy.set_column_valves("open")
        realistic_twin_proxy.blank_beam()
        try:
            with h5py.File(realistic_twin_proxy.acquire_spectrum("eds"), "r") as h5:
                spectrum = h5["spectrum"][()]
        finally:
            realistic_twin_proxy.unblank_beam()

        assert float(np.max(spectrum)) < 0.1, "blanked beam must yield the noise floor"

    def test_ideal_twin_is_unaffected(self, twin_proxy: tango.DeviceProxy):
        """The default twin keeps ideal behavior so deterministic tests stay valid."""
        twin_proxy.move_stage([5e-6, 0.0, 0.0, 0.0, 0.0])

        first = np.asarray(twin_proxy.get_stage(), dtype=np.float64)
        time.sleep(0.05)
        second = np.asarray(twin_proxy.get_stage(), dtype=np.float64)

        # Lands exactly on target (up to the float32 Tango wire) and does not drift.
        assert first[0] == pytest.approx(5e-6, rel=1e-6)
        assert first[1:].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert np.array_equal(first, second)
        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
