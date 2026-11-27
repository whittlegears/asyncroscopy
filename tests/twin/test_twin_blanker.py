"""Every twin blanks with the one blanker model (TwinBlanker): blanked, its detectors record no beam signal, the
saved data say so, and the MCP server fails such an acquisition as it does on DigitalTwin."""

import json

import numpy as np
import pytest
import tango

from asyncroscopy.data import views
from asyncroscopy.instruments.electron_microscope import (digital_twin_diffraction, digital_twin_particles, digital_twin_tilt,
                                                          polycrystalline_gold_digital_twin)
from asyncroscopy.instruments.electron_microscope.twin import atomistic

TWINS = {
    "asyncroscopy/digitaltwinparticles/default": (digital_twin_particles, lambda d: d._acquire_scanned_image(32, 1e-6, ["haadf"])),
    "asyncroscopy/digitaltwintilt/default": (digital_twin_tilt, lambda d: d._acquire_scanned_image(32, 1e-6, ["haadf"])),
    "asyncroscopy/atomistic/diffraction": (digital_twin_diffraction, lambda d: d._acquire_scanned_image(32, 1e-6, ["haadf"])),
    "asyncroscopy/atomistic/polycrystal": (polycrystalline_gold_digital_twin, lambda d: d._acquire_scanned_image(32, 1e-5, ["haadf"])),
}


def _capture(monkeypatch, *modules):
    saved = []

    def capture(device, data_server, kind, detectors, arrays, dataset_attrs=None, file_attrs=None, **kwargs):
        saved.append((np.asarray(arrays), dataset_attrs if not isinstance(dataset_attrs, list) else dataset_attrs[0]))
        return "key"

    for module in modules:
        monkeypatch.setattr(module, "save_acquisition", capture)
    return saved


@pytest.mark.parametrize("name", sorted(TWINS))
def test_an_atomistic_twin_blanks_and_a_blanked_image_has_no_beam_signal(tango_ctx, monkeypatch, name):
    module, acquire = TWINS[name]
    saved = _capture(monkeypatch, module)
    proxy = tango.DeviceProxy(tango_ctx.get_device_access(name))
    device = tango.Util.instance().get_device_by_name(name)
    proxy.unblank_beam()
    acquire(device)
    lit, lit_attrs = saved[-1]
    assert lit_attrs["beam_blanked"] is False and np.any(lit > 0)
    proxy.blank_beam()
    try:
        assert json.loads(proxy.get_parameters())["beam_blanked"] is True
        acquire(device)
        dark, dark_attrs = saved[-1]
        assert dark_attrs["beam_blanked"] is True
        assert dark.shape == lit.shape and not np.array_equal(dark, lit)
    finally:
        proxy.unblank_beam()


def test_a_blanked_spectrum_has_no_counts(tango_ctx, monkeypatch):
    saved = _capture(monkeypatch, atomistic)
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/atomistic/diffraction"))
    device = tango.Util.instance().get_device_by_name("asyncroscopy/atomistic/diffraction")
    proxy.blank_beam()
    try:
        device._acquire_spectrum("eds", 1.0)
        counts, attrs = saved[-1]
        assert attrs["beam_blanked"] is True and not np.any(counts)
    finally:
        proxy.unblank_beam()


def test_stop_all_blanks_an_atomistic_twin(tango_ctx):
    from asyncroscopy.mcp.mcp_server import MCPServer

    class Exported:
        value_string = [tango_ctx.get_device_access("asyncroscopy/digitaltwintilt/default")]

        def get_device_exported(self, pattern):
            return self

    server = MCPServer("t", "localhost", 0, blocked_functions={}, blocked_classes=[], data_device_address="", database=Exported(), verbose=False)
    server.setup(print_summary=False)
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwintilt/default"))
    try:
        report = server.estop.stop_all()
        assert report["stopped"] is True, report
        assert json.loads(proxy.get_parameters())["beam_blanked"] is True
    finally:
        proxy.unblank_beam()


def test_the_server_fails_an_acquisition_saved_with_the_beam_blanked():
    from tests.analysis.test_views import Container, Recorded

    for blanked, expected in ((True, True), (False, None)):
        node = Container({"image": Container({"HAADF": Recorded((8, 8), metadata={"beam_blanked": blanked}, values=np.zeros((8, 8)))})})
        summary, _png = views.view(node, "k.h5", "text")
        assert summary.get("beam_blanked") is expected
