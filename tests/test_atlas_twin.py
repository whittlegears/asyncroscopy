"""Return-to-site on the DigitalTwin Tango device, including a simulated remount."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from asyncroscopy.atlas import revisit
from asyncroscopy.atlas.revisit import RevisitParams
from asyncroscopy.atlas.store import Atlas
from asyncroscopy.atlas.tango_backend import TangoMicroscope

FOV, N = 6e-9, 64


def read_h5(path: str) -> np.ndarray:
    with h5py.File(Path(path), "r") as h5:
        return h5["image/HAADF"][()]


@pytest.fixture()
def twin(twin_proxy, scan_proxy):
    scan_proxy.dwell_time = 2e-6
    twin_proxy.write_attribute("specimen_mount", [0.0, 0.0, 0.0])
    twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    yield TangoMicroscope(twin_proxy, scan_proxy, read_h5)
    twin_proxy.write_attribute("specimen_mount", [0.0, 0.0, 0.0])
    twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])


def test_twin_reports_fov(twin):
    twin.instrument.set_fov(FOV)
    assert twin.current_fov() == pytest.approx(FOV)


def test_specimen_mount_moves_the_specimen(twin):
    a = twin.acquire(FOV, N)
    twin.instrument.write_attribute("specimen_mount", [2e-9, 0.0, 0.0])
    b = twin.acquire(FOV, N)
    twin.move_stage([2e-9, 0.0, 0.0, 0.0, 0.0])          # follow the specimen
    c = twin.acquire(FOV, N)
    assert not np.allclose(a, b)
    assert np.corrcoef(a.ravel(), c.ravel())[0, 1] > 0.9


def test_return_to_site_on_twin_after_remount(tmp_path, twin):
    atlas = Atlas(tmp_path)
    mapping = revisit.calibrate_mapping(atlas, twin, FOV, N)
    assert mapping.calibrated
    twin.move_stage([0.6e-9, -0.4e-9, 0.0, 0.0, 0.0])
    site = revisit.save_site(atlas, twin, "particle", FOV, N)

    twin.instrument.write_attribute("specimen_mount", [0.7e-9, 0.5e-9, 0.0])   # unload / remount
    atlas.new_mount("remount-1")
    params = RevisitParams(survey_factor=2.0, max_rotation_deg=0.0)
    visit = revisit.return_to_site(atlas, site.id, twin, params=params)

    truth = np.array(site.stage[:2]) + [0.7e-9, 0.5e-9]
    error_m = np.linalg.norm(np.array(twin.get_stage()[:2]) - truth)
    assert visit["status"] == "match", visit["reasons"]
    assert error_m < 3 * FOV / N
