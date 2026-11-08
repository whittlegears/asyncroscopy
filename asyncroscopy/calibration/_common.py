"""Shared helpers for the calibration scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import tango
from skimage.registration import phase_cross_correlation


def parse_args(description: str, extra: Callable[[argparse.ArgumentParser], object] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", default="asyncroscopy/instrument/default", help="Microscope Tango device.")
    parser.add_argument("--scan", default="asyncroscopy/scan/default", help="SCAN settings device.")
    parser.add_argument("--fov", type=float, default=200e-9, help="Field of view (m).")
    parser.add_argument("--imsize", type=int, default=256, help="Image size (pixels per side).")
    parser.add_argument("--dwell", type=float, default=2e-6, help="Dwell time per pixel (s).")
    if extra:
        extra(parser)
    return parser.parse_args()


def connect(args: argparse.Namespace) -> tuple[tango.DeviceProxy, tango.DeviceProxy]:
    microscope = tango.DeviceProxy(args.device)
    microscope.set_timeout_millis(120_000)
    scan = tango.DeviceProxy(args.scan)
    scan.imsize = args.imsize
    scan.dwell_time = args.dwell
    scan.scan_region = [0.0, 0.0, 1.0, 1.0]
    microscope.set_fov(args.fov)
    return microscope, scan


def load_image(microscope: tango.DeviceProxy, key: str) -> np.ndarray:
    """Read an acquisition back from a local HDF5 path or, failing that, through the DATA device's Tiled server."""
    path = Path(key)
    if path.exists():
        with h5py.File(path, "r") as h5:
            node = h5["image"] if "image" in h5 else h5
            first = next(iter(node))
            return np.asarray(node[first][()], dtype=np.float64)
    from asyncroscopy.data import tiled_client

    address = json.loads(microscope.get_parameters()).get("data_device_address", "asyncroscopy/data/default")
    node = tiled_client.node_for_key(key, tiled_client.uri_from_data_device(address))
    while hasattr(node, "keys") and not hasattr(node, "read"):
        node = node[list(node.keys())[0]]
    return np.asarray(node.read(), dtype=np.float64)


def acquire(microscope: tango.DeviceProxy) -> np.ndarray:
    return load_image(microscope, microscope.acquire_scanned_image([]))


def shift_m(reference: np.ndarray, image: np.ndarray, pixel_m: float) -> np.ndarray:
    """(x, y) shift of image relative to reference, in metres."""
    shift, _error, _phase = phase_cross_correlation(reference, image, upsample_factor=10)
    return np.array([shift[1], shift[0]]) * pixel_m


def print_yaml(title: str, values: dict) -> None:
    print(f"# {title}")
    for key, value in values.items():
        print(f"    {key}: {value:.3e}" if isinstance(value, float) else f"    {key}: {value}")
