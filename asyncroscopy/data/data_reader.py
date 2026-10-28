"""Shared logic for reading Tiled dataset metadata and small previews.

Both the MCP bridge (``get_data_from_key``) and the electron microscope's
legacy byte-over-Tango command (``get_image_data_cached``) need the same
thing: given an already-resolved Tiled node, describe its shape/dtype/attrs
and a small flattened preview. This module is the one place that logic
lives; callers are responsible for resolving the Tiled client/node
themselves, since that involves a DeviceProxy in one case and an in-process
DeviceProxy in the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return numpy_to_python(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = [numpy_to_python(v) for v in obj]
        return tuple(conv) if isinstance(obj, tuple) else conv
    return obj


def describe_tiled_node(key: str, uri: str, node: Any, max_values: int = 64) -> dict[str, Any]:
    """Build shape/dtype/attrs metadata plus a small flattened preview for one Tiled node."""
    limit = max(0, int(max_values))
    suffix = key.rsplit(".", 1)[-1].lower() if "." in key else "unknown"
    result: dict[str, Any] = {
        "key": key,
        "uri": uri,
        "format": "hdf5" if suffix in {"h5", "hdf5"} else suffix,
        "attrs": numpy_to_python(dict(getattr(node, "metadata", {}) or {})),
    }
    datasets: list[dict[str, Any]] = []

    def visit(current: Any, name: str = "") -> None:
        read = getattr(current, "read", None)
        if callable(read):
            shape = tuple(getattr(current, "shape", ()) or ())
            if limit == 0:
                array = np.asarray([], dtype=getattr(current, "dtype", float))
            elif shape:
                remaining = limit
                slices = []
                for size in reversed(shape):
                    take = min(int(size), max(1, remaining))
                    slices.append(slice(0, take))
                    remaining = (remaining + take - 1) // take
                array = np.asarray(read(tuple(reversed(slices))))
            else:
                array = np.asarray(read())
            item: dict[str, Any] = {
                "name": name,
                "shape": list(shape or array.shape),
                "dtype": str(getattr(current, "dtype", array.dtype)),
                "attrs": numpy_to_python(dict(getattr(current, "metadata", {}) or {})),
                "preview": numpy_to_python(array.reshape(-1)[:limit]),
            }
            datasets.append(item)
            return

        keys = getattr(current, "keys", None)
        if callable(keys):
            for child_name in keys():
                child_path = f"{name}/{child_name}" if name else str(child_name)
                visit(current[child_name], child_path)

    visit(node)
    result["datasets"] = datasets
    return result


def read_acquisition(key: str, data_proxy: Any = None) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Every dataset of one acquisition as ``{name: (array, attrs)}``.

    ``key`` is what an acquire command returned: a local HDF5 path when no DATA
    device is configured, otherwise a Tiled key resolved through the DATA
    device's Tiled server.
    """
    from pathlib import Path

    path = Path(key)
    out: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    if path.suffix.lower() in {".h5", ".hdf5"} and path.exists():
        import h5py

        with h5py.File(path, "r") as h5:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    out[name] = (obj[()], dict(obj.attrs))
            h5.visititems(visit)
        return out
    if data_proxy is None:
        raise FileNotFoundError(f"{key!r} is not a local file and no DATA device was given to resolve it")
    import json

    from tiled.client import from_uri

    node = from_uri(json.loads(data_proxy.get_config())["uri"])[key]

    def walk(current: Any, name: str) -> None:
        if callable(getattr(current, "read", None)) and not hasattr(current, "keys"):
            out[name] = (np.asarray(current.read()), numpy_to_python(dict(getattr(current, "metadata", {}) or {})))
            return
        for child in current.keys():
            walk(current[child], f"{name}/{child}" if name else child)

    walk(node, "")
    return out
