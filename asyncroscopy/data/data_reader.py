"""Reading acquisitions back as arrays, and making their values JSON-safe.

To *look* at stored data (PNG or text, sliced on the server) use
asyncroscopy.data.views; this module is for code that needs the values.
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

    from asyncroscopy.data import tiled_client

    node = tiled_client.node_for_key(key, json.loads(data_proxy.get_config())["uri"])

    def walk(current: Any, name: str) -> None:
        if callable(getattr(current, "read", None)) and not hasattr(current, "keys"):
            out[name] = (np.asarray(current.read()), numpy_to_python(dict(getattr(current, "metadata", {}) or {})))
            return
        for child in current.keys():
            walk(current[child], f"{name}/{child}" if name else child)

    walk(node, "")
    return out
