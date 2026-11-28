"""Write a grid scan as a map run that the pyTEMlib analysis server can stitch.

pyTEMlib's ``assemble_map`` reads a run: one HDF5 file with a ``tiles`` group of
``tile_NNNNN`` datasets (layout 1), each carrying ``stage_reported_m`` (the
tile centre) and ``pixel_size_m``. It assumes image rows run along +y and
columns along +x. It refines from there, but it predicts the first match
between neighbours from those positions, so a stage whose axes are swapped,
mirrored or rotated in the image must be converted first.

The grid scan (``acquire_map_grid``) measures that relation as a 2 x 2 matrix
(``calibration.px_per_m``: content shift in pixels per metre of stage). This
module keeps its direction, R (the nearest rotation or mirror, scaled by
1 / pixel size: the length of a test move is less certain than the stage
reading), and puts every tile centre in the image frame:

    view shift (rows, cols) = -R @ (stage - stage of the first tile)
    centre (x, y)           = first tile's stage (x, y) + (cols, rows) * pixel size

The raw stage readings stay in each tile's attributes (``stage_actual_m``,
``commanded_xy_m``), so nothing measured is lost.

Production runs go through the analysis server's ``collect_map_run`` with the
converted positions (``image_frame_positions``); ``write_map_run`` writes the
same run straight to a local file, for a server started with ``--local-root``
on that folder and for tests.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Sequence

import h5py
import numpy as np

RUN_KIND = "map_run"


def direction_only(px_per_m, pixel_size_m: float) -> np.ndarray:
    """The rotation (and mirror) in a measured matrix, with the scale set by the pixel size.

    The test moves measure direction well but length only to the calibration's
    tolerance (15%), while the stage reads metres and the pixel size is known.
    So keep the nearest orthogonal matrix (polar decomposition) and scale it by
    1 / pixel size. A real stretch or shear between stage and scan is left for
    pyTEMlib's own scan_linear fit, which measures it from the matched tiles.
    """
    matrix = np.asarray(px_per_m, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("px_per_m must be 2 x 2")
    u, _s, vt = np.linalg.svd(matrix)
    return (u @ vt) / float(pixel_size_m)


def image_frame_positions(stage_xy_m: Sequence[Sequence[float]], px_per_m, pixel_size_m: float) -> list[list[float]]:
    """Tile centres in the frame pyTEMlib expects (columns along +x, rows along +y), in metres.

    Uses only the direction of ``px_per_m`` (see ``direction_only``).
    """
    stage = np.asarray(stage_xy_m, dtype=float)[:, :2]
    matrix = direction_only(px_per_m, pixel_size_m)
    view = -(matrix @ (stage - stage[0]).T).T  # (rows, cols) in pixels, per tile
    x = stage[0, 0] + view[:, 1] * pixel_size_m
    y = stage[0, 1] + view[:, 0] * pixel_size_m
    return np.column_stack([x, y]).tolist()


def _first_2d(path: str) -> np.ndarray:
    """The first 2D image in an acquisition file, preferring image/HAADF."""
    with h5py.File(path, "r") as h5:
        if "image/HAADF" in h5:
            return np.asarray(h5["image/HAADF"][()])
        found = []
        h5.visititems(lambda name, obj: found.append(name) if isinstance(obj, h5py.Dataset) and obj.ndim == 2 else None)
        if not found:
            raise ValueError(f"No 2D image in {path}")
        return np.asarray(h5[found[0]][()])


def write_map_run(
    path: str | Path,
    manifest: dict,
    read_image: Optional[Callable[[str], np.ndarray]] = None,
    plan: Optional[dict] = None,
) -> Path:
    """Write the run for an ``acquire_map_grid`` manifest and return its path.

    ``read_image(key)`` loads a tile; by default keys are local HDF5 files.
    The file is written beside the target and renamed into place, so a
    half-written run is never left where the stitcher would read it.
    """
    read_image = read_image or _first_2d
    tiles = manifest["tiles"]
    if not tiles:
        raise ValueError("The manifest has no tiles")
    pixel_size_m = float(manifest["pixel_size_nm"]) * 1e-9
    matrix = (manifest.get("calibration") or {}).get("px_per_m")
    stage = [t["stage_xy_m"] for t in tiles]
    if matrix is None:
        centres = [list(map(float, xy[:2])) for xy in stage]  # no calibration: trust the stage axes
    else:
        centres = image_frame_positions(stage, matrix, pixel_size_m)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    meta = {
        "kind": RUN_KIND,
        "run_layout": 1,
        "id": uuid.uuid4().hex,
        "plan": json.dumps(plan or {"grid": manifest.get("grid"), "overlap": manifest.get("overlap"), "fov_m": manifest.get("fov_m")}),
        "source": "asyncroscopy.acquire_map_grid",
        "tiles": len(tiles),
        "created": time.time(),
        "calibration": json.dumps(manifest.get("calibration") or {}),
        "axes": "stage_reported_m: tile centres with image rows along +y and columns along +x (converted with calibration.px_per_m); "
        "stage_actual_m: the stage as read back",
    }
    try:
        with h5py.File(partial, "w") as h5:
            for key, value in meta.items():
                h5.attrs[key] = value
            group = h5.create_group("tiles")
            for index, (tile, centre) in enumerate(zip(tiles, centres)):
                data = group.create_dataset(f"tile_{index:05d}", data=read_image(tile["key"]))
                data.attrs["index"] = index
                data.attrs["stage_reported_m"] = [float(v) for v in centre]
                data.attrs["pixel_size_m"] = pixel_size_m
                data.attrs["source_key"] = str(tile["key"])
                data.attrs["row_col"] = [int(tile["row"]), int(tile["col"])]
                data.attrs["stage_actual_m"] = [float(v) for v in tile["stage_xy_m"][:2]]
                if tile.get("commanded_xy_m") is not None:
                    data.attrs["commanded_xy_m"] = [float(v) for v in tile["commanded_xy_m"][:2]]
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path
