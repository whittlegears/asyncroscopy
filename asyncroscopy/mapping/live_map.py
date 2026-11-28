"""A sample map that grows tile by tile, with findings that stay on their evidence.

The map is a set of registered tiles, not one stitched image. Each tile keeps
its own pixels; the stitcher only decides where each tile sits on the canvas.
That lets work start before the scan is done:

- ``add_tile`` stores a tile, runs every attached detector on it straight away,
  and re-registers the grid. Findings appear while the map is still forming.
- A finding (an *annotation*) is anchored to the pixels of the tile it was found
  in, never to canvas coordinates. When a later tile moves the stitch, the
  finding moves with its tile, so it stays on the same feature.
- Two findings of the same label from overlapping tiles that land within
  ``merge_radius_px`` of each other on the canvas are one feature: the later one
  is marked ``duplicate`` of the earlier one.
- A finding is ``placed`` only when its tile is tied to the map by at least one
  trusted seam (or is the only tile). Until then its canvas position is a guess.
- Every change to a finding (proposed, accepted, rejected, corrected) is kept in
  its ``history`` with who made it: ``human:<name>``, ``agent:<name>`` or
  ``model:<name>@<version>``. Findings from models and agents start as
  ``provisional``; only a review makes them ``accepted`` or ``rejected``.
- ``crop`` returns raw pixels from the source tile, never blended canvas
  pixels, so a reviewer looks at what the detector looked at.

Coordinates: tile pixels are (row, col). Canvas positions are (x, y) =
(col, row) of the tile's top-left corner, as GridStitcher returns them.
Specimen coordinates are canvas pixels times the pixel size, in nanometres.

``save`` writes a JSON manifest (format ``asyncroscopy-livemap/1``) next to the
tile arrays; ``LiveMap.load`` reads it back. Nothing here imports Tango.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grid_stitcher import GridStitcher, GridStitchResult

FORMAT = "asyncroscopy-livemap/1"
GridKey = Tuple[int, int]

STATUSES = ("provisional", "accepted", "rejected", "duplicate")


@dataclass
class Tile:
    row: int
    col: int
    image: np.ndarray
    stage_xy_m: Optional[Tuple[float, float]] = None
    source: Optional[str] = None
    acquired_at: float = field(default_factory=time.time)

    @property
    def key(self) -> GridKey:
        return (self.row, self.col)


@dataclass
class Annotation:
    """One finding, anchored to a tile's own pixels."""

    id: str
    tile: GridKey
    points_px: List[Tuple[float, float]]  # (row, col) in the tile; one point, or a polygon
    label: str
    author: str
    score: Optional[float] = None
    reason: str = ""
    status: str = "provisional"
    duplicate_of: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    created_in_revision: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return "point" if len(self.points_px) == 1 else "polygon"

    def centroid_px(self) -> Tuple[float, float]:
        pts = np.asarray(self.points_px, dtype=float)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())


Detector = Callable[[np.ndarray], List[Dict[str, Any]]]


class LiveMap:
    """Tiles, their registered positions, and the findings anchored to them."""

    def __init__(
        self,
        name: str = "sample_map",
        overlap: float = 0.15,
        pixel_size_nm: Optional[float] = None,
        merge_radius_px: float = 6.0,
        use_stage_positions: bool = True,
        stitcher: Optional[GridStitcher] = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.overlap = float(overlap)
        self.pixel_size_nm = pixel_size_nm
        self.merge_radius_px = float(merge_radius_px)
        self.use_stage_positions = use_stage_positions
        self.stitcher = stitcher or GridStitcher(overlap=self.overlap)
        self.tiles: Dict[GridKey, Tile] = {}
        self.annotations: Dict[str, Annotation] = {}
        self.detectors: Dict[str, Tuple[Detector, str]] = {}
        self.listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self.revision = 0  # bumps every time tile positions are re-solved
        self.positions: Dict[GridKey, Tuple[float, float]] = {}
        self.result: Optional[GridStitchResult] = None
        self.events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ events

    def _emit(self, event: str, **payload: Any) -> None:
        record = {"event": event, "revision": self.revision, "t": time.time(), **payload}
        self.events.append(record)
        for listener in list(self.listeners):
            listener(event, record)

    # ------------------------------------------------------------- tiles

    def add_detector(self, name: str, detector: Detector, version: str = "0") -> None:
        """Run ``detector(image) -> [{"px": (row, col) or [(row, col), ...], "score", "label", "reason"}]`` on every tile."""
        self.detectors[name] = (detector, str(version))

    def add_tile(
        self,
        row: int,
        col: int,
        image: np.ndarray,
        stage_xy_m: Optional[Sequence[float]] = None,
        source: Optional[str] = None,
        detect: bool = True,
        restitch: bool = True,
    ) -> Dict[str, Any]:
        """Store a tile, run the detectors on it, and re-register the map.

        Returns what changed: the new finding ids and the stitch summary.
        """
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f"A tile must be a 2D image, got shape {image.shape}")
        tile = Tile(int(row), int(col), image, tuple(float(v) for v in stage_xy_m[:2]) if stage_xy_m is not None else None, source)
        replaced = tile.key in self.tiles
        self.tiles[tile.key] = tile
        self._emit("tile_ready", tile=list(tile.key), source=source, replaced=replaced)
        new_ids: List[str] = []
        if detect:
            for name in self.detectors:
                new_ids += self.run_detector(name, tiles=[tile.key])
        summary = self.restitch() if restitch else {}
        return {"tile": list(tile.key), "new_annotations": new_ids, **summary}

    def restitch(self) -> Dict[str, Any]:
        """Re-solve every tile position, then re-check duplicates and placement."""
        if not self.tiles:
            return {}
        before = dict(self.positions)
        images = {k: t.image for k, t in self.tiles.items()}
        stage = None
        if self.use_stage_positions and all(t.stage_xy_m is not None for t in self.tiles.values()):
            stage = {k: t.stage_xy_m for k, t in self.tiles.items()}
        if len(images) == 1:
            key = next(iter(images))
            self.positions = {key: (0.0, 0.0)}
            self.result = None
        else:
            self.result = self.stitcher.stitch(images, stage_positions=stage)
            self.positions = dict(self.result.positions)
        self.revision += 1
        moved = {
            str(list(k)): float(np.hypot(*np.subtract(self.positions[k], before[k])))
            for k in before
            if k in self.positions
        }
        self._mark_duplicates()
        summary = {
            "revision": self.revision,
            "tiles_total": len(self.tiles),
            "edges_total": len(self.result.edges) if self.result else 0,
            "edges_trusted": sum(e.used for e in self.result.edges) if self.result else 0,
            "flagged_seams": [
                {"a": list(e.a), "b": list(e.b), "reason": e.reason} for e in (self.result.flagged_edges if self.result else [])
            ],
            "max_tile_move_px": max(moved.values(), default=0.0),
        }
        self._emit("stitch_updated", **summary)
        return summary

    def placed_tiles(self) -> set:
        """Tiles tied to the map by at least one trusted seam (or the only tile)."""
        if len(self.tiles) == 1:
            return set(self.tiles)
        if self.result is None:
            return set()
        placed = set()
        for e in self.result.edges:
            if e.used:
                placed.update((e.a, e.b))
        return placed

    # ------------------------------------------------------- coordinates

    def tile_to_canvas(self, tile: GridKey, row: float, col: float) -> Tuple[float, float]:
        """Canvas (x, y) of a pixel in a tile, at the current revision."""
        x0, y0 = self.positions.get(tuple(tile), (np.nan, np.nan))
        return float(x0 + col), float(y0 + row)

    def canvas_to_tile(self, x: float, y: float) -> Optional[Tuple[GridKey, float, float]]:
        """The tile whose centre is closest to canvas (x, y), among tiles that contain it."""
        best = None
        for key, (x0, y0) in self.positions.items():
            h, w = self.tiles[key].image.shape
            r, c = y - y0, x - x0
            if 0 <= r < h and 0 <= c < w:
                distance = np.hypot(r - h / 2, c - w / 2)
                if best is None or distance < best[0]:
                    best = (distance, key, r, c)
        return None if best is None else (best[1], float(best[2]), float(best[3]))

    def _canvas_point(self, annotation: Annotation) -> Tuple[float, float]:
        r, c = annotation.centroid_px()
        return self.tile_to_canvas(annotation.tile, r, c)

    # ------------------------------------------------------- annotations

    def annotate(
        self,
        tile: Sequence[int],
        points_px: Sequence[Sequence[float]] | Sequence[float],
        label: str,
        author: str,
        score: Optional[float] = None,
        reason: str = "",
        status: Optional[str] = None,
    ) -> str:
        """Add a finding. People's findings start accepted; models' and agents' start provisional."""
        key = (int(tile[0]), int(tile[1]))
        if key not in self.tiles:
            raise ValueError(f"No tile {list(key)} on this map")
        pts = np.asarray(points_px, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points_px must be (row, col) or a list of (row, col)")
        h, w = self.tiles[key].image.shape
        if np.any(pts < 0) or np.any(pts[:, 0] >= h) or np.any(pts[:, 1] >= w):
            raise ValueError(f"points_px must lie inside the {h} x {w} tile")
        if status is None:
            status = "accepted" if author.startswith("human:") else "provisional"
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        annotation = Annotation(
            id=uuid.uuid4().hex[:10],
            tile=key,
            points_px=[(float(r), float(c)) for r, c in pts],
            label=str(label),
            author=str(author),
            score=None if score is None else float(score),
            reason=str(reason),
            status=status,
            created_in_revision=self.revision,
        )
        annotation.history.append({"t": annotation.created_at, "actor": author, "action": "proposed", "status": status, "note": reason})
        self.annotations[annotation.id] = annotation
        self._mark_duplicates()
        self._emit("annotation_added", id=annotation.id, label=label, author=author)
        return annotation.id

    def review(self, annotation_id: str, actor: str, status: str, note: str = "", points_px=None, label: Optional[str] = None) -> Dict[str, Any]:
        """Accept, reject or correct a finding. The previous state stays in its history."""
        annotation = self._get(annotation_id)
        if status not in ("accepted", "rejected", "provisional"):
            raise ValueError("status must be accepted, rejected or provisional")
        change: Dict[str, Any] = {"t": time.time(), "actor": actor, "action": "reviewed", "from": annotation.status, "status": status, "note": note}
        if points_px is not None:
            pts = np.asarray(points_px, dtype=float).reshape(-1, 2)
            change["old_points_px"] = annotation.points_px
            annotation.points_px = [(float(r), float(c)) for r, c in pts]
        if label is not None and label != annotation.label:
            change["old_label"] = annotation.label
            annotation.label = label
        annotation.status = status
        annotation.history.append(change)
        self._mark_duplicates()
        self._emit("annotation_reviewed", id=annotation_id, status=status, actor=actor)
        return self.describe(annotation_id)

    def _get(self, annotation_id: str) -> Annotation:
        if annotation_id not in self.annotations:
            raise ValueError(f"No annotation {annotation_id!r}")
        return self.annotations[annotation_id]

    def _mark_duplicates(self) -> None:
        """Same label, different tiles, within merge_radius_px on the canvas: keep the earliest."""
        live = [a for a in self.annotations.values() if a.status != "rejected"]
        for a in live:
            if a.status == "duplicate":
                a.status, a.duplicate_of = "provisional", None
        live.sort(key=lambda a: (a.status != "accepted", a.created_at))
        kept: List[Annotation] = []
        for a in live:
            if not self.positions or a.tile not in self.positions:
                kept.append(a)
                continue
            pa = self._canvas_point(a)
            twin = next(
                (
                    k
                    for k in kept
                    if k.label == a.label
                    and k.tile != a.tile
                    and k.tile in self.positions
                    and np.hypot(*np.subtract(pa, self._canvas_point(k))) <= self.merge_radius_px
                ),
                None,
            )
            if twin is not None and a.status != "accepted":
                a.status, a.duplicate_of = "duplicate", twin.id
            else:
                kept.append(a)

    def describe(self, annotation_id: str) -> Dict[str, Any]:
        """A finding with its current canvas and specimen position and its evidence trail."""
        a = self._get(annotation_id)
        placed = a.tile in self.placed_tiles()
        x, y = self._canvas_point(a) if a.tile in self.positions else (None, None)
        out = {
            "id": a.id,
            "label": a.label,
            "kind": a.kind,
            "status": a.status,
            "duplicate_of": a.duplicate_of,
            "author": a.author,
            "score": a.score,
            "reason": a.reason,
            "tile": list(a.tile),
            "tile_source": self.tiles[a.tile].source,
            "points_px": [list(p) for p in a.points_px],
            "canvas_xy": None if x is None else [x, y],
            "specimen_nm": None if x is None or not self.pixel_size_nm else [x * self.pixel_size_nm, y * self.pixel_size_nm],
            "placed": placed,
            "revision": self.revision,
            "created_in_revision": a.created_in_revision,
            "history": a.history,
        }
        return out

    def list_annotations(self, status: Optional[str] = None, label: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            self.describe(a.id)
            for a in sorted(self.annotations.values(), key=lambda a: a.created_at)
            if (status is None or a.status == status) and (label is None or a.label == label)
        ]

    # ---------------------------------------------------------- detectors

    def run_detector(self, name: str, tiles: Optional[Sequence[Sequence[int]]] = None) -> List[str]:
        """Run one attached detector on some tiles (default: all). Returns the new finding ids."""
        if name not in self.detectors:
            raise ValueError(f"No detector {name!r}; attached: {sorted(self.detectors)}")
        detector, version = self.detectors[name]
        keys = [tuple(k) for k in tiles] if tiles is not None else sorted(self.tiles)
        new_ids = []
        for key in keys:
            image = self.tiles[key].image
            for hit in detector(image):
                pts = np.asarray(hit["px"], dtype=float).reshape(-1, 2)
                h, w = image.shape
                pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
                new_ids.append(
                    self.annotate(
                        key,
                        pts,
                        label=hit.get("label", name),
                        author=f"model:{name}@{version}",
                        score=hit.get("score"),
                        reason=hit.get("reason", ""),
                    )
                )
        return new_ids

    # ------------------------------------------------------------- crops

    def crop(self, annotation_id: Optional[str] = None, tile=None, center_px=None, half_size: int = 32) -> Dict[str, Any]:
        """Raw pixels around a finding, or around (row, col) in a tile, from the source tile only."""
        if annotation_id is not None:
            a = self._get(annotation_id)
            tile, center_px = a.tile, a.centroid_px()
        key = (int(tile[0]), int(tile[1]))
        image = self.tiles[key].image
        r, c = int(round(center_px[0])), int(round(center_px[1]))
        r0, r1 = max(0, r - half_size), min(image.shape[0], r + half_size)
        c0, c1 = max(0, c - half_size), min(image.shape[1], c + half_size)
        return {
            "tile": list(key),
            "source": self.tiles[key].source,
            "origin_px": [r0, c0],
            "pixels": image[r0:r1, c0:c1],
            "pixel_size_nm": self.pixel_size_nm,
        }

    # ------------------------------------------------------------ summary

    def status(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for a in self.annotations.values():
            counts[a.status] = counts.get(a.status, 0) + 1
        rows = [k[0] for k in self.tiles]
        cols = [k[1] for k in self.tiles]
        return {
            "map_id": self.id,
            "name": self.name,
            "revision": self.revision,
            "tiles_total": len(self.tiles),
            "tiles_placed": len(self.placed_tiles()),
            "grid_extent": [max(rows, default=-1) + 1, max(cols, default=-1) + 1],
            "annotations": counts,
            "detectors": {name: version for name, (_d, version) in self.detectors.items()},
            "pixel_size_nm": self.pixel_size_nm,
        }

    def list_tiles(self) -> List[Dict[str, Any]]:
        placed = self.placed_tiles()
        out = []
        for key, t in sorted(self.tiles.items()):
            pos = self.positions.get(key)
            out.append(
                {
                    "tile": list(key),
                    "source": t.source,
                    "size_px": list(t.image.shape),
                    "stage_xy_m": list(t.stage_xy_m) if t.stage_xy_m else None,
                    "canvas_xy": list(pos) if pos else None,
                    "placed": key in placed,
                    "findings": sum(1 for a in self.annotations.values() if a.tile == key and a.status != "rejected"),
                }
            )
        return out

    # -------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> Path:
        """Write map.json and one .npy per tile. Positions are re-solved on load."""
        directory = Path(directory)
        (directory / "tiles").mkdir(parents=True, exist_ok=True)
        tiles = []
        for key, t in sorted(self.tiles.items()):
            name = f"tiles/r{key[0]:03d}_c{key[1]:03d}.npy"
            np.save(directory / name, t.image)
            tiles.append({"row": key[0], "col": key[1], "file": name, "stage_xy_m": t.stage_xy_m, "source": t.source, "acquired_at": t.acquired_at})
        manifest = {
            "format": FORMAT,
            "id": self.id,
            "name": self.name,
            "overlap": self.overlap,
            "pixel_size_nm": self.pixel_size_nm,
            "merge_radius_px": self.merge_radius_px,
            "revision": self.revision,
            "positions": {f"{k[0]},{k[1]}": list(v) for k, v in self.positions.items()},
            "detectors": {name: version for name, (_d, version) in self.detectors.items()},
            "tiles": tiles,
            "annotations": [_annotation_to_json(a) for a in self.annotations.values()],
            "events": self.events[-500:],
        }
        path = directory / "map.json"
        path.write_text(json.dumps(manifest, indent=2, default=_json_default))
        return path

    @classmethod
    def load(cls, directory: str | Path, restitch: bool = True) -> "LiveMap":
        directory = Path(directory)
        manifest = json.loads((directory / "map.json").read_text())
        if manifest.get("format") != FORMAT:
            raise ValueError(f"Not a {FORMAT} map: {manifest.get('format')!r}")
        live = cls(manifest["name"], manifest["overlap"], manifest["pixel_size_nm"], manifest["merge_radius_px"])
        live.id = manifest["id"]
        for t in manifest["tiles"]:
            live.tiles[(t["row"], t["col"])] = Tile(
                t["row"], t["col"], np.load(directory / t["file"]), tuple(t["stage_xy_m"]) if t["stage_xy_m"] else None, t["source"], t["acquired_at"]
            )
        for raw in manifest["annotations"]:
            raw = dict(raw)
            raw["tile"] = tuple(raw["tile"])
            raw["points_px"] = [tuple(p) for p in raw["points_px"]]
            live.annotations[raw["id"]] = Annotation(**raw)
        live.positions = {tuple(int(v) for v in k.split(",")): tuple(p) for k, p in manifest["positions"].items()}
        live.revision = manifest["revision"]
        live.events = manifest.get("events", [])
        if restitch:
            live.restitch()
        return live


def _annotation_to_json(a: Annotation) -> Dict[str, Any]:
    d = asdict(a)
    d["tile"] = list(a.tile)
    d["points_px"] = [list(p) for p in a.points_px]
    return d


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")
