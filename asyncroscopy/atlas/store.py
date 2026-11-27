"""The specimen atlas: saved sites, their native images, and every return visit.

A teammate should be able to open the atlas later and see *where* something
happened and *how sure* the instrument was that it went back to the same
place. So each site keeps the native reference image (full precision ``.npy``),
the instrument settings it was taken with, the action/run it came from, and an
append-only list of visits, each with its own image and registration evidence.

Layout on disk::

    <root>/atlas.json                  specimen, mount transform, mapping, site index
    <root>/history.jsonl               append-only log of atlas events
    <root>/sites/<site_id>/reference.npy
    <root>/sites/<site_id>/site.json   settings and provenance
    <root>/sites/<site_id>/visits/<visit_id>.npy
    <root>/sites/<site_id>/visits/<visit_id>.json
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asyncroscopy.atlas.stage_mapping import MountTransform, StageMapping

ATLAS_VERSION = 1


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower()[:40] or "site"


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


@dataclass
class Site:
    id: str
    name: str
    stage: list[float]                 # [x, y, z, alpha, beta] on the ORIGINAL mount (m, deg)
    fov_m: float
    imsize: int
    pixel_size_m: float
    created: float = field(default_factory=time.time)
    landmark: bool = False
    notes: str = ""
    settings: dict = field(default_factory=dict)      # instrument parameters at acquisition
    provenance: dict = field(default_factory=dict)    # action id, run id, requested by
    mount_id: str = "original"
    visits: list[dict] = field(default_factory=list)  # summaries; details in visits/*.json

    def to_json(self) -> dict:
        return _json_safe(asdict(self))

    @classmethod
    def from_json(cls, data: dict) -> Site:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Atlas:
    """Thread-safe on-disk atlas. All writes go through this object."""

    def __init__(self, root: str | Path, specimen_id: str = "specimen"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sites").mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._index_path = self.root / "atlas.json"
        if self._index_path.exists():
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        else:
            data = {"version": ATLAS_VERSION, "specimen_id": specimen_id, "created": time.time(),
                    "mount": {"id": "original", "transform": MountTransform().to_json()},
                    "mapping": StageMapping().to_json(), "sites": []}
        self.specimen_id: str = data.get("specimen_id", specimen_id)
        self.created: float = data.get("created", time.time())
        self.mount_id: str = data.get("mount", {}).get("id", "original")
        self.mount = MountTransform.from_json(data.get("mount", {}).get("transform"))
        self.mapping = StageMapping.from_json(data.get("mapping"))
        self.sites: dict[str, Site] = {}
        for site_id in data.get("sites", []):
            path = self._site_dir(site_id) / "site.json"
            if path.exists():
                self.sites[site_id] = Site.from_json(json.loads(path.read_text(encoding="utf-8")))
        if not self._index_path.exists():
            self._write_index()

    # -- paths -----------------------------------------------------------------
    def _site_dir(self, site_id: str) -> Path:
        return self.root / "sites" / site_id

    def reference_path(self, site_id: str) -> Path:
        return self._site_dir(site_id) / "reference.npy"

    def visit_image_path(self, site_id: str, visit_id: str) -> Path:
        return self._site_dir(site_id) / "visits" / f"{visit_id}.npy"

    # -- persistence -----------------------------------------------------------
    def _write_index(self) -> None:
        data = {"version": ATLAS_VERSION, "specimen_id": self.specimen_id, "created": self.created,
                "mount": {"id": self.mount_id, "transform": self.mount.to_json()},
                "mapping": self.mapping.to_json(), "sites": list(self.sites)}
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

    def _write_site(self, site: Site) -> None:
        d = self._site_dir(site.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "site.json.tmp"
        tmp.write_text(json.dumps(site.to_json(), indent=2), encoding="utf-8")
        tmp.replace(d / "site.json")

    def log(self, kind: str, **data) -> None:
        row = {"ts": time.time(), "kind": kind, **_json_safe(data)}
        with self._lock, (self.root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def history(self) -> list[dict]:
        path = self.root / "history.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    # -- sites -----------------------------------------------------------------
    def add_site(self, name: str, image: np.ndarray, stage, fov_m: float, *, landmark: bool = False,
                 notes: str = "", settings: dict | None = None, provenance: dict | None = None) -> Site:
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f"a site reference must be a 2-D image, got shape {image.shape}")
        stage = [float(v) for v in stage]
        stage = (stage + [0.0] * 5)[:5]
        with self._lock:
            site_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
            site = Site(id=site_id, name=name, stage=stage, fov_m=float(fov_m), imsize=int(image.shape[1]),
                        pixel_size_m=float(fov_m) / image.shape[1], landmark=bool(landmark), notes=notes,
                        settings=dict(settings or {}), provenance=dict(provenance or {}),
                        mount_id=self.mount_id)
            if self.mount_id != "original" and not self.mount.identity:
                # Store sites in original-mount coordinates so one transform maps them all.
                site.stage[:2] = list(self._invert_mount(stage[:2]))
                site.settings.setdefault("stage_on_mount", {})[self.mount_id] = stage
            self._site_dir(site_id).mkdir(parents=True, exist_ok=True)
            np.save(self.reference_path(site_id), image)
            self.sites[site_id] = site
            self._write_site(site)
            self._write_index()
            self.log("site_saved", site=site_id, name=name, stage=stage, fov_m=fov_m,
                     landmark=landmark, provenance=site.provenance)
            return site

    def _invert_mount(self, xy) -> tuple[float, float]:
        import math
        t = math.radians(-self.mount.rotation_deg)
        x, y = xy[0] - self.mount.translation_m[0], xy[1] - self.mount.translation_m[1]
        return (math.cos(t) * x - math.sin(t) * y, math.sin(t) * x + math.cos(t) * y)

    def get(self, site_id: str) -> Site:
        try:
            return self.sites[site_id]
        except KeyError:
            raise KeyError(f"no site {site_id!r} in the atlas (have: {', '.join(self.sites) or 'none'})") from None

    def reference(self, site_id: str) -> np.ndarray:
        return np.load(self.reference_path(self.get(site_id).id))

    def update_site(self, site_id: str, **changes) -> Site:
        allowed = {"name", "notes", "landmark"}
        with self._lock:
            site = self.get(site_id)
            for key, value in changes.items():
                if key not in allowed:
                    raise ValueError(f"cannot change {key!r}; editable fields: {sorted(allowed)}")
                setattr(site, key, value)
            self._write_site(site)
            self.log("site_updated", site=site_id, changes=changes)
            return site

    def predicted_stage(self, site_id: str) -> list[float]:
        """Where the site should be on the *current* mount (a starting guess only)."""
        site = self.get(site_id)
        x, y = self.mount.apply(site.stage[:2])
        return [x, y, *site.stage[2:5]]

    def landmarks(self) -> list[Site]:
        return [s for s in self.sites.values() if s.landmark]

    # -- visits ----------------------------------------------------------------
    def record_visit(self, site_id: str, result: dict, image: np.ndarray | None) -> dict:
        with self._lock:
            site = self.get(site_id)
            visit_id = f"v{len(site.visits) + 1:03d}-{uuid.uuid4().hex[:4]}"
            visits_dir = self._site_dir(site_id) / "visits"
            visits_dir.mkdir(exist_ok=True)
            if image is not None:
                np.save(self.visit_image_path(site_id, visit_id), np.asarray(image))
            detail = {"id": visit_id, "site": site_id, "ts": time.time(), "mount_id": self.mount_id,
                      "has_image": image is not None, **_json_safe(result)}
            (visits_dir / f"{visit_id}.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
            summary = {k: detail.get(k) for k in
                       ("id", "ts", "mount_id", "status", "residual_nm", "sigma_nm", "ncc", "ambiguity",
                        "action_id", "run_id", "requested_by", "has_image")}
            site.visits.append(summary)
            self._write_site(site)
            self.log("site_visited", site=site_id, visit=visit_id, status=detail.get("status"),
                     residual_nm=detail.get("residual_nm"), sigma_nm=detail.get("sigma_nm"))
            return detail

    def visit(self, site_id: str, visit_id: str) -> dict:
        path = self._site_dir(site_id) / "visits" / f"{visit_id}.json"
        if not path.exists():
            raise KeyError(f"site {site_id!r} has no visit {visit_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def visit_image(self, site_id: str, visit_id: str) -> np.ndarray:
        return np.load(self.visit_image_path(site_id, visit_id))

    # -- mount and mapping -----------------------------------------------------
    def set_mapping(self, mapping: StageMapping) -> None:
        with self._lock:
            self.mapping = mapping
            self._write_index()
            self.log("mapping_calibrated", mapping=mapping.to_json())

    def new_mount(self, mount_id: str | None = None) -> str:
        """The specimen was unloaded and remounted: old coordinates are now only a rough guess."""
        with self._lock:
            self.mount_id = mount_id or f"mount-{time.strftime('%Y%m%dT%H%M%S')}"
            self.mount = MountTransform()
            self._write_index()
            self.log("remounted", mount_id=self.mount_id)
            return self.mount_id

    def set_mount_transform(self, transform: MountTransform) -> None:
        with self._lock:
            self.mount = transform
            self._write_index()
            self.log("mount_transform_fitted", mount_id=self.mount_id, transform=transform.to_json())

    # -- views -----------------------------------------------------------------
    def summary(self) -> dict:
        return _json_safe({
            "specimen_id": self.specimen_id,
            "mount": {"id": self.mount_id, "transform": self.mount.to_json()},
            "mapping": self.mapping.to_json(),
            "sites": [self._site_summary(s) for s in self.sites.values()],
        })

    def _site_summary(self, site: Site) -> dict:
        last = site.visits[-1] if site.visits else None
        return {"id": site.id, "name": site.name, "landmark": site.landmark, "stage": site.stage,
                "predicted_stage": self.predicted_stage(site.id), "fov_m": site.fov_m,
                "imsize": site.imsize, "created": site.created, "visits": len(site.visits),
                "last_visit": last, "notes": site.notes}


def to_png(image: np.ndarray, max_side: int = 512) -> bytes:
    """Grayscale PNG for display (percentile-stretched); the .npy stays the record."""
    from PIL import Image

    a = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(a, [0.5, 99.5]) if a.size else (0.0, 1.0)
    a = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    img = Image.fromarray((a * 255).astype(np.uint8))
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
