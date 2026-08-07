"""
Export a stitched mosaic as a "google-maps" style map bundle.

A map bundle is a plain directory that is simultaneously:

* **Human-viewable** - ``index.html`` is a self-contained pan/zoom tile viewer
  (no network, no external libraries; works from ``file://`` and inside
  strict-CSP webviews when tiles are fed as data URLs).
* **Agent-interpretable** - ``map.json`` is a machine-readable manifest
  describing every zoom level, the tile naming scheme, physical calibration
  (nm per pixel, stage coordinates), the source tiles and any point signals
  (e.g. EDS spectra) anchored to canvas coordinates.  An agent can navigate
  the sample coarse-to-fine by reading tiles like
  ``tiles/{z}/{x}/{y}.png`` without ever loading the full-resolution image.

Bundle layout::

    <out_dir>/
        map.json            manifest (see MANIFEST_FORMAT)
        index.html          self-contained zoomable viewer
        preview.png         small overview image
        tiles/{z}/{x}/{y}.png
        signals/...         optional copied signal files (EDS spectra etc.)

Zoom convention follows web maps: ``z = 0`` is the coarsest level (whole
mosaic in one tile), ``z = max_zoom`` is native resolution, each level doubles
the linear resolution of the previous one.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MANIFEST_FORMAT = "sciagent-map/1.1"


def export_map(
    canvas: np.ndarray,
    out_dir: str | Path,
    *,
    name: str = "sample_map",
    tile_size: int = 256,
    pixel_size_nm: Optional[float] = None,
    stage_origin: Optional[Sequence[float]] = None,
    source_tiles: Optional[List[Dict[str, Any]]] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    overviews: Optional[List[Dict[str, Any]]] = None,
    signal_files: Optional[Dict[str, str | Path]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a complete map bundle for *canvas* and return the bundle path.

    Parameters
    ----------
    canvas:
        Stitched mosaic (2-D greyscale or 3-D BGR, any integer/float dtype).
    out_dir:
        Bundle directory (created; existing tiles/ content is replaced).
    name:
        Human-readable map name stored in the manifest and viewer title.
    tile_size:
        Edge length of the square tiles in pixels.
    pixel_size_nm:
        Physical size of one canvas pixel in nanometres, if known.  Enables
        the scale bar in the viewer and physical coordinates for agents.
    stage_origin:
        Stage position (metres, instrument convention) of the canvas origin,
        if known.
    source_tiles:
        Optional provenance list, one entry per acquired tile, e.g.::

            {"grid": [r, c], "canvas_xy": [x, y], "size_px": [w, h],
             "stage_xy_m": [x, y], "data_key": "scan_HAADF_2026...",
             "signals": {"eds": "signals/eds_r0c0.json"}}
    annotations:
        Optional point/rect annotations, e.g.::

            {"type": "eds_point", "canvas_xy": [x, y], "label": "spot 1",
             "data": {...} | "file": "signals/eds_r0c0.json"}
    signal_files:
        Mapping of bundle-relative destination (must start with ``signals/``)
        to an existing source file that will be copied into the bundle.
    extra_metadata:
        Free-form dict merged into the manifest under ``"metadata"``.
    """
    out = Path(out_dir)
    tiles_dir = out / "tiles"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = _to_uint8(canvas)
    h, w = img.shape[:2]

    max_zoom = max(0, math.ceil(math.log2(max(1.0, max(h, w) / tile_size))))
    levels = []
    for z in range(max_zoom + 1):
        scale = 2.0 ** (z - max_zoom)
        lw, lh = max(1, round(w * scale)), max(1, round(h * scale))
        if z == max_zoom:
            level_img = img
        else:
            level_img = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)
        cols = math.ceil(lw / tile_size)
        rows = math.ceil(lh / tile_size)
        _write_level_tiles(level_img, tiles_dir / str(z), tile_size)
        levels.append(
            {
                "z": z,
                "scale": scale,
                "width_px": lw,
                "height_px": lh,
                "tile_cols": cols,
                "tile_rows": rows,
                "pixel_size_nm": (pixel_size_nm / scale) if pixel_size_nm else None,
            }
        )

    preview = img
    if max(h, w) > 512:
        s = 512.0 / max(h, w)
        preview = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "preview.png"), preview)

    if signal_files:
        for rel, src in signal_files.items():
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(f"signal destination must be bundle-relative: {rel}")
            dest = out / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    overviews_dir = out / "overviews"
    if overviews_dir.exists():
        shutil.rmtree(overviews_dir)
    overview_entries: List[Dict[str, Any]] = []
    if overviews:
        overviews_dir.mkdir(parents=True, exist_ok=True)
        for i, ov in enumerate(overviews):
            image = ov.get("image")
            entry = {k: v for k, v in ov.items() if k != "image"}
            if image is not None:
                rel_path = f"overviews/overview_{i:03d}.png"
                cv2.imwrite(str(out / rel_path), _to_uint8(np.asarray(image)))
                entry["image_path"] = rel_path
            overview_entries.append(entry)

    manifest = {
        "format": MANIFEST_FORMAT,
        "name": name,
        "canvas": {"width_px": w, "height_px": h},
        "tile_size": tile_size,
        "tile_url_template": "tiles/{z}/{x}/{y}.png",
        "max_zoom": max_zoom,
        "levels": levels,
        "pixel_size_nm": pixel_size_nm,
        "stage_origin_m": list(stage_origin) if stage_origin is not None else None,
        "source_tiles": source_tiles or [],
        "annotations": annotations or [],
        "overviews": overview_entries,
        "metadata": extra_metadata or {},
        "agent_instructions": (
            "This bundle is a zoomable map of a microscopy sample. "
            "Tiles are PNG images addressed as tiles/{z}/{x}/{y}.png where z=0 "
            "is the coarsest zoom (whole sample in one tile) and z=max_zoom is "
            "native resolution; each level doubles linear resolution. A canvas "
            "pixel (cx, cy) at native resolution appears at level z in tile "
            "x = floor(cx * scale / tile_size), y = floor(cy * scale / tile_size) "
            "where scale = 2**(z - max_zoom). To survey the sample, read the z=0 "
            "tile, then descend into interesting regions by increasing z. "
            "Entries in 'annotations' and 'source_tiles' are anchored to native "
            "canvas coordinates via 'canvas_xy'; 'signals' link to spectra or "
            "other measurements taken at those points. If pixel_size_nm is set, "
            "physical position (nm) = canvas_xy * pixel_size_nm. "
            "'overviews' are separately acquired, lower-magnification images "
            "registered against this canvas's coordinate system (not resamples "
            "of the fine pixels): each has 'image_path' plus a 2x3 'matrix' "
            "mapping [overview_x, overview_y, 1] to this canvas's native pixel "
            "coordinates, and a 'scale' (fine px per overview px). Use them for "
            "wider spatial context around the fine map, or as a starting point "
            "for further acquisition."
        ),
    }
    (out / "map.json").write_text(json.dumps(manifest, indent=2))

    (out / "index.html").write_text(_render_viewer_html(manifest))
    logger.info("Map bundle written to %s (%d zoom levels).", out, len(levels))
    return out


def export_static_html(bundle_dir: str | Path, out_html: Optional[str | Path] = None) -> Path:
    """Flatten a map bundle into one self-contained HTML file.

    Every tile is inlined as a base64 data URI, so the resulting file can be
    e-mailed, published, or opened anywhere with zero side files.  Practical
    for small/medium maps; for very large maps prefer the bundle itself.
    """
    import base64

    bundle = Path(bundle_dir)
    manifest = json.loads((bundle / "map.json").read_text())
    tiles: Dict[str, str] = {}
    for png in sorted((bundle / "tiles").rglob("*.png")):
        z, x, y = png.parent.parent.name, png.parent.name, png.stem
        tiles[f"{z}/{x}/{y}"] = "data:image/png;base64," + base64.b64encode(
            png.read_bytes()
        ).decode("ascii")

    html = _render_viewer_html(manifest)
    inline_script = (
        "<script>window.__TILE_DATA__ = "
        + json.dumps(tiles).replace("</", "<\\/")
        + ";</script>"
    )
    html = html.replace('<script id="manifest"', inline_script + '\n<script id="manifest"')

    out = Path(out_html) if out_html else bundle / f"{bundle.name}_standalone.html"
    out.write_text(html)
    logger.info("Standalone map written to %s (%d tiles inlined).", out, len(tiles))
    return out


def _write_level_tiles(level_img: np.ndarray, level_dir: Path, tile_size: int) -> None:
    lh, lw = level_img.shape[:2]
    cols = math.ceil(lw / tile_size)
    rows = math.ceil(lh / tile_size)
    for x in range(cols):
        col_dir = level_dir / str(x)
        col_dir.mkdir(parents=True, exist_ok=True)
        for y in range(rows):
            x0, y0 = x * tile_size, y * tile_size
            patch = level_img[y0: y0 + tile_size, x0: x0 + tile_size]
            cv2.imwrite(str(col_dir / f"{y}.png"), patch)


def _to_uint8(canvas: np.ndarray) -> np.ndarray:
    img = np.asarray(canvas)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _render_viewer_html(manifest: Dict[str, Any]) -> str:
    """Self-contained pan/zoom viewer with the manifest inlined.

    No external requests except relative tile <img> loads, so it works from
    file://, an artifact host, or a Tauri webview.
    """
    manifest_json = json.dumps(manifest).replace("</", "<\\/")
    title = manifest.get("name", "Sample map")
    return _VIEWER_TEMPLATE.replace("__TITLE__", title).replace(
        "__MANIFEST__", manifest_json
    )


_VIEWER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; overflow: hidden;
    background: #101214; color: #e6e6e6;
    font: 13px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
  #map { position: absolute; inset: 0; cursor: grab; overflow: hidden; }
  #map.dragging { cursor: grabbing; }
  #world { position: absolute; transform-origin: 0 0; }
  #world img.tile { position: absolute; image-rendering: pixelated;
    -webkit-user-drag: none; user-select: none; pointer-events: none; }
  .marker { position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px;
    border-radius: 50%; background: #ff7a1a; border: 2px solid #fff;
    box-shadow: 0 1px 4px rgba(0,0,0,.6); cursor: pointer; }
  .marker:hover { background: #ffa04d; }
  #hud { position: absolute; top: 10px; left: 10px; background: rgba(16,18,20,.85);
    border: 1px solid #2c2f33; border-radius: 8px; padding: 8px 12px; z-index: 5; }
  #hud h1 { font-size: 14px; font-weight: 600; margin-bottom: 2px; }
  #hud .sub { color: #9aa0a6; font-size: 11px; }
  #scalebar { position: absolute; left: 12px; bottom: 12px; z-index: 5;
    background: rgba(16,18,20,.85); border: 1px solid #2c2f33; border-radius: 6px;
    padding: 5px 10px; font-size: 11px; color: #cfd2d6; }
  #scalebar .bar { height: 3px; background: #e6e6e6; margin-top: 3px; }
  #panel { position: absolute; top: 0; right: 0; bottom: 0; width: 300px;
    background: rgba(16,18,20,.96); border-left: 1px solid #2c2f33; z-index: 6;
    padding: 14px; overflow-y: auto; display: none; }
  #panel.open { display: block; }
  #panel h2 { font-size: 13px; margin-bottom: 6px; }
  #panel pre { font-size: 11px; white-space: pre-wrap; word-break: break-all;
    background: #16181b; border-radius: 6px; padding: 8px; margin-top: 8px; }
  #panel .close { float: right; cursor: pointer; color: #9aa0a6; }
  #coords { position: absolute; right: 12px; bottom: 12px; z-index: 5;
    background: rgba(16,18,20,.85); border: 1px solid #2c2f33; border-radius: 6px;
    padding: 5px 10px; font-size: 11px; color: #9aa0a6; }
  #zoomctl { position: absolute; top: 10px; right: 10px; z-index: 5;
    display: flex; flex-direction: column; gap: 4px; }
  #zoomctl button { width: 30px; height: 30px; border-radius: 6px;
    border: 1px solid #2c2f33; background: rgba(16,18,20,.85); color: #e6e6e6;
    font-size: 16px; cursor: pointer; }
  #zoomctl button:hover { background: #24272b; }
</style>
</head>
<body>
<div id="map"><div id="world"></div></div>
<div id="hud"><h1>__TITLE__</h1><div class="sub" id="hudsub"></div></div>
<div id="zoomctl">
  <button id="zin" title="Zoom in">+</button>
  <button id="zout" title="Zoom out">&minus;</button>
  <button id="zfit" title="Fit" style="font-size:11px">fit</button>
</div>
<div id="scalebar" style="display:none"><span id="sblabel"></span><div class="bar" id="sbbar"></div></div>
<div id="coords"></div>
<div id="panel"><span class="close" id="pclose">&times;</span><div id="pbody"></div></div>
<script id="manifest" type="application/json">__MANIFEST__</script>
<script>
"use strict";
const M = JSON.parse(document.getElementById("manifest").textContent);
const TS = M.tile_size, MAXZ = M.max_zoom;
const W = M.canvas.width_px, H = M.canvas.height_px;
const INLINE = window.__TILE_DATA__ || null;
function tileSrc(z, x, y) {
  const k = `${z}/${x}/${y}`;
  if (INLINE && INLINE[k]) return INLINE[k];
  return M.tile_url_template.replace("{z}", z).replace("{x}", x).replace("{y}", y);
}
const mapEl = document.getElementById("map");
const world = document.getElementById("world");
const panel = document.getElementById("panel");
const pbody = document.getElementById("pbody");
let view = { scale: 1, x: 0, y: 0 };
let tileCache = new Map();

function fit() {
  const r = mapEl.getBoundingClientRect();
  const s = Math.min(r.width / W, r.height / H) * 0.95;
  view.scale = s;
  view.x = (r.width - W * s) / 2;
  view.y = (r.height - H * s) / 2;
  render();
}

function levelForScale(s) {
  let z = MAXZ + Math.ceil(Math.log2(Math.min(s, 1)));
  return Math.max(0, Math.min(MAXZ, z));
}

function render() {
  world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  const z = levelForScale(view.scale);
  const lv = M.levels[z];
  const r = mapEl.getBoundingClientRect();
  const upscale = 1 / lv.scale;
  const vx0 = Math.max(0, -view.x / view.scale), vy0 = Math.max(0, -view.y / view.scale);
  const vx1 = Math.min(W, (r.width - view.x) / view.scale);
  const vy1 = Math.min(H, (r.height - view.y) / view.scale);
  const tx0 = Math.max(0, Math.floor(vx0 * lv.scale / TS));
  const ty0 = Math.max(0, Math.floor(vy0 * lv.scale / TS));
  const tx1 = Math.min(lv.tile_cols - 1, Math.floor(vx1 * lv.scale / TS));
  const ty1 = Math.min(lv.tile_rows - 1, Math.floor(vy1 * lv.scale / TS));
  const wanted = new Set();
  for (let x = tx0; x <= tx1; x++) for (let y = ty0; y <= ty1; y++) {
    const key = `${z}/${x}/${y}`;
    wanted.add(key);
    if (!tileCache.has(key)) {
      const img = document.createElement("img");
      img.className = "tile";
      img.src = tileSrc(z, x, y);
      img.style.left = (x * TS * upscale) + "px";
      img.style.top = (y * TS * upscale) + "px";
      img.style.width = (Math.min(TS, lv.width_px - x * TS) * upscale) + "px";
      img.style.height = (Math.min(TS, lv.height_px - y * TS) * upscale) + "px";
      world.appendChild(img);
      tileCache.set(key, img);
    }
  }
  for (const [key, img] of tileCache) {
    const kz = +key.split("/")[0];
    if (kz !== z && !wanted.has(key)) { img.remove(); tileCache.delete(key); }
  }
  updateScalebar();
  document.getElementById("hudsub").textContent =
    `zoom level ${z}/${MAXZ} - ${W} x ${H} px` + (M.pixel_size_nm ? ` - ${(M.pixel_size_nm).toPrecision(3)} nm/px` : "");
}

function updateScalebar() {
  const sb = document.getElementById("scalebar");
  if (!M.pixel_size_nm) return;
  sb.style.display = "block";
  const targetPx = 120;
  const nm = targetPx / view.scale * M.pixel_size_nm;
  const nice = Math.pow(10, Math.floor(Math.log10(nm)));
  const val = [1, 2, 5, 10].map(m => m * nice).reduce((a, b) => Math.abs(b - nm) < Math.abs(a - nm) ? b : a);
  const px = val / M.pixel_size_nm * view.scale;
  document.getElementById("sbbar").style.width = px + "px";
  document.getElementById("sblabel").textContent = val >= 1000 ? (val / 1000).toPrecision(3) + " um" : val.toPrecision(3) + " nm";
}

function zoomAt(cx, cy, factor) {
  const ns = Math.min(Math.max(view.scale * factor, 0.01), 40);
  const k = ns / view.scale;
  view.x = cx - (cx - view.x) * k;
  view.y = cy - (cy - view.y) * k;
  view.scale = ns;
  render();
}

mapEl.addEventListener("wheel", e => {
  e.preventDefault();
  zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015));
}, { passive: false });

let drag = null;
mapEl.addEventListener("pointerdown", e => {
  if (e.target.classList.contains("marker")) return;
  drag = { x: e.clientX, y: e.clientY };
  mapEl.classList.add("dragging");
  mapEl.setPointerCapture(e.pointerId);
});
mapEl.addEventListener("pointermove", e => {
  const cx = (e.clientX - view.x) / view.scale, cy = (e.clientY - view.y) / view.scale;
  let txt = `${Math.round(cx)}, ${Math.round(cy)} px`;
  if (M.pixel_size_nm) txt += ` | ${(cx * M.pixel_size_nm).toPrecision(4)}, ${(cy * M.pixel_size_nm).toPrecision(4)} nm`;
  document.getElementById("coords").textContent = txt;
  if (!drag) return;
  view.x += e.clientX - drag.x;
  view.y += e.clientY - drag.y;
  drag = { x: e.clientX, y: e.clientY };
  render();
});
mapEl.addEventListener("pointerup", e => { drag = null; mapEl.classList.remove("dragging"); });
document.getElementById("zin").onclick = () => { const r = mapEl.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1.5); };
document.getElementById("zout").onclick = () => { const r = mapEl.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1 / 1.5); };
document.getElementById("zfit").onclick = fit;
document.getElementById("pclose").onclick = () => panel.classList.remove("open");

(M.annotations || []).forEach((a, i) => {
  if (!a.canvas_xy) return;
  const m = document.createElement("div");
  m.className = "marker";
  m.style.left = a.canvas_xy[0] + "px";
  m.style.top = a.canvas_xy[1] + "px";
  m.title = a.label || a.type || ("annotation " + i);
  m.addEventListener("pointerdown", e => e.stopPropagation());
  m.addEventListener("click", async e => {
    e.stopPropagation();
    let body = `<h2>${a.label || a.type || "Annotation"}</h2>` +
      `<div class="sub">canvas: ${a.canvas_xy[0].toFixed(0)}, ${a.canvas_xy[1].toFixed(0)} px</div>`;
    if (a.data) body += `<pre>${JSON.stringify(a.data, null, 2)}</pre>`;
    if (a.file) {
      try {
        const t = await (await fetch(a.file)).text();
        body += `<pre>${t.slice(0, 4000).replace(/</g, "&lt;")}</pre>`;
      } catch (_) {
        body += `<pre>signal file: ${a.file}\n(open it next to this bundle)</pre>`;
      }
    }
    pbody.innerHTML = body;
    panel.classList.add("open");
  });
  world.appendChild(m);
});

const scaledMarkers = () => {
  document.querySelectorAll(".marker").forEach(m => {
    m.style.transform = `scale(${1 / view.scale})`;
  });
};
const origRender = render;
render = function () { origRender(); scaledMarkers(); };

window.addEventListener("resize", render);
fit();
</script>
</body>
</html>
"""
