"""EDS: what X-ray spectra mean, registered as view kinds through ``register_kind``.

The generic views (``asyncroscopy.data.views``) find peaks; this module says which X-ray lines could make them, which
are detector artefacts (Si escape, sum peaks, the window's rise of the continuum), and which elements best explain
them. ``spectrum`` and ``spectrum_image`` are re-registered with this knowledge, ``composition`` is added.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from asyncroscopy.data.views import (
    Axis,
    Dataset,
    Kind,
    LineDomain,
    PeakRow,
    ViewOptions,
    _fmt,
    _png,
    _read,
    _real,
    energy_axis,
    line_kind,
    line_peaks as _line_peaks,
    register_kind,
    spectrum_image_kind,
)

# Below WINDOW_RISE_EV an EDS peak must also stand RISE_SIGMA above a smooth curve fitted with it over
# +-RISE_HALF_WIDTHS line widths: the detector window bends the continuum into a broad rise near 0.6 keV.
WINDOW_RISE_EV = 1200.0
RISE_HALF_WIDTHS = 3.0
RISE_SIGMA = 3.0
SI_ESCAPE_EV = 1740.0
# Silicon-drift EDS detector resolution, as the twin models it (twin/eds_model.py).
EDS_FWHM_MN_KA_EV = 130.0
MN_KA_EV = 5895.0


def eds_fwhm_ev(energy_ev: Any) -> np.ndarray:
    """Silicon-drift detector line width at an energy (eV), at least 30 eV."""
    return np.sqrt(np.maximum(EDS_FWHM_MN_KA_EV**2 + 2.5 * (np.asarray(energy_ev, dtype=np.float64) - MN_KA_EV), 30.0**2))


@lru_cache(maxsize=1)
def _xray_lines() -> tuple[tuple[float, str, float], ...]:
    """(energy eV, "Fe K-L3", weight) for the X-ray lines in pyTEMlib's tables, down to 1% of their family's
    strongest: a Pt particle's L3-M1 and L2-N4 peaks (weights 0.06 and 0.08) were left unnamed at 10%."""
    try:
        from pyTEMlib import xrpa_x_sections
    except ImportError:
        return ()
    out = []
    for entry in xrpa_x_sections.x_sections.values():
        for line, info in (entry.get("lines") or {}).items():
            if float(info.get("weight", 0)) >= 0.01 and 100.0 <= float(info["position"]) <= 40000.0:
                out.append((float(info["position"]), f"{entry['name']} {line}", float(info["weight"])))
    return tuple(sorted(out))


def xray_candidates(energy_ev: float, tolerance_ev: float, peaks_ev: list[float] | None = None, energy_max_ev: float = math.inf) -> str:
    """The lines that could make a peak, nearest first; "unidentified" when lines are near but none could.

    A line could not when it is a minor line whose family's main line has no peak, or when it is not in its element's
    main family (K if in range, else L, else M) and that family's main line has no peak. A carbon peak was once listed as Ca L and Ag M lines and read as both; listing them as
    "less likely" still had them named as present (qwen runs 2026-11-03), so they are left out.
    """
    near = sorted((abs(e - energy_ev), name, w) for e, name, w in _xray_lines() if abs(e - energy_ev) <= tolerance_ev)
    likely = [name for _d, name, weight in near if peaks_ev is None or not _unlikely_line(name, weight, peaks_ev, energy_max_ev)]
    if near and not likely:
        return "unidentified: no element whose main lines are present has a line here"
    return ", ".join(likely[:3])


@lru_cache(maxsize=1)
def _lines_by_element() -> dict[str, list[tuple[float, str, float]]]:
    out: dict[str, list[tuple[float, str, float]]] = {}
    for energy, name, weight in _xray_lines():
        element, line = name.split(" ", 1)
        out.setdefault(element, []).append((energy, line, weight))
    return out


def _unlikely_line(name: str, weight: float, peaks_ev: list[float], energy_max_ev: float) -> str:
    element, line = name.split(" ", 1)
    lines = _lines_by_element()[element]
    _w, top_energy, top_line = max((w, e, l) for e, l, w in lines if l[0] == line[0])
    if weight < _w and not _seen(top_energy, peaks_ev):
        return f"minor line, no {element} {top_line} peak at {top_energy:.0f} eV"
    # The element's main family is its K lines when they are in range, else its L lines, else its M lines.
    main = next((f for f in "KLM" if any(l[0] == f and e <= energy_max_ev for e, l, _w in lines)), line[0])
    if line[0] == main:
        return ""
    _w, main_energy, main_line = max((w, e, l) for e, l, w in lines if l[0] == main and e <= energy_max_ev)
    return "" if _seen(main_energy, peaks_ev) else f"no {element} {main_line} peak at {main_energy:.0f} eV"


def _seen(energy_ev: float, peaks_ev: list[float]) -> bool:
    return any(abs(p - energy_ev) <= float(eds_fwhm_ev(energy_ev)) / 2 for p in peaks_ev)


def is_xray(ds: Dataset) -> bool:
    words = " ".join(str(ds.attrs.get(k, "")) + " " + str(ds.parent_attrs.get(k, "")) for k in ("detector", "acquisition_type", "technique")).lower()
    return "eds" in words or "edx" in words or "x-ray" in words or "xray" in words


def _width_channels(x: np.ndarray, axis: Axis, step: int) -> Any:
    return eds_fwhm_ev(x) / abs(axis.scale * step) if axis.scale else 3.0


def _name_peaks(rows: list[PeakRow], x: np.ndarray, y: np.ndarray, axis: Axis, step: int, fwhm: Any) -> list[PeakRow]:
    """Notes for the peaks of an X-ray spectrum, given strongest (by significance) first: artefacts, the detector window's
    rise, then candidate lines. Every peak is known before any is named, so a line can be checked against its element's other lines."""
    noted: list[PeakRow] = []
    for position, net, sigma, index, _note in rows:
        note = _artefact(position, [row[0] for row in noted], _line_tolerance(position, axis, step))
        noted.append((position, net, sigma, index, note))
    indices = [row[3] for row in noted]
    noted = [
        (e, net, sig, i, WINDOW_RISE_NOTE if not note and e < WINDOW_RISE_EV and _line_sigma(y, fwhm, i, indices) <= RISE_SIGMA else note)
        for e, net, sig, i, note in noted
    ]
    positions = [row[0] for row in noted if row[4] != WINDOW_RISE_NOTE]
    top = float(np.nanmax(x)) if x.size else math.inf
    return [(e, net, sig, i, note or xray_candidates(e, _line_tolerance(e, axis, step), positions, top)) for e, net, sig, i, note in noted]


WINDOW_RISE_NOTE = "unidentified: a broad rise of the continuum (the detector window), not an X-ray line"


def _line_sigma(values: np.ndarray, fwhm_channels: Any, index: int, others: list[int]) -> float:
    """How many sigmas a line-width peak at ``index`` stands above a smooth (quadratic) curve fitted with it.

    The detector window bends the low-energy continuum into a rise near 0.6 keV that the running background takes
    for a peak, which then read as O or F (qwen runs 2026-11-03; 93 of 288 twin spectra without O or F). A quadratic
    takes that rise; a carbon K peak on it keeps its line-width shape. Other peaks in the window are fitted with it.
    """
    y = np.nan_to_num(np.asarray(values, dtype=np.float64))
    width = np.broadcast_to(np.asarray(fwhm_channels, dtype=np.float64), y.shape)
    reach = max(2, int(round(RISE_HALF_WIDTHS * float(width[index]))))
    lo, hi = max(0, index - reach), min(y.size, index + reach + 1)
    x = np.arange(lo, hi, dtype=np.float64)
    u = (x - index) / reach
    centres = [index] + [j for j in others if lo <= j < hi and j != index]
    columns = [np.ones_like(u), u, u * u] + [np.exp(-0.5 * ((x - j) / (max(float(width[j]), 1.0) / 2.355)) ** 2) for j in centres]
    weight = 1.0 / np.sqrt(np.maximum(y[lo:hi], 1.0))  # Poisson: variance = counts
    design = np.stack(columns, axis=1) * weight[:, None]
    if design.shape[0] <= design.shape[1]:
        return math.inf  # too few channels to tell: keep the peak
    coefficients, *_ = np.linalg.lstsq(design, y[lo:hi] * weight, rcond=None)
    variance = float(np.linalg.pinv(design.T @ design)[3, 3])
    return float(coefficients[3]) / math.sqrt(max(variance, 1e-300))


def _line_tolerance(energy_ev: float, axis: Axis, step: int) -> float:
    """Half the line width, or all of it below 1 keV: there the detector window cuts the background steeply and a
    peak's top shifts up (a carbon K peak, 277 eV, is found at 298 to 308 eV)."""
    width = float(eds_fwhm_ev(energy_ev))
    return max(width if energy_ev < 1000.0 else width / 2, 2 * abs(axis.scale * step))


def _artefact(energy: float, stronger: list[float], tolerance: float) -> str:
    """A note when an EDS peak may be a Si escape or a sum (pile-up) peak of a stronger one."""
    for e in stronger:
        if abs(energy - (e - SI_ESCAPE_EV)) <= tolerance:
            return f"possible Si escape peak of the {_fmt(e)} eV peak"
        if abs(energy - 2 * e) <= tolerance:
            return f"possible sum peak (2 x {_fmt(e)} eV)"
    return ""


def note_elements(note: str) -> set[str]:
    """The elements a peak's note says could make it: none for an escape, sum or unidentified peak."""
    if not note or note.startswith(("possible", "unidentified")):
        return set()
    return {c.strip().split(" ")[0] for c in note.split(",") if c.strip()}


def likely_elements(rows: list[tuple], share: float = 0.25) -> str:
    """The fewest elements that explain the peaks, as the share of the peak signal (summed sigma) each explains:
    "Au 85%, Pt 7%", kept while an element explains at least ``share`` of what the best one does.

    The element explaining the most of what is left takes its peaks, then the next; elements tied on the same peaks
    share them. Every element that could make a peak once got all of it: a Pd particle's weaker L lines read as Cd and
    In too, and one peak as "Ca 100%, Ag 100%" (qwen runs 2026-11-03).
    """
    left = {i: (float(row[2]), note_elements(row[4])) for i, row in enumerate(rows)}
    score: dict[str, float] = {}
    while True:
        gain: dict[str, float] = {}
        for sigma, elements in left.values():
            for element in elements:
                gain[element] = gain.get(element, 0.0) + sigma
        if not gain:
            break
        best = max(gain.values())
        tied = sorted(e for e in gain if gain[e] >= best * (1 - 1e-9))
        taken = [i for i, (_s, elements) in left.items() if set(tied) & elements]
        for i in taken:
            sigma, elements = left.pop(i)
            for element in set(tied) & elements:
                score[element] = score.get(element, 0.0) + sigma / len(set(tied) & elements)
    if not score:
        return ""
    total = sum(float(r[2]) for r in rows) or 1.0
    top = max(score.values())
    ranked = sorted((e for e in score if score[e] >= share * top), key=lambda e: (-score[e], e))[:4]
    return ", ".join(f"{e} {100.0 * score[e] / total:.0f}%" for e in ranked)


def _reading(rows: list[PeakRow], total: float) -> list[str]:
    likely = likely_elements(rows)
    if likely:
        # A small model given only per-peak candidates did not conclude; the ranking says which elements the
        # peaks point to, strongest first, and stays a reading of the peaks (a light element can still hide).
        return [f"elements most consistent with these peaks (the share of the peak signal each one explains): {likely}"]
    # only a window rise, escape, sum or unidentified peaks: say so, or the data rows get read as lines
    return [f"no X-ray line of any element detected above noise at this exposure (total {_fmt(total)} counts); "
            "a longer exposure is needed to rule elements out"]


def _no_peaks(total: float) -> str:
    if total == 0:
        return ("0 counts: no electrons reached the detector, so a longer exposure will not help. The beam was probably "
                "blanked (unblank_beam, then acquire again) or the detector retracted")
    return (f"no peaks detected above noise at this exposure (total {_fmt(total)} counts); "
            "a longer exposure is needed to rule elements out")


XRAY = LineDomain(
    applies=lambda ds, axis: axis.units == "eV" and is_xray(ds),
    width_channels=_width_channels,
    name_peaks=_name_peaks,
    reading=_reading,
    no_peaks=_no_peaks,
)


def line_peaks(x: np.ndarray, y: np.ndarray, axis: Axis, step: int = 1) -> list[PeakRow]:
    """The significant peaks of an X-ray spectrum, strongest (net counts) first, each with its candidate lines or artefact note."""
    return _line_peaks(x, y, axis, step, XRAY)


def _composition_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str, str]:
    values = _read(ds, (slice(0, ds.shape[0]),))
    figure = Figure(figsize=(6.0, 3.5), dpi=100)
    plot = figure.add_subplot()
    plot.bar(np.arange(values.size), values)
    plot.set_xticks(np.arange(values.size))
    plot.set_xticklabels(ds.attrs["elements"])
    plot.set_xlabel("element")
    plot.set_ylabel("relative intensity")
    figure.tight_layout()
    return _png(figure), f"read [0:{ds.shape[0]}]", _composition_text(ds, opts)[0]


def _composition_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    values = _read(ds, (slice(0, ds.shape[0]),))
    return "\n".join(f"{element}: {_fmt(float(v))}" for element, v in zip(ds.attrs["elements"], values)), f"read [0:{ds.shape[0]}]"


def _has_elements(ds: Dataset) -> bool:
    elements = ds.attrs.get("elements")
    return ds.ndim == 1 and _real(ds) and isinstance(elements, list) and len(elements) == ds.shape[0] and all(isinstance(e, str) for e in elements)


register_kind(
    line_kind(
        "spectrum", lambda ds: ds.ndim == 1 and _real(ds) and energy_axis(ds.attrs) is not None, lambda ds: energy_axis(ds.attrs),
        "1-D data on a calibrated axis (EDS, EELS, Raman, a diffractogram ...); its significant peaks listed strongest first, "
        "with candidate X-ray lines for an X-ray spectrum",
        domain=XRAY,
    ),
    before="profile",
)
register_kind(Kind("composition", _has_elements, _composition_png, _composition_text, "one value per element"), before="profile")
register_kind(spectrum_image_kind(XRAY), before="stack")
