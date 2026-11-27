"""Image registration with an honest uncertainty, for "return to this spot".

Coordinates from the stage are only a starting guess: backlash, drift, and
above all unloading and remounting the specimen move a site by far more than a
field of view. Before calling a revisit a match, the old image has to be found
again in a new one, and the match has to be good enough to trust.

:func:`locate` finds a template (usually a crop of the stored reference image)
inside a search image with FFT-based normalized cross-correlation, and reports
four independent pieces of evidence alongside the position:

``ncc``
    Peak normalized cross-correlation, -1..1. How alike the two images are.
``ambiguity``
    Best score *outside* an exclusion radius around the peak, divided by the
    peak. Repetitive specimens (lattices, arrays of identical particles) give
    a second peak almost as high as the first: that is how a registration
    picks the wrong spot confidently. Near 1 means "could be either".
``sigma_px``
    Position uncertainty from internal consistency. The template is split
    into quadrants, each is located on its own near the global answer, and
    the spread of their disagreement (divided by sqrt(n)) is the uncertainty.
``consistency``
    Fraction of the textured quadrants that agree with the global position
    within ``agree_px``. A wrong match rarely has all four parts agreeing.

:func:`judge` turns these into ``match`` / ``ambiguous`` / ``no_match`` against
:class:`MatchCriteria`. Only ``match`` lets a caller claim it is back at the
site.

Convention: positions are (row, col) = (y, x) in pixels, y down. A template
"found at" ``(r, c)`` means the template's centre sits at pixel ``(r, c)`` of
the search image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

_EPS = 1e-12


# ----------------------------------------------------------------------------
# Normalized cross-correlation
# ----------------------------------------------------------------------------

def _window_sums(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Sum of every h x w window of *image* (valid positions only)."""
    c = np.cumsum(np.cumsum(image, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]


def ncc_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of *template* at every valid offset of *search*.

    Returns an array of shape ``(H - h + 1, W - w + 1)``; entry ``(i, j)`` is the
    Pearson correlation of the template with ``search[i:i+h, j:j+w]``.
    """
    search = np.asarray(search, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    H, W = search.shape
    h, w = template.shape
    if h > H or w > W:
        raise ValueError(f"template {template.shape} is larger than the search image {search.shape}")
    t = template - template.mean()
    t_norm = np.sqrt(np.sum(t * t))
    if t_norm < _EPS:
        return np.zeros((H - h + 1, W - w + 1))
    shape = (H, W)
    corr = np.fft.irfft2(np.fft.rfft2(search, shape) * np.fft.rfft2(t[::-1, ::-1], shape), shape)
    corr = corr[h - 1:H, w - 1:W]
    s1 = _window_sums(search, h, w)
    s2 = _window_sums(search * search, h, w)
    var = np.maximum(s2 - s1 * s1 / (h * w), 0.0)
    denom = np.sqrt(var) * t_norm
    # Flat search windows have no defined correlation; score them 0, not inf.
    floor = 1e-6 * float(denom.max() if denom.size else 1.0) + _EPS
    out = np.where(denom > floor, corr / np.maximum(denom, floor), 0.0)
    return np.clip(out, -1.0, 1.0)


def _subpixel(score: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """Parabolic refinement of an integer peak."""
    def fit(m1: float, c: float, p1: float) -> float:
        d = m1 - 2 * c + p1
        return 0.0 if abs(d) < _EPS else float(np.clip(0.5 * (m1 - p1) / d, -0.5, 0.5))
    di = fit(score[i - 1, j], score[i, j], score[i + 1, j]) if 0 < i < score.shape[0] - 1 else 0.0
    dj = fit(score[i, j - 1], score[i, j], score[i, j + 1]) if 0 < j < score.shape[1] - 1 else 0.0
    return i + di, j + dj


def _texture(image: np.ndarray) -> float:
    """Structure, not noise: standard deviation after a light blur that suppresses shot noise."""
    image = np.asarray(image, dtype=np.float64)
    return float(ndimage.gaussian_filter(image, 1.5).std())


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

@dataclass
class Location:
    """Where a template was found in a search image, with the evidence for it."""

    row: float                 # template centre, search-image pixels
    col: float
    ncc: float
    ambiguity: float           # second-best peak / best peak (outside exclusion radius)
    second_row: float | None
    second_col: float | None
    sigma_px: float            # 1-sigma position uncertainty, pixels
    consistency: float         # fraction of textured quadrants agreeing
    quadrants_used: int
    rotation_deg: float = 0.0  # rotation applied to the template for this match
    scale: float = 1.0         # zoom applied to the template before matching

    def to_json(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in asdict(self).items()}


@dataclass
class MatchCriteria:
    """When a registration may be called a match. Defaults are deliberately strict."""

    min_ncc: float = 0.45
    max_ambiguity: float = 0.80
    max_sigma_px: float = 1.5
    min_consistency: float = 0.75

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    status: str                       # "match" | "ambiguous" | "no_match"
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "match"

    def to_json(self) -> dict:
        return {"status": self.status, "reasons": list(self.reasons)}


def judge(loc: Location, criteria: MatchCriteria, sigma_limit_px: float | None = None) -> Verdict:
    """Classify a :class:`Location`.

    ``no_match``: the images are not alike enough (low NCC). ``ambiguous``: they
    are alike, but another position is nearly as good, the parts of the image
    disagree, or the position is too uncertain. Either way, the caller must not
    claim it found the site.
    """
    limit = criteria.max_sigma_px if sigma_limit_px is None else sigma_limit_px
    if loc.ncc < criteria.min_ncc:
        return Verdict("no_match", [f"ncc {loc.ncc:.2f} < {criteria.min_ncc:.2f}"])
    reasons = []
    if loc.ambiguity > criteria.max_ambiguity:
        reasons.append(f"second peak at {loc.ambiguity:.0%} of the best (> {criteria.max_ambiguity:.0%}): "
                       f"repetitive structure, another position fits almost as well")
    if loc.consistency < criteria.min_consistency:
        reasons.append(f"only {loc.consistency:.0%} of image quadrants agree on the position "
                       f"(need {criteria.min_consistency:.0%})")
    if not np.isfinite(loc.sigma_px) or loc.sigma_px > limit:
        reasons.append(f"position uncertainty {loc.sigma_px:.2f} px > {limit:.2f} px")
    return Verdict("ambiguous", reasons) if reasons else Verdict("match", [])


# ----------------------------------------------------------------------------
# Locating
# ----------------------------------------------------------------------------

def _peak(score: np.ndarray, exclusion_px: float) -> tuple[int, int, float, float, int | None, int | None]:
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    best = float(score[i, j])
    rr, cc = np.ogrid[:score.shape[0], :score.shape[1]]
    # Rivals are other *local maxima*: the shoulder of a broad main peak is not
    # an alternative position, however high it is.
    local_max = score >= ndimage.maximum_filter(score, size=5, mode="nearest")
    outside = ((rr - i) ** 2 + (cc - j) ** 2 > exclusion_px ** 2) & local_max
    if outside.any():
        masked = np.where(outside, score, -np.inf)
        i2, j2 = np.unravel_index(int(np.argmax(masked)), score.shape)
        second = float(masked[i2, j2])
    else:
        i2 = j2 = None
        second = -1.0
    return int(i), int(j), best, second, i2, j2


def _quadrant_check(search: np.ndarray, template: np.ndarray, top: float, left: float,
                    radius_px: int, agree_px: float) -> tuple[float, float, int]:
    """Locate each quadrant of *template* near where the whole one was found."""
    h, w = template.shape
    hh, hw = h // 2, w // 2
    if hh < 8 or hw < 8:
        return float("nan"), 0.0, 0
    full_tex = _texture(template)
    devs = []
    for r0, c0 in ((0, 0), (0, hw), (hh, 0), (hh, hw)):
        quad = template[r0:r0 + hh, c0:c0 + hw]
        if _texture(quad) < 0.25 * full_tex:   # featureless quadrant: no evidence either way
            continue
        exp_r, exp_c = top + r0, left + c0
        r_lo = int(max(0, np.floor(exp_r) - radius_px))
        c_lo = int(max(0, np.floor(exp_c) - radius_px))
        r_hi = int(min(search.shape[0], np.ceil(exp_r) + radius_px + hh + 1))
        c_hi = int(min(search.shape[1], np.ceil(exp_c) + radius_px + hw + 1))
        window = search[r_lo:r_hi, c_lo:c_hi]
        if window.shape[0] < hh or window.shape[1] < hw:
            continue
        score = ncc_map(window, quad)
        qi, qj = np.unravel_index(int(np.argmax(score)), score.shape)
        si, sj = _subpixel(score, int(qi), int(qj))
        devs.append(np.hypot(r_lo + si - exp_r, c_lo + sj - exp_c))
    if not devs:
        return float("nan"), 0.0, 0
    devs = np.asarray(devs)
    agree = float(np.mean(devs <= agree_px))
    sigma = float(np.sqrt(np.mean(devs ** 2)) / np.sqrt(len(devs)))
    return max(sigma, 0.1), agree, len(devs)


def _rotate_crop(template: np.ndarray, angle_deg: float, keep: float) -> np.ndarray:
    """Rotate about the centre and keep the central *keep* fraction (no blank corners)."""
    if angle_deg:
        template = ndimage.rotate(template, angle_deg, reshape=False, order=1, mode="reflect")
    return _central(template, keep)


def _central(image: np.ndarray, fraction: float) -> np.ndarray:
    """Central crop that keeps the exact centre (same size parity as the image)."""
    h, w = image.shape
    ch, cw = max(8, int(round(h * fraction))), max(8, int(round(w * fraction)))
    ch -= (h - ch) % 2
    cw -= (w - cw) % 2
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return image[r0:r0 + ch, c0:c0 + cw]


def locate(template: np.ndarray, search: np.ndarray, *, rotations_deg=(0.0,), scale: float = 1.0,
           exclusion_px: float | None = None, agree_px: float = 2.0,
           quadrant_radius_px: int = 4) -> Location:
    """Find *template* in *search*.

    Args:
        rotations_deg: candidate rotations of the template (the specimen may have
            been remounted at a different angle). The best-scoring one wins and is
            refined to 0.25 deg when more than one is given.
        scale: zoom the template by this factor first, e.g. to bring a
            high-magnification reference onto a lower-magnification survey grid.
        exclusion_px: radius around the best peak that does not count as an
            alternative position. Default: a quarter of the template size.
    """
    search = np.asarray(search, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if scale != 1.0:
        template = ndimage.zoom(template, scale, order=1)
    rotations = [float(a) for a in rotations_deg] or [0.0]
    keep = 1.0 if rotations == [0.0] else 0.70

    def score_at(angle: float):
        t = _rotate_crop(template, angle, keep)
        return t, ncc_map(search, t)

    best = None
    for angle in rotations:
        t, score = score_at(angle)
        peak = float(score.max())
        if best is None or peak > best[0]:
            best = (peak, angle, t, score)
    if len(rotations) > 1:
        step = min(np.diff(sorted(rotations))) if len(rotations) > 1 else 1.0
        for angle in np.arange(best[1] - step, best[1] + step + 1e-9, 0.25):
            t, score = score_at(float(angle))
            peak = float(score.max())
            if peak > best[0]:
                best = (peak, float(angle), t, score)
    _, angle, t, score = best

    h, w = t.shape
    excl = exclusion_px if exclusion_px is not None else max(3.0, min(h, w) / 4.0)
    i, j, top_score, second, i2, j2 = _peak(score, excl)
    si, sj = _subpixel(score, i, j)
    sigma, agree, used = _quadrant_check(search, t, si, sj, quadrant_radius_px, agree_px)
    ambiguity = second / top_score if top_score > _EPS else 1.0
    return Location(
        row=si + (h - 1) / 2.0,
        col=sj + (w - 1) / 2.0,
        ncc=top_score,
        ambiguity=float(max(0.0, ambiguity)),
        second_row=None if i2 is None else float(i2 + (h - 1) / 2.0),
        second_col=None if j2 is None else float(j2 + (w - 1) / 2.0),
        sigma_px=sigma,
        consistency=agree,
        quadrants_used=used,
        rotation_deg=angle,
        scale=scale,
    )


def center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Central *fraction* of an image (for same-magnification verification)."""
    return _central(np.asarray(image), fraction)


def offset_from_center(loc: Location, shape: tuple[int, int]) -> tuple[float, float]:
    """(dx, dy) in pixels, x right / y down, of the match from the search image centre."""
    rows, cols = shape
    return loc.col - (cols - 1) / 2.0, loc.row - (rows - 1) / 2.0


def rotation_range(max_deg: float, step_deg: float = 1.0) -> list[float]:
    if max_deg <= 0:
        return [0.0]
    n = int(np.floor(max_deg / step_deg))
    return [float(k * step_deg) for k in range(-n, n + 1)]
