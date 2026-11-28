"""Detectors that run on each map tile as it arrives.

A detector is any callable ``detector(image) -> list of hits``, where a hit is
``{"px": (row, col) or [(row, col), ...], "score": float, "label": str,
"reason": str}`` in the tile's own pixels. Attach one with
``LiveMap.add_detector(name, detector, version)``; every finding it makes is
credited to ``model:<name>@<version>`` and starts as provisional.

``BlobDetector`` is the baseline: bright (or dark) blobs at one size scale,
found with a Laplacian-of-Gaussian filter. It needs only numpy and scipy, so a
map can show findings on any machine. A trained model (BlobNet, SAM2, ...) is
attached the same way: wrap it in a function with the signature above. Compare
any model against this baseline before claiming it helps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy import ndimage


@dataclass
class BlobDetector:
    """Blobs of about ``sigma_px`` radius whose contrast stands out from the tile.

    ``threshold`` is in robust standard deviations of the filtered tile (median
    absolute deviation), so it means the same on bright and dim tiles.
    """

    sigma_px: float = 4.0
    threshold: float = 6.0
    bright: bool = True
    label: str = "blob"
    min_distance_px: int = 0  # 0: 2 x sigma
    edge_px: int = 0  # ignore hits this close to the tile edge; 0: sigma
    max_hits: int = 200

    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2:
            raise ValueError("BlobDetector needs a 2D image")
        response = -ndimage.gaussian_laplace(img, self.sigma_px) * self.sigma_px**2
        if not self.bright:
            response = -response
        median = float(np.median(response))
        spread = float(np.median(np.abs(response - median))) * 1.4826
        if spread == 0:
            return []
        z = (response - median) / spread
        distance = self.min_distance_px or max(1, int(round(2 * self.sigma_px)))
        peaks = (z == ndimage.maximum_filter(z, size=2 * distance + 1)) & (z >= self.threshold)
        edge = self.edge_px or int(np.ceil(self.sigma_px))
        peaks[:edge, :] = peaks[-edge:, :] = False
        peaks[:, :edge] = peaks[:, -edge:] = False
        rows, cols = np.nonzero(peaks)
        order = np.argsort(-z[rows, cols])[: self.max_hits]
        return [
            {
                "px": (float(rows[i]), float(cols[i])),
                "score": float(z[rows[i], cols[i]]),
                "label": self.label,
                "reason": f"LoG sigma {self.sigma_px:g} px, {z[rows[i], cols[i]]:.1f} robust SD above the tile",
            }
            for i in order
        ]
