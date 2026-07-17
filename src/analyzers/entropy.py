"""Shannon entropy of the grayscale intensity histogram (bits)."""

from __future__ import annotations

import numpy as np

from ..utils.config import Config
from ..utils.image_ops import ensure_gray


def shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    # Build the fixed-range histogram explicitly.  NumPy 2.2 on Python 3.14
    # can produce a 257-element internal bincount for ``np.histogram`` at the
    # upper uint8 boundary.  Explicit index scaling is both deterministic and
    # correct for any positive configured bin count.
    bins = max(1, int(bins))
    values = gray.reshape(-1).astype(np.int64, copy=False)
    indices = np.minimum((values * bins) // 256, bins - 1)
    hist = np.bincount(indices, minlength=bins)[:bins]
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def analyze(image: np.ndarray, config: Config | None = None) -> dict[str, float]:
    bins = 256
    if config is not None:
        bins = int(config.get("analyzers.entropy.histogram_bins", 256))
    return {"entropy": shannon_entropy(image, bins=bins)}
