"""Shannon entropy of the grayscale intensity histogram (bits)."""

from __future__ import annotations

import numpy as np

from ..utils.config import Config
from ..utils.image_ops import ensure_gray


def shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
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
