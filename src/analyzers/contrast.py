"""Contrast metrics.

Computes RMS contrast (standard deviation of intensities) and Michelson
contrast ((max-min)/(max+min)). Both are deterministic local measures.
"""

from __future__ import annotations

import numpy as np

from ..utils.image_ops import ensure_gray


def rms_contrast(image: np.ndarray) -> float:
    gray = ensure_gray(image).astype(np.float64)
    if gray.size == 0:
        return 0.0
    return float(gray.std())


def michelson_contrast(image: np.ndarray) -> float:
    gray = ensure_gray(image).astype(np.float64)
    if gray.size == 0:
        return 0.0
    gmax = float(gray.max())
    gmin = float(gray.min())
    denom = gmax + gmin
    if denom <= 1e-9:
        return 0.0
    return (gmax - gmin) / denom


def analyze(image: np.ndarray) -> dict[str, float]:
    return {
        "contrast_rms": rms_contrast(image),
        "contrast_michelson": michelson_contrast(image),
    }
