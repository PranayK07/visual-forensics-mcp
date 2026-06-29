"""Blur metric: variance of the Laplacian.

Lower variance => fewer high-frequency details => blurrier region. This is the
classic Pech-Pacheco focus measure and is fully deterministic.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_ops import ensure_gray


def laplacian_variance(image: np.ndarray) -> float:
    """Return the variance of the Laplacian of ``image``."""
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def analyze(image: np.ndarray) -> dict[str, float]:
    """Return ``{"blur_score": <laplacian variance>}``."""
    return {"blur_score": laplacian_variance(image)}
