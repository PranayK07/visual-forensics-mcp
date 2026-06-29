"""Noise estimation.

Two complementary deterministic estimators:

* ``noise_sigma`` -- robust Gaussian-noise estimate using the median absolute
  deviation of a Laplacian convolution (Immerkaer-style), scaled to sigma.
* ``noise_median_residual`` -- std of the residual after median filtering,
  capturing impulsive / compression noise.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..utils.config import Config
from ..utils.image_ops import ensure_gray


def noise_sigma(image: np.ndarray) -> float:
    """Immerkaer fast noise variance estimation -> sigma."""
    gray = ensure_gray(image).astype(np.float64)
    if gray.size == 0 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian_mask = np.array(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64
    )
    convolved = cv2.filter2D(gray, -1, laplacian_mask)
    h, w = gray.shape
    sigma = np.sum(np.abs(convolved))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def noise_median_residual(image: np.ndarray, kernel: int = 3) -> float:
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    k = kernel if kernel % 2 == 1 else kernel + 1
    smoothed = cv2.medianBlur(gray, k)
    residual = gray.astype(np.float64) - smoothed.astype(np.float64)
    return float(residual.std())


def analyze(image: np.ndarray, config: Config | None = None) -> dict[str, float]:
    kernel = 3
    if config is not None:
        kernel = int(config.get("analyzers.noise.median_kernel", 3))
    return {
        "noise_sigma": noise_sigma(image),
        "noise_median_residual": noise_median_residual(image, kernel=kernel),
    }
