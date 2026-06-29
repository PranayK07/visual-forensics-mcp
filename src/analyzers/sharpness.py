"""Sharpness metric: edge-energy based focus measures.

Uses the Tenengrad measure (mean squared Sobel gradient magnitude) plus the
mean gradient magnitude. Higher values indicate sharper edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_ops import ensure_gray


def tenengrad(image: np.ndarray) -> float:
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx * gx + gy * gy
    return float(fm.mean())


def mean_gradient(image: np.ndarray) -> float:
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def analyze(image: np.ndarray) -> dict[str, float]:
    return {
        "sharpness_tenengrad": tenengrad(image),
        "sharpness_mean_gradient": mean_gradient(image),
    }
