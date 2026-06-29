"""Shared image helpers used across analyzers.

All functions are deterministic and operate on numpy arrays. Inputs are
normalised to single-channel uint8 grayscale where appropriate.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised indirectly
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Return a 2D uint8 grayscale view of ``img``."""
    if img is None:
        raise ValueError("image is None")
    arr = np.asarray(img)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if cv2 is not None:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:  # luminosity fallback
            gray = (
                0.299 * arr[:, :, 0]
                + 0.587 * arr[:, :, 1]
                + 0.114 * arr[:, :, 2]
            )
    else:
        raise ValueError(f"unsupported image shape {arr.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def safe_round(value: float, decimals: int = 4) -> float:
    """Round, coercing non-finite values to 0.0 for JSON safety."""
    if value is None:
        return 0.0
    v = float(value)
    if not np.isfinite(v):
        return 0.0
    return round(v, decimals)


def robust_zscores(values: np.ndarray) -> np.ndarray:
    """Return signed z-scores of ``values`` (mean/std based).

    Returns an all-zero array when the standard deviation is ~0.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / std
