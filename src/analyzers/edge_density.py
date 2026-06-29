"""Edge density: fraction of pixels classified as edges by Canny."""

from __future__ import annotations

import cv2
import numpy as np

from ..utils.config import Config
from ..utils.image_ops import ensure_gray


def edge_density(image: np.ndarray, low: int = 50, high: int = 150) -> float:
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, low, high)
    return float(np.count_nonzero(edges)) / float(edges.size)


def analyze(image: np.ndarray, config: Config | None = None) -> dict[str, float]:
    low, high = 50, 150
    if config is not None:
        low = int(config.get("analyzers.edge_density.canny_low", 50))
        high = int(config.get("analyzers.edge_density.canny_high", 150))
    return {"edge_density": edge_density(image, low=low, high=high)}
