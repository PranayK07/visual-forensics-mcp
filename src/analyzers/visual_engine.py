"""Visual analysis engine.

Runs every per-tile visual analyzer and assembles a flat metrics dict for a
tile or a whole page. This is the single place that defines which metrics are
computed for each region.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from ..utils.config import Config
from ..utils.image_ops import ensure_gray, safe_round
from . import blur, contrast, edge_density, entropy, noise, sharpness
from .ocr import OCREngine


def text_density(image: np.ndarray, config: Config) -> float:
    """Foreground (ink) fraction via adaptive binarisation.

    A deterministic proxy for "how much text/content" occupies the region,
    independent of OCR availability.
    """
    gray = ensure_gray(image)
    if gray.size == 0:
        return 0.0
    block = int(config.get("analyzers.text_density.binarization_block", 35))
    c = int(config.get("analyzers.text_density.binarization_c", 10))
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    h, w = gray.shape[:2]
    if h < block or w < block:
        # Fall back to Otsu for tiles smaller than the block.
        _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            c,
        )
    return float(np.count_nonzero(binr)) / float(binr.size)


def compute_tile_metrics(
    image_gray: np.ndarray,
    image_rgb: np.ndarray,
    config: Config,
    ocr_engine: Optional[OCREngine] = None,
    blank: bool = False,
) -> dict[str, Any]:
    """Compute the full visual metric set for one tile."""
    decimals = int(config.get("output.round_decimals", 4))
    metrics: dict[str, Any] = {}

    metrics.update(blur.analyze(image_gray))
    metrics.update(sharpness.analyze(image_gray))
    metrics.update(contrast.analyze(image_gray))
    metrics.update(entropy.analyze(image_gray, config))
    metrics.update(edge_density.analyze(image_gray, config))
    metrics.update(noise.analyze(image_gray, config))
    metrics["text_density"] = text_density(image_gray, config)

    if ocr_engine is not None and ocr_engine.available and not blank:
        ocr_res = ocr_engine.analyze(image_gray)
        metrics["ocr_confidence"] = ocr_res.ocr_confidence
        metrics["ocr_word_count"] = ocr_res.word_count
        metrics["ocr_char_count"] = ocr_res.char_count
    else:
        metrics["ocr_confidence"] = None
        metrics["ocr_word_count"] = 0
        metrics["ocr_char_count"] = 0

    # Round float metrics for JSON stability; leave ints/None untouched.
    rounded: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = safe_round(value, decimals)
        else:
            rounded[key] = value
    return rounded


def compute_page_metrics(
    image_gray: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    """Whole-page visual rollup (no OCR; that is summarised from tiles)."""
    decimals = int(config.get("output.round_decimals", 4))
    metrics: dict[str, Any] = {}
    metrics.update(blur.analyze(image_gray))
    metrics.update(contrast.analyze(image_gray))
    metrics.update(entropy.analyze(image_gray, config))
    metrics.update(edge_density.analyze(image_gray, config))
    metrics["text_density"] = text_density(image_gray, config)
    return {k: (safe_round(v, decimals) if isinstance(v, float) else v) for k, v in metrics.items()}
