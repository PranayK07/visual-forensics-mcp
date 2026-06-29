"""Analyzer unit tests: blur, scaling, fonts, entropy, edge density, OCR."""

from __future__ import annotations

import cv2
import numpy as np

from src.analyzers import blur, edge_density, entropy, font_analysis, image_scaling
from src.analyzers.ocr import OCREngine, OCRResult
from src.render import load_document
from src.utils.config import load_config


def _text_like_image() -> np.ndarray:
    """A high-frequency image (sharp)."""
    rng = np.random.default_rng(0)
    return (rng.integers(0, 2, size=(256, 256)) * 255).astype(np.uint8)


def test_blur_computation_sharp_vs_blurred():
    sharp = _text_like_image()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4)
    sharp_score = blur.analyze(sharp)["blur_score"]
    blurred_score = blur.analyze(blurred)["blur_score"]
    assert sharp_score > blurred_score
    assert blurred_score >= 0.0


def test_image_scaling_detection():
    s = image_scaling.compute_scaling(
        original_width=40,
        original_height=40,
        display_width_pts=200.0,
        display_height_pts=120.0,
    )
    assert abs(s["scale_x"] - 5.0) < 1e-6
    assert abs(s["scale_y"] - 3.0) < 1e-6
    assert s["stretch_ratio"] > 0.1  # non-uniform
    # Effective DPI: 40 px over 200/72 inches = 14.4 dpi
    assert abs(s["effective_dpi_x"] - 14.4) < 1e-3
    assert abs(s["effective_dpi_y"] - 24.0) < 1e-3


def test_font_extraction_from_pdf(sample_pdf):
    config = load_config()
    loaded = load_document(sample_pdf, config)
    try:
        page = loaded.pdf_doc[0]
        result = font_analysis.analyze_page_fonts(page)
    finally:
        loaded.close()
    assert result["span_count"] > 0
    assert len(result["fonts"]) >= 1
    # Subset prefixes are normalised away.
    for f in result["fonts"]:
        assert "+" not in f["name"] or len(f["name"].split("+")[0]) != 6


def test_entropy_and_edge_density_ranges():
    uniform = np.full((128, 128), 128, dtype=np.uint8)
    noisy = (np.random.default_rng(1).integers(0, 256, (128, 128))).astype(np.uint8)
    assert entropy.shannon_entropy(uniform) < 0.5
    assert entropy.shannon_entropy(noisy) > 5.0

    ed_uniform = edge_density.edge_density(uniform)
    ed_noisy = edge_density.edge_density(noisy)
    assert 0.0 <= ed_uniform <= 1.0
    assert ed_noisy >= ed_uniform


def test_ocr_confidence_extraction_graceful(fast_options):
    """OCR confidence extraction must degrade gracefully without Tesseract."""
    config = load_config()
    engine = OCREngine(config)
    result = engine.analyze(_text_like_image())
    assert isinstance(result, OCRResult)
    assert 0.0 <= result.ocr_confidence <= 1.0
    if not engine.available:
        # Graceful path: no crash, zeroed metrics.
        assert result.ocr_confidence == 0.0
        assert result.word_count == 0
    else:
        # If Tesseract is installed, confidence is a normalised 0-1 value.
        assert result.word_count >= 0
