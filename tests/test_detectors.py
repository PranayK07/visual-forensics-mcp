"""Detector unit tests using synthetic metric distributions."""

from __future__ import annotations

from src.detectors import (
    blur_detector,
    dpi_detector,
    font_detector,
    stretch_detector,
)
from src.schemas.models import TileResult
from src.utils.config import load_config


def _tile(tid: int, blur_score: float, td: float = 0.2) -> TileResult:
    return TileResult(
        tile_id=tid,
        page=1,
        bbox=[float(tid), 0.0, float(tid) + 10, 10.0],
        blank=False,
        metrics={"blur_score": blur_score, "text_density": td, "ocr_confidence": None},
    )


def test_blur_detector_flags_outlier():
    config = load_config()
    tiles = [_tile(i, 500.0) for i in range(10)]
    tiles.append(_tile(99, 5.0))  # strong low outlier
    findings = blur_detector.detect(tiles, page=1, config=config)
    assert any(f.type == "blur_anomaly" for f in findings)
    flagged = [f for f in findings if f.type == "blur_anomaly"]
    assert all(0.0 <= f.confidence <= 1.0 for f in flagged)
    assert all(len(f.bbox) == 4 for f in flagged)


def test_blur_detector_no_false_positive_uniform():
    config = load_config()
    tiles = [_tile(i, 500.0) for i in range(10)]
    findings = blur_detector.detect(tiles, page=1, config=config)
    assert findings == []


def test_stretch_detector():
    config = load_config()
    images = [
        {
            "xref": 1,
            "scale_x": 5.0,
            "scale_y": 3.0,
            "bbox": [0, 0, 200, 120],
        }
    ]
    findings = stretch_detector.detect(images, page=1, config=config)
    assert any(f.type == "image_stretch" for f in findings)


def test_dpi_detector():
    config = load_config()
    images = [
        {
            "xref": 1,
            "effective_dpi_x": 14.4,
            "effective_dpi_y": 24.0,
            "bbox": [0, 0, 200, 120],
        }
    ]
    findings = dpi_detector.detect(images, page=1, render_dpi=400.0, config=config)
    assert any(f.type == "resolution_anomaly" for f in findings)


def test_font_detector_rare_font():
    config = load_config()
    fonts = [
        {"name": "Helvetica", "span_count": 100, "sizes": [11.0]},
        {"name": "Comic Sans", "span_count": 2, "sizes": [11.0]},  # ~2% share
    ]
    findings = font_detector.detect(fonts, config)
    assert any(f.metrics.get("font") == "Comic Sans" for f in findings)
