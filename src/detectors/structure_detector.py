"""Structure anomaly detector.

Flags two structural conditions per page:

* a raster region embedded inside an otherwise vector page (text/vector page
  with a localised image insert); and
* strongly overlapping / stacked images (unusual layering).
"""

from __future__ import annotations

from typing import Any

from ..schemas.models import Finding
from ..utils.config import Config


def detect(
    page_structure: dict[str, Any],
    page: int,
    config: Config,
) -> list[Finding]:
    cfg = config.section("detectors").get("structure", {})
    if not cfg.get("enabled", True):
        return []
    cov_min = float(cfg.get("raster_coverage_min", 0.02))
    cov_max = float(cfg.get("raster_coverage_max", 0.85))
    overlap_iou = float(cfg.get("overlap_iou_threshold", 0.6))

    findings: list[Finding] = []
    coverage = float(page_structure.get("raster_coverage", 0.0))
    is_vector = bool(page_structure.get("is_vector", False))
    span_count = int(page_structure.get("span_count", 0))
    embedded = page_structure.get("embedded_images", [])

    # Raster region inside an otherwise vector/text page.
    if is_vector and span_count > 0 and cov_min <= coverage <= cov_max and embedded:
        for img in embedded:
            findings.append(
                Finding(
                    type="raster_in_vector_anomaly",
                    page=page,
                    bbox=list(img.get("bbox", [0, 0, 0, 0])),
                    metrics={
                        "raster_coverage": round(coverage, 4),
                        "span_count": span_count,
                        "image_count": page_structure.get("image_count", 0),
                        "xref": img.get("xref"),
                    },
                    confidence=round(min(1.0, 0.4 + coverage), 4),
                    explanation=(
                        "A raster image region is embedded inside an otherwise "
                        "vector/text page, which can indicate a pasted insert."
                    ),
                )
            )

    # Overlapping / stacked images.
    for ov in page_structure.get("overlaps", []):
        if float(ov.get("iou", 0.0)) >= overlap_iou:
            findings.append(
                Finding(
                    type="object_overlap_anomaly",
                    page=page,
                    bbox=list(ov.get("bbox", [0, 0, 0, 0])),
                    metrics={
                        "iou": ov.get("iou"),
                        "xref_a": ov.get("xref_a"),
                        "xref_b": ov.get("xref_b"),
                    },
                    confidence=round(min(1.0, float(ov.get("iou", 0.0))), 4),
                    explanation=(
                        "Two embedded images strongly overlap (unusual layering "
                        "/ stacking of raster objects)."
                    ),
                )
            )
    return findings
