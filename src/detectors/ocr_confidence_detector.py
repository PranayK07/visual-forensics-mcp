"""OCR confidence anomaly detector.

Flags content tiles whose mean OCR word confidence crosses the configured
low-side page-distribution comparison or absolute floor.
"""

from __future__ import annotations

from ..schemas.models import Finding, TileResult
from ..utils.config import Config
from ._common import low_side_zscore_findings


def detect(tiles: list[TileResult], page: int, config: Config) -> list[Finding]:
    cfg = config.section("detectors").get("ocr_confidence", {})
    if not cfg.get("enabled", True):
        return []
    # OCR confidence is stored 0-1; the configured floor is 0-100.
    floor = cfg.get("absolute_floor", None)
    absolute_floor = float(floor) / 100.0 if floor is not None else None
    z_threshold = float(cfg.get("z_threshold", 2.0))
    return low_side_zscore_findings(
        tiles,
        "ocr_confidence",
        page=page,
        finding_type="ocr_confidence_anomaly",
        z_threshold=z_threshold,
        text_density_min=float(cfg.get("min_text_density", 0.05)),
        min_neighbors=int(cfg.get("min_neighbors", 4)),
        absolute_floor=absolute_floor,
        explanation=(
            "Region OCR confidence met the configured low-side detector: "
            f"z-score at or below {-z_threshold:g}"
            + (
                f" or value below the absolute floor {absolute_floor:g}."
                if absolute_floor is not None
                else "."
            )
        ),
        extra_metrics=lambda _tile: {
            "z_threshold": z_threshold,
            "absolute_floor": absolute_floor,
        },
    )
