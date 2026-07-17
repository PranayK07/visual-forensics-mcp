"""Blur anomaly detector.

Flags tiles whose Laplacian-variance blur score crosses the configured low-side
comparison against the page's content-tile distribution.
"""

from __future__ import annotations

from ..schemas.models import Finding, TileResult
from ..utils.config import Config
from ._common import low_side_zscore_findings


def detect(tiles: list[TileResult], page: int, config: Config) -> list[Finding]:
    cfg = config.section("detectors").get("blur", {})
    if not cfg.get("enabled", True):
        return []
    absolute_floor = cfg.get("absolute_floor", 0.0)
    absolute_floor = float(absolute_floor) if absolute_floor else None
    z_threshold = float(cfg.get("z_threshold", 2.0))
    return low_side_zscore_findings(
        tiles,
        "blur_score",
        page=page,
        finding_type="blur_anomaly",
        z_threshold=z_threshold,
        text_density_min=float(cfg.get("min_text_density", 0.02)),
        min_neighbors=int(cfg.get("min_neighbors", 4)),
        absolute_floor=absolute_floor,
        explanation=(
            "Region Laplacian variance met the configured low-side detector: "
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
