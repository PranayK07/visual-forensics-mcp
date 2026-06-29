"""Blur anomaly detector.

Flags tiles whose Laplacian-variance blur score is significantly lower than the
distribution of blur scores across the page's content tiles.
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
    return low_side_zscore_findings(
        tiles,
        "blur_score",
        page=page,
        finding_type="blur_anomaly",
        z_threshold=float(cfg.get("z_threshold", 2.0)),
        text_density_min=float(cfg.get("min_text_density", 0.02)),
        min_neighbors=int(cfg.get("min_neighbors", 4)),
        absolute_floor=absolute_floor,
        explanation=(
            "Region sharpness (Laplacian variance) is significantly lower than "
            "neighbouring content regions on the page."
        ),
    )
