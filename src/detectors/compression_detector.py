"""Compression-artifact anomaly detector.

Flags tiles whose noise signature deviates strongly (in either direction) from
the distribution measured across the page's content tiles.
"""

from __future__ import annotations

from ..schemas.models import Finding, TileResult
from ..utils.config import Config
from ._common import two_sided_zscore_findings


def detect(tiles: list[TileResult], page: int, config: Config) -> list[Finding]:
    cfg = config.section("detectors").get("compression", {})
    if not cfg.get("enabled", True):
        return []
    z_threshold = float(cfg.get("z_threshold", 2.5))
    return two_sided_zscore_findings(
        tiles,
        "noise_median_residual",
        page=page,
        finding_type="compression_artifact_anomaly",
        z_threshold=z_threshold,
        text_density_min=float(cfg.get("min_text_density", 0.02)),
        min_neighbors=int(cfg.get("min_neighbors", 4)),
        explanation=(
            "Region median-residual noise has an absolute page-distribution "
            f"z-score at or above the configured threshold {z_threshold:g}."
        ),
    )
