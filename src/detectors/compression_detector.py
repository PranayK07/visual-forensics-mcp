"""Compression-artifact anomaly detector.

Flags tiles whose noise signature deviates strongly (either direction) from the
page's content tiles. Localised spikes or drops in residual noise often mark a
spliced-in region that was compressed differently from its surroundings.
"""

from __future__ import annotations

from ..schemas.models import Finding, TileResult
from ..utils.config import Config
from ._common import two_sided_zscore_findings


def detect(tiles: list[TileResult], page: int, config: Config) -> list[Finding]:
    cfg = config.section("detectors").get("compression", {})
    if not cfg.get("enabled", True):
        return []
    return two_sided_zscore_findings(
        tiles,
        "noise_median_residual",
        page=page,
        finding_type="compression_artifact_anomaly",
        z_threshold=float(cfg.get("z_threshold", 2.5)),
        text_density_min=float(cfg.get("min_text_density", 0.02)),
        min_neighbors=int(cfg.get("min_neighbors", 4)),
        explanation=(
            "Region noise/compression signature differs significantly from "
            "neighbouring regions, consistent with separate compression."
        ),
    )
