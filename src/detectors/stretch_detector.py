"""Stretch anomaly detector.

Flags embedded images whose horizontal and vertical scale factors differ by
more than the configured non-uniform scaling threshold.
"""

from __future__ import annotations

from typing import Any

from ..schemas.models import Finding
from ..utils.config import Config


def detect(
    embedded_images: list[dict[str, Any]],
    page: int,
    config: Config,
) -> list[Finding]:
    cfg = config.section("detectors").get("stretch", {})
    if not cfg.get("enabled", True):
        return []
    ratio_threshold = float(cfg.get("ratio_threshold", 0.10))
    min_display_px = float(cfg.get("min_display_px", 16))

    findings: list[Finding] = []
    for img in embedded_images:
        sx = float(img.get("scale_x", 0.0))
        sy = float(img.get("scale_y", 0.0))
        denom = max(abs(sx), abs(sy), 1e-9)
        stretch_ratio = abs(sx - sy) / denom

        bbox = img.get("bbox", [0, 0, 0, 0])
        disp_w = abs(bbox[2] - bbox[0])
        disp_h = abs(bbox[3] - bbox[1])
        if disp_w < min_display_px or disp_h < min_display_px:
            continue

        if stretch_ratio > ratio_threshold:
            conf = max(0.0, min(1.0, stretch_ratio))
            findings.append(
                Finding(
                    type="image_stretch",
                    page=page,
                    bbox=list(bbox),
                    metrics={
                        "scale_x": round(sx, 6),
                        "scale_y": round(sy, 6),
                        "stretch_ratio": round(stretch_ratio, 4),
                        "xref": img.get("xref"),
                    },
                    confidence=round(conf, 4),
                    explanation=(
                        "Embedded image horizontal and vertical scale factors "
                        f"differ by {stretch_ratio:.1%}; the configured "
                        f"threshold is {ratio_threshold:.1%}."
                    ),
                )
            )
    return findings
