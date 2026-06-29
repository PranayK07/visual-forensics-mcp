"""Resolution / effective-DPI anomaly detector.

Flags embedded images whose effective DPI is far below the page render DPI or
below an absolute floor -- a low-resolution insert placed into a high-resolution
document.
"""

from __future__ import annotations

from typing import Any

from ..schemas.models import Finding
from ..utils.config import Config


def detect(
    embedded_images: list[dict[str, Any]],
    page: int,
    render_dpi: float,
    config: Config,
) -> list[Finding]:
    cfg = config.section("detectors").get("dpi", {})
    if not cfg.get("enabled", True):
        return []
    ratio_threshold = float(cfg.get("ratio_threshold", 0.5))
    absolute_min = float(cfg.get("absolute_min_dpi", 150.0))

    findings: list[Finding] = []
    for img in embedded_images:
        eff_x = float(img.get("effective_dpi_x", 0.0))
        eff_y = float(img.get("effective_dpi_y", 0.0))
        eff = min(eff_x, eff_y)
        if eff <= 0:
            continue
        ratio = eff / render_dpi if render_dpi > 0 else 1.0
        below_ratio = ratio < ratio_threshold
        below_floor = eff < absolute_min
        if below_ratio or below_floor:
            # Confidence grows as effective DPI drops relative to the floor.
            conf = max(0.0, min(1.0, 1.0 - (eff / max(absolute_min, 1.0))))
            conf = max(conf, 0.5 if below_floor else 0.4)
            findings.append(
                Finding(
                    type="resolution_anomaly",
                    page=page,
                    bbox=list(img.get("bbox", [0, 0, 0, 0])),
                    metrics={
                        "effective_dpi_x": round(eff_x, 2),
                        "effective_dpi_y": round(eff_y, 2),
                        "render_dpi": round(float(render_dpi), 2),
                        "dpi_ratio": round(ratio, 4),
                        "xref": img.get("xref"),
                    },
                    confidence=round(conf, 4),
                    explanation=(
                        "Embedded image effective resolution is significantly "
                        "lower than the surrounding document resolution."
                    ),
                )
            )
    return findings
