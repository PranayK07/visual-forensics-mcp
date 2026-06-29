"""Font mismatch detector.

Operates at the document level. Flags two measurable conditions:

* an unexpectedly large number of distinct font families; and
* rare fonts used on only a tiny share of text spans (a common signature of a
  single edited/inserted line that did not match the document font).
"""

from __future__ import annotations

from typing import Any

from ..schemas.models import Finding
from ..utils.config import Config


def detect(
    aggregated_fonts: list[dict[str, Any]],
    config: Config,
) -> list[Finding]:
    cfg = config.section("detectors").get("font", {})
    if not cfg.get("enabled", True):
        return []
    rare_share = float(cfg.get("rare_font_max_share", 0.05))
    max_families = int(cfg.get("max_expected_families", 4))
    min_spans = int(cfg.get("min_spans", 20))

    total_spans = sum(int(f.get("span_count", 0)) for f in aggregated_fonts)
    findings: list[Finding] = []
    if total_spans < min_spans or not aggregated_fonts:
        return findings

    family_count = len(aggregated_fonts)
    if family_count > max_families:
        conf = min(1.0, (family_count - max_families) / max(max_families, 1))
        findings.append(
            Finding(
                type="font_mismatch",
                page=0,
                bbox=[0, 0, 0, 0],
                metrics={
                    "distinct_font_families": family_count,
                    "max_expected_families": max_families,
                    "fonts": [f["name"] for f in aggregated_fonts[:20]],
                },
                confidence=round(max(conf, 0.4), 4),
                explanation=(
                    "Document uses more distinct font families than expected, "
                    "which can indicate inserted or edited content."
                ),
            )
        )

    for font in aggregated_fonts:
        share = int(font.get("span_count", 0)) / total_spans
        if 0 < share <= rare_share:
            conf = min(1.0, (rare_share - share) / rare_share + 0.4)
            findings.append(
                Finding(
                    type="font_mismatch",
                    page=0,
                    bbox=[0, 0, 0, 0],
                    metrics={
                        "font": font["name"],
                        "span_count": int(font.get("span_count", 0)),
                        "share": round(share, 4),
                        "sizes": font.get("sizes", []),
                    },
                    confidence=round(min(conf, 1.0), 4),
                    explanation=(
                        "A font family is used on only a small fraction of text "
                        "spans, an unusual change versus the dominant fonts."
                    ),
                )
            )
    return findings
