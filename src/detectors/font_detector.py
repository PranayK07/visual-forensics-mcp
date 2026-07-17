"""Font mismatch detector.

Operates at the document level. Flags two measurable conditions:

* an unexpectedly large number of distinct font families; and
* font families used on no more than the configured share of text spans.
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
    min_confidence = float(cfg.get("min_confidence", 0.9))

    total_spans = sum(int(f.get("span_count", 0)) for f in aggregated_fonts)
    findings: list[Finding] = []
    if total_spans < min_spans or not aggregated_fonts:
        return findings

    family_count = len(aggregated_fonts)
    if family_count > max_families:
        conf = min(1.0, (family_count - max_families) / max(max_families, 1))
        conf = max(conf, 0.4)
        if conf >= min_confidence:
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
                    confidence=round(conf, 4),
                    explanation=(
                        f"Document contains {family_count} distinct font families; "
                        f"the configured comparison limit is {max_families}."
                    ),
                )
            )

    for font in aggregated_fonts:
        share = int(font.get("span_count", 0)) / total_spans
        if 0 < share <= rare_share:
            conf = min(1.0, (rare_share - share) / rare_share + 0.4)
            conf = min(conf, 1.0)
            if conf < min_confidence:
                continue
            findings.append(
                Finding(
                    type="font_mismatch",
                    page=0,
                    bbox=[0, 0, 0, 0],
                    metrics={
                        "font": font["name"],
                        "span_count": int(font.get("span_count", 0)),
                        "share": round(share, 4),
                        "rare_font_max_share": rare_share,
                        "sizes": font.get("sizes", []),
                    },
                    confidence=round(conf, 4),
                    explanation=(
                        f"Font family '{font['name']}' accounts for "
                        f"{share:.1%} of text spans; the configured maximum "
                        f"share for this finding is {rare_share:.1%}."
                    ),
                )
            )
    return findings
