"""Font outlier detector: locate text that is not in the document's dominant font.

The document's most common font family (by span count) is treated as the
dominant font. Every span set in any other family is an outlier candidate. To
avoid one box per word, adjacent spans of the same family are merged into
regions (same line, then vertically adjacent lines).

Confidence is a deterministic function of how rare the font is relative to the
dominant one::

    confidence = dominant_span_count / (dominant_span_count + font_span_count)

so a single inserted line in a foreign font scores near 1.0, while a 50/50
split between two fonts scores 0.5 for each region.
"""

from __future__ import annotations

from typing import Any

from ..schemas.models import Finding
from ..utils.config import Config

_TEXT_SNIPPET_LEN = 60


def _merge_regions(
    spans: list[dict[str, Any]],
    gap_px: float,
) -> list[dict[str, Any]]:
    """Merge same-font spans that sit on the same line or on adjacent lines.

    Two spans merge when their boxes, expanded by ``gap_px``, intersect. The
    result keeps the union bbox, concatenated text, all sizes, and span count.
    """
    regions: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda s: (s["bbox"][1], s["bbox"][0])):
        x1, y1, x2, y2 = span["bbox"]
        merged = False
        for region in regions:
            rx1, ry1, rx2, ry2 = region["bbox"]
            if (
                x1 <= rx2 + gap_px
                and x2 >= rx1 - gap_px
                and y1 <= ry2 + gap_px
                and y2 >= ry1 - gap_px
            ):
                region["bbox"] = [
                    min(rx1, x1), min(ry1, y1), max(rx2, x2), max(ry2, y2)
                ]
                region["text"] += " " + span["text"]
                region["sizes"].add(span["size"])
                region["span_count"] += 1
                merged = True
                break
        if not merged:
            regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "text": span["text"],
                    "sizes": {span["size"]},
                    "span_count": 1,
                }
            )
    return regions


def detect(
    page_spans: list[dict[str, Any]],
    aggregated_fonts: list[dict[str, Any]],
    page: int,
    pts_to_px: float,
    config: Config,
) -> list[Finding]:
    """Return one located finding per non-dominant-font region on ``page``.

    ``page_spans`` comes from :func:`font_analysis.extract_font_spans` (pixel
    bboxes); ``aggregated_fonts`` is the document-level font list, ordered by
    span count, from :func:`font_analysis.aggregate_fonts`.
    """
    cfg = config.section("detectors").get("font_outlier", {})
    if not cfg.get("enabled", True):
        return []
    min_doc_spans = int(cfg.get("min_doc_spans", 10))
    max_share = float(cfg.get("max_share", 0.5))
    merge_gap_pts = float(cfg.get("merge_gap_pts", 9.0))
    max_regions = int(cfg.get("max_regions_per_page", 40))
    ignore = {str(n) for n in cfg.get("ignore_fonts", [])}

    if len(aggregated_fonts) < 2 or not page_spans:
        return []
    total_spans = sum(int(f.get("span_count", 0)) for f in aggregated_fonts)
    if total_spans < min_doc_spans:
        return []

    dominant = aggregated_fonts[0]
    dominant_name = dominant["name"]
    dominant_count = int(dominant.get("span_count", 0))
    doc_counts = {
        f["name"]: int(f.get("span_count", 0)) for f in aggregated_fonts
    }

    # Group this page's outlier spans by font family.
    by_font: dict[str, list[dict[str, Any]]] = {}
    for span in page_spans:
        name = span["font"]
        if name == dominant_name or name in ignore:
            continue
        doc_count = doc_counts.get(name, 0)
        if total_spans and doc_count / total_spans > max_share:
            continue  # co-dominant family, not an outlier
        by_font.setdefault(name, []).append(span)

    findings: list[Finding] = []
    gap_px = merge_gap_pts * pts_to_px
    for name, spans in by_font.items():
        doc_count = doc_counts.get(name, len(spans))
        share = doc_count / total_spans if total_spans else 0.0
        denom = dominant_count + doc_count
        confidence = dominant_count / denom if denom else 0.0
        for region in _merge_regions(spans, gap_px):
            text = region["text"].strip()
            if len(text) > _TEXT_SNIPPET_LEN:
                text = text[: _TEXT_SNIPPET_LEN - 3] + "..."
            findings.append(
                Finding(
                    type="font_outlier",
                    page=page,
                    bbox=list(region["bbox"]),
                    metrics={
                        "font": name,
                        "dominant_font": dominant_name,
                        "share": round(share, 4),
                        "span_count": region["span_count"],
                        "sizes": sorted(region["sizes"]),
                        "text": text,
                    },
                    confidence=round(confidence, 4),
                    explanation=(
                        f"Text set in '{name}' while the document's dominant "
                        f"font is '{dominant_name}'; this font covers only "
                        f"{share:.1%} of all text spans."
                    ),
                )
            )
    findings.sort(key=lambda f: (-f.confidence, f.bbox[1], f.bbox[0]))
    return findings[:max_regions]
