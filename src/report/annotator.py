"""Annotate a copy of a PDF with its forensic findings.

Draws an accurate bounding box around every located finding, labels it with the
finding type, confidence, and the specific metrics that triggered it, and prints
an overall fraud-risk badge (HIGH / MEDIUM / LOW) in the top-right of each page.
A summary cover page lists document-level findings (e.g. font mismatches) that
have no single location, plus a colour legend.

Coordinate handling
-------------------
Finding bounding boxes are produced in *rendered pixel* coordinates at the
page's render DPI. PyMuPDF draws in *PDF points* (1/72 inch). We convert with::

    points = pixels * 72 / render_dpi

so boxes land exactly on the element that produced them. Pages whose rendered
size does not match the source page (e.g. rotated pages) are detected and the
per-axis scale is derived from the actual page rectangle as a fallback, keeping
boxes aligned.
"""

from __future__ import annotations

import os
from typing import Any

import fitz  # PyMuPDF

from ..utils.config import Config, load_config
from ..utils.logging import get_logger

logger = get_logger("annotator")

# RGB colours (0-1) per finding type.
TYPE_COLORS: dict[str, tuple[float, float, float]] = {
    "image_stretch": (1.00, 0.55, 0.00),            # orange
    "resolution_anomaly": (0.85, 0.10, 0.10),       # red
    "raster_in_vector_anomaly": (0.60, 0.20, 0.80), # purple
    "blur_anomaly": (0.00, 0.50, 1.00),             # blue
    "compression_artifact_anomaly": (0.00, 0.70, 0.60),  # teal
    "ocr_confidence_anomaly": (0.90, 0.35, 0.60),   # pink
    "object_overlap_anomaly": (0.55, 0.35, 0.10),   # brown
    "font_mismatch": (0.40, 0.40, 0.40),            # gray (document-level)
}
DEFAULT_COLOR = (0.20, 0.20, 0.20)

RISK_COLORS = {
    "HIGH": (0.80, 0.10, 0.10),
    "MEDIUM": (0.95, 0.60, 0.10),
    "LOW": (0.15, 0.60, 0.25),
}

# Human-readable, short explanation of *what* each metric indicates.
_METRIC_LABELS: dict[str, str] = {
    "blur_score": "blur (Laplacian var)",
    "zscore": "z-score vs page",
    "ocr_confidence": "OCR confidence",
    "effective_dpi_x": "eff. DPI x",
    "effective_dpi_y": "eff. DPI y",
    "dpi_ratio": "DPI ratio",
    "scale_x": "scale x",
    "scale_y": "scale y",
    "stretch_ratio": "stretch ratio",
    "noise_median_residual": "noise (residual)",
    "raster_coverage": "raster coverage",
    "iou": "overlap IoU",
    "share": "usage share",
    "span_count": "spans",
    "distinct_font_families": "font families",
    "font": "font",
}

# Which metrics to surface (in order) per finding type.
_TYPE_METRICS: dict[str, list[str]] = {
    "blur_anomaly": ["blur_score", "zscore"],
    "ocr_confidence_anomaly": ["ocr_confidence", "zscore"],
    "resolution_anomaly": ["effective_dpi_x", "effective_dpi_y", "dpi_ratio"],
    "image_stretch": ["scale_x", "scale_y", "stretch_ratio"],
    "compression_artifact_anomaly": ["noise_median_residual", "zscore"],
    "raster_in_vector_anomaly": ["raster_coverage", "span_count"],
    "object_overlap_anomaly": ["iou"],
    "font_mismatch": ["font", "share", "distinct_font_families", "span_count"],
}


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _metric_lines(finding: dict[str, Any]) -> list[str]:
    """Build short 'what is off' lines for a finding."""
    metrics = finding.get("metrics", {})
    keys = _TYPE_METRICS.get(finding["type"], list(metrics.keys()))
    lines: list[str] = []
    for key in keys:
        if key in metrics and metrics[key] is not None:
            label = _METRIC_LABELS.get(key, key)
            lines.append(f"{label}: {_fmt_value(metrics[key])}")
    return lines


def _pretty_type(t: str) -> str:
    return t.replace("_", " ").title()


def compute_fraud_risk(result: dict[str, Any], config: Config) -> dict[str, Any]:
    """Return overall fraud risk as {level, score, factors}.

    Transparent, deterministic blend of the strongest finding, the number of
    distinct anomaly types, and the total finding count. All weights/thresholds
    come from config (report.risk).
    """
    risk_cfg = config.get("report.risk", {}) or {}
    w_conf = float(risk_cfg.get("w_max_confidence", 0.6))
    w_types = float(risk_cfg.get("w_distinct_types", 0.08))
    w_count = float(risk_cfg.get("w_finding_count", 0.02))
    high_t = float(risk_cfg.get("high_threshold", 66))
    med_t = float(risk_cfg.get("medium_threshold", 33))

    findings: list[dict[str, Any]] = []
    for page in result.get("page_results", []):
        findings.extend(page.get("findings", []))
    findings.extend(result.get("document_findings", []))

    if not findings:
        return {"level": "LOW", "score": 0, "factors": {
            "max_confidence": 0.0, "distinct_types": 0, "finding_count": 0}}

    max_conf = max(float(f.get("confidence", 0.0)) for f in findings)
    distinct_types = len({f.get("type") for f in findings})
    count = len(findings)

    score01 = w_conf * max_conf + w_types * distinct_types + w_count * count
    score01 = max(0.0, min(1.0, score01))
    score = round(score01 * 100)

    if score >= high_t:
        level = "HIGH"
    elif score >= med_t:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "score": score,
        "factors": {
            "max_confidence": round(max_conf, 4),
            "distinct_types": distinct_types,
            "finding_count": count,
        },
    }


def _page_scale(page: "fitz.Page", page_result: dict[str, Any]) -> tuple[float, float]:
    """Pixels -> points scale factors for x and y, robust to size mismatches."""
    rect = page.rect
    width_px = float(page_result.get("width", 0)) or 1.0
    height_px = float(page_result.get("height", 0)) or 1.0
    dpi = float(page_result.get("dpi", 0)) or 0.0

    if dpi > 0:
        sx = sy = 72.0 / dpi
        # Sanity check against the actual page rectangle; if rendered pixel size
        # disagrees with rect*scale by >2%, fall back to a direct ratio.
        if abs(width_px * sx - rect.width) > 0.02 * max(rect.width, 1):
            sx = rect.width / width_px
        if abs(height_px * sy - rect.height) > 0.02 * max(rect.height, 1):
            sy = rect.height / height_px
        return sx, sy
    return rect.width / width_px, rect.height / height_px


def _draw_label_block(
    page: "fitz.Page",
    anchor: tuple[float, float],
    header: str,
    body: list[str],
    color: tuple[float, float, float],
    fontsize: float,
) -> None:
    fontname = "helv"
    pad = 3.0
    line_h = fontsize + 2.5
    all_lines = [header] + body
    max_w = max(
        fitz.get_text_length(ln, fontname=fontname, fontsize=fontsize)
        for ln in all_lines
    )
    box_w = max_w + 2 * pad
    box_h = line_h * len(all_lines) + 2 * pad

    x, y = anchor
    x = max(2.0, min(x, page.rect.width - box_w - 2.0))
    y = max(2.0, min(y, page.rect.height - box_h - 2.0))

    # Header bar.
    header_rect = fitz.Rect(x, y, x + box_w, y + line_h + pad)
    page.draw_rect(header_rect, color=color, fill=color, fill_opacity=0.9, width=0)
    # Body background (light).
    body_rect = fitz.Rect(x, y + line_h + pad, x + box_w, y + box_h)
    page.draw_rect(body_rect, color=color, fill=(1, 1, 1), fill_opacity=0.85, width=0.5)

    page.insert_text(
        (x + pad, y + pad + fontsize),
        header,
        fontname=fontname,
        fontsize=fontsize,
        color=(1, 1, 1),
    )
    for i, ln in enumerate(body):
        page.insert_text(
            (x + pad, y + line_h + pad + fontsize + i * line_h),
            ln,
            fontname=fontname,
            fontsize=fontsize,
            color=(0.1, 0.1, 0.1),
        )


def _draw_risk_badge(page: "fitz.Page", risk: dict[str, Any], fontsize: float) -> None:
    level = risk["level"]
    score = risk["score"]
    color = RISK_COLORS.get(level, DEFAULT_COLOR)
    text = f"FRAUD RISK: {level}"
    sub = f"score {score}/100"

    w = max(
        fitz.get_text_length(text, fontname="helv", fontsize=fontsize),
        fitz.get_text_length(sub, fontname="helv", fontsize=fontsize - 2),
    ) + 16
    h = fontsize * 2 + 16
    x1 = page.rect.width - w - 12
    y0 = 12
    rect = fitz.Rect(x1, y0, x1 + w, y0 + h)
    page.draw_rect(rect, color=color, fill=color, fill_opacity=0.95, width=0)
    page.insert_text(
        (x1 + 8, y0 + fontsize + 2),
        text,
        fontname="hebo",
        fontsize=fontsize,
        color=(1, 1, 1),
    )
    page.insert_text(
        (x1 + 8, y0 + fontsize * 2 + 4),
        sub,
        fontname="helv",
        fontsize=fontsize - 2,
        color=(1, 1, 1),
    )


def _group_findings(findings: list[dict[str, Any]]) -> dict[tuple, list[dict]]:
    """Group findings that share (rounded) bbox so labels don't overlap."""
    groups: dict[tuple, list[dict]] = {}
    for f in findings:
        bbox = f.get("bbox", [0, 0, 0, 0])
        if not bbox or len(bbox) != 4 or (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            continue
        key = tuple(round(v, 1) for v in bbox)
        groups.setdefault(key, []).append(f)
    return groups


def _add_cover_page(
    doc: "fitz.Document",
    result: dict[str, Any],
    risk: dict[str, Any],
    badge_fontsize: float,
) -> None:
    first = doc[0]
    width, height = first.rect.width, first.rect.height
    page = doc.new_page(pno=0, width=width, height=height)

    margin = 40
    y = margin
    page.insert_text((margin, y), "Visual Document Forensics — Annotated Report",
                     fontname="hebo", fontsize=18, color=(0.1, 0.1, 0.1))
    y += 10
    _draw_risk_badge(page, risk, badge_fontsize + 2)

    y += 30
    page.insert_text((margin, y), f"Document ID: {result.get('document_id', '')}",
                     fontname="helv", fontsize=10, color=(0.2, 0.2, 0.2))
    y += 16
    page.insert_text((margin, y), f"Type: {result.get('document_type', '')}    "
                     f"Pages: {result.get('summary', {}).get('page_count', 0)}    "
                     f"Findings: {result.get('summary', {}).get('finding_count', 0)}",
                     fontname="helv", fontsize=10, color=(0.2, 0.2, 0.2))
    y += 16
    factors = risk.get("factors", {})
    page.insert_text(
        (margin, y),
        f"Risk basis: max confidence {factors.get('max_confidence')}, "
        f"{factors.get('distinct_types')} anomaly type(s), "
        f"{factors.get('finding_count')} finding(s).",
        fontname="helv", fontsize=10, color=(0.2, 0.2, 0.2),
    )

    # Findings by type.
    y += 28
    page.insert_text((margin, y), "Findings by type", fontname="hebo",
                     fontsize=12, color=(0.1, 0.1, 0.1))
    y += 18
    by_type = result.get("summary", {}).get("findings_by_type", {})
    for t, c in by_type.items():
        col = TYPE_COLORS.get(t, DEFAULT_COLOR)
        swatch = fitz.Rect(margin, y - 8, margin + 12, y + 2)
        page.draw_rect(swatch, color=col, fill=col, width=0)
        page.insert_text((margin + 18, y), f"{_pretty_type(t)}: {c}",
                         fontname="helv", fontsize=10, color=(0.15, 0.15, 0.15))
        y += 16

    # Document-level findings (no location).
    doc_findings = result.get("document_findings", [])
    if doc_findings:
        y += 14
        page.insert_text((margin, y), "Document-level findings (no single location)",
                         fontname="hebo", fontsize=12, color=(0.1, 0.1, 0.1))
        y += 18
        for f in doc_findings:
            header = f"{_pretty_type(f['type'])}  (confidence {f.get('confidence')})"
            page.insert_text((margin, y), header, fontname="helv", fontsize=10,
                             color=(0.15, 0.15, 0.15))
            y += 14
            for ln in _metric_lines(f):
                page.insert_text((margin + 16, y), ln, fontname="helv",
                                 fontsize=9, color=(0.35, 0.35, 0.35))
                y += 12
            expl = f.get("explanation", "")
            if expl:
                page.insert_textbox(
                    fitz.Rect(margin + 16, y, width - margin, y + 28),
                    expl, fontname="helv", fontsize=8, color=(0.45, 0.45, 0.45),
                )
                y += 26
            y += 6
            if y > height - margin:
                break

    page.insert_text(
        (margin, height - margin),
        "Boxes are measured visual/structural evidence, not proof of fraud. "
        "Interpret alongside business context.",
        fontname="helv", fontsize=8, color=(0.5, 0.5, 0.5),
    )


def annotate_document(
    input_pdf_path: str,
    result: dict[str, Any],
    output_path: str | None = None,
    config: Config | None = None,
) -> str:
    """Create an annotated copy of ``input_pdf_path`` and return its path."""
    if config is None:
        config = load_config()

    draw_cfg = config.get("report.draw", {}) or {}
    box_width = float(draw_cfg.get("box_width", 2.0))
    label_fs = float(draw_cfg.get("label_fontsize", 8.0))
    badge_fs = float(draw_cfg.get("badge_fontsize", 12.0))
    min_conf = float(draw_cfg.get("min_confidence_to_draw", 0.0))
    add_cover = bool(draw_cfg.get("add_cover_page", True))

    if output_path is None:
        base, ext = os.path.splitext(input_pdf_path)
        output_path = f"{base} - annotated{ext or '.pdf'}"

    doc = fitz.open(input_pdf_path)
    risk = compute_fraud_risk(result, config)

    page_results = {p["page"]: p for p in result.get("page_results", [])}

    # Annotate located findings page by page (original page indices).
    for page_no, page_result in page_results.items():
        idx = page_no - 1
        if idx < 0 or idx >= doc.page_count:
            continue
        page = doc[idx]
        sx, sy = _page_scale(page, page_result)

        located = [
            f for f in page_result.get("findings", [])
            if float(f.get("confidence", 0.0)) >= min_conf
        ]
        groups = _group_findings(located)

        for bbox_key, group in groups.items():
            x1, y1, x2, y2 = bbox_key
            rect = fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            # Strongest finding drives the box colour.
            top = max(group, key=lambda f: float(f.get("confidence", 0.0)))
            color = TYPE_COLORS.get(top["type"], DEFAULT_COLOR)
            page.draw_rect(rect, color=color, width=box_width)

            header = f"{_pretty_type(top['type'])}  c={top.get('confidence')}"
            body: list[str] = []
            for f in group:
                if f is not top:
                    body.append(f"+ {_pretty_type(f['type'])} c={f.get('confidence')}")
            for ln in _metric_lines(top):
                body.append(ln)

            anchor = (rect.x0, max(2.0, rect.y0 - (label_fs + 6) * (len(body) + 1)))
            _draw_label_block(page, anchor, header, body, color, label_fs)

        _draw_risk_badge(page, risk, badge_fs)

    if add_cover and doc.page_count > 0:
        _add_cover_page(doc, result, risk, badge_fs)

    doc.save(output_path, garbage=3, deflate=True)
    doc.close()
    logger.info("Annotated report written to %s", output_path)
    return output_path
