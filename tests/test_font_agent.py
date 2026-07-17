"""Tests for the font-outlier detector, the font banner, and the font agent."""

from __future__ import annotations

import fitz  # PyMuPDF

import font_agent
from src.analyzers import font_analysis
from src.detectors import font_outlier_detector
from src.report.annotator import annotate_document, collect_font_summary
from src.tools.analyze import analyze_document
from src.utils.config import load_config


def _span(font: str, x: float, y: float, text: str = "abc", size: float = 11.0):
    return {
        "font": font,
        "size": size,
        "bbox": [x, y, x + 50.0, y + 12.0],
        "text": text,
    }


# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------

def test_font_outlier_flags_rare_font_with_location():
    config = load_config()
    aggregated = [
        {"name": "Helvetica", "span_count": 95, "sizes": [11.0]},
        {"name": "Courier", "span_count": 5, "sizes": [11.0]},
    ]
    spans = [_span("Helvetica", 0, i * 20.0) for i in range(10)]
    spans.append(_span("Courier", 100, 300.0, text="inserted line"))

    findings = font_outlier_detector.detect(spans, aggregated, 1, 1.0, config)
    assert len(findings) == 1
    f = findings[0]
    assert f.type == "font_outlier"
    assert f.page == 1
    assert f.bbox == [100.0, 300.0, 150.0, 312.0]
    assert f.metrics["font"] == "Courier"
    assert f.metrics["dominant_font"] == "Helvetica"
    assert f.metrics["text"] == "inserted line"
    assert f.confidence == 0.95  # 95 / (95 + 5)


def test_font_outlier_ignores_dominant_font():
    config = load_config()
    aggregated = [{"name": "Helvetica", "span_count": 100, "sizes": [11.0]}]
    spans = [_span("Helvetica", 0, i * 20.0) for i in range(10)]
    assert font_outlier_detector.detect(spans, aggregated, 1, 1.0, config) == []


def test_font_outlier_merges_adjacent_spans():
    config = load_config()
    aggregated = [
        {"name": "Helvetica", "span_count": 90, "sizes": [11.0]},
        {"name": "Courier", "span_count": 10, "sizes": [11.0]},
    ]
    spans = [_span("Helvetica", 0, i * 100.0) for i in range(9)]
    # Two Courier spans on the same line, close together -> one region.
    spans.append(_span("Courier", 100, 300.0, text="first"))
    spans.append(_span("Courier", 155, 300.0, text="second"))
    findings = font_outlier_detector.detect(spans, aggregated, 1, 1.0, config)
    assert len(findings) == 1
    assert findings[0].metrics["span_count"] == 2
    assert "first" in findings[0].metrics["text"]
    assert "second" in findings[0].metrics["text"]


def test_font_outlier_respects_min_doc_spans():
    config = load_config(overrides={
        "detectors": {"font_outlier": {"min_doc_spans": 50}}
    })
    aggregated = [
        {"name": "Helvetica", "span_count": 8, "sizes": [11.0]},
        {"name": "Courier", "span_count": 2, "sizes": [11.0]},
    ]
    spans = [_span("Courier", 0, 0)]
    assert font_outlier_detector.detect(spans, aggregated, 1, 1.0, config) == []


def test_font_outlier_requires_configured_confidence():
    config = load_config(overrides={
        "detectors": {"font_outlier": {"min_confidence": 0.9}}
    })
    aggregated = [
        {"name": "Helvetica", "span_count": 80, "sizes": [11.0]},
        {"name": "Courier", "span_count": 20, "sizes": [11.0]},
    ]
    spans = [_span("Courier", 0, 0)]
    assert font_outlier_detector.detect(spans, aggregated, 1, 1.0, config) == []


# ---------------------------------------------------------------------------
# Span extraction
# ---------------------------------------------------------------------------

def test_font_name_normalisation_groups_style_variants():
    normalise = font_analysis._normalise_font_name
    assert normalise("ABCDEF+TimesNewRomanPS-BoldMT") == "TimesNewRoman"
    assert normalise("TimesNewRomanPSMT") == "TimesNewRoman"
    assert normalise("Times New Roman Bold") == "Times New Roman"
    assert normalise("Times New Roman") == "Times New Roman"
    assert normalise("Arial-BoldItalicMT") == "Arial"
    assert normalise("Alegreya-Regular") == "Alegreya"
    assert normalise("Alegreya-Sans") == "Alegreya-Sans"

def test_extract_font_spans_scales_to_pixels(sample_pdf):
    with fitz.open(sample_pdf) as doc:
        spans_pts = font_analysis.extract_font_spans(doc[0], 1.0)
        spans_px = font_analysis.extract_font_spans(doc[0], 2.0)
    assert spans_pts and len(spans_pts) == len(spans_px)
    for a, b in zip(spans_pts, spans_px):
        assert all(abs(2 * va - vb) < 1e-6 for va, vb in zip(a["bbox"], b["bbox"]))
    assert any(s["font"] != spans_pts[0]["font"] for s in spans_pts)


# ---------------------------------------------------------------------------
# End-to-end: pipeline emits located font_outlier findings
# ---------------------------------------------------------------------------

def test_pipeline_emits_font_outlier(sample_pdf, fast_options):
    result = analyze_document(sample_pdf, fast_options)["results"][0]
    outliers = [
        f
        for page in result["page_results"]
        for f in page["findings"]
        if f["type"] == "font_outlier"
    ]
    assert outliers, "expected the rare-font line to be boxed"
    f = outliers[0]
    x1, y1, x2, y2 = f["bbox"]
    assert x2 > x1 and y2 > y1  # a real location, not a placeholder box
    assert f["metrics"]["dominant_font"] == "Helvetica"
    assert f["metrics"]["font"] != "Helvetica"
    assert 0.0 < f["confidence"] <= 1.0

    summary = collect_font_summary(result)
    assert summary["dominant"]["name"] == "Helvetica"
    assert any(o["confidence"] for o in summary["others"])


# ---------------------------------------------------------------------------
# Agent CLI + annotated output
# ---------------------------------------------------------------------------

def test_font_agent_annotates_pdf(sample_pdf, tmp_path, capsys):
    out_pdf = str(tmp_path / "fonts annotated.pdf")
    import sys

    argv = sys.argv
    sys.argv = ["font_agent.py", sample_pdf, "--out", out_pdf]
    try:
        font_agent.main()
    finally:
        sys.argv = argv

    console = capsys.readouterr().out
    assert "Dominant font : Helvetica" in console
    assert "Boxed regions :" in console

    with fitz.open(out_pdf) as doc:
        assert doc.page_count == 1
        page_text = doc[0].get_text()
        assert "DOMINANT FONT: Helvetica" in page_text
        assert "Observed font:" in page_text

def test_annotate_without_banner(sample_pdf, fast_options, tmp_path):
    result = analyze_document(sample_pdf, fast_options)["results"][0]
    config = load_config(overrides={
        "report": {"draw": {"add_font_banner": False}}
    })
    out = annotate_document(sample_pdf, result, str(tmp_path / "plain.pdf"), config)
    with fitz.open(out) as doc:
        text = "".join(p.get_text() for p in doc)
    assert "DOMINANT FONT" not in text
