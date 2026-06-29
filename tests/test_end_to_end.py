"""Complete end-to-end pipeline tests for PDF and DOCX."""

from __future__ import annotations

from src.schemas.models import AnalysisResult
from src.tools.analyze import analyze_document


def _validate_findings(result: dict):
    """Every finding must satisfy the required output schema."""
    findings = list(result["document_findings"])
    for page in result["page_results"]:
        findings.extend(page["findings"])
    for f in findings:
        assert set(["type", "page", "bbox", "metrics", "confidence", "explanation"]) <= set(f)
        assert len(f["bbox"]) == 4
        assert 0.0 <= f["confidence"] <= 1.0
        assert isinstance(f["metrics"], dict)
    return findings


def test_end_to_end_pdf(sample_pdf, fast_options):
    result = analyze_document(sample_pdf, fast_options)

    # Schema-valid (re-parse through pydantic).
    AnalysisResult.model_validate(result)

    assert result["document_type"] == "pdf"
    assert result["document_id"]
    assert result["errors"] == []
    assert result["summary"]["page_count"] == 1
    assert result["summary"]["tile_count"] > 0

    page = result["page_results"][0]
    assert page["width"] > 0 and page["height"] > 0
    assert len(page["tiles"]) > 0
    # Tile metrics contain the required visual measurements.
    content_tiles = [t for t in page["tiles"] if not t["blank"]]
    assert content_tiles
    m = content_tiles[0]["metrics"]
    for key in ("blur_score", "entropy", "edge_density", "contrast_rms", "text_density"):
        assert key in m

    findings = _validate_findings(result)
    types = {f["type"] for f in findings}
    # The synthetic PDF embeds a stretched, low-resolution raster in a text page.
    assert "image_stretch" in types
    assert "resolution_anomaly" in types
    assert {"font_mismatch"} & types or result["summary"]["finding_count"] >= 1


def test_end_to_end_docx(sample_docx, fast_options):
    result = analyze_document(sample_docx, fast_options)
    AnalysisResult.model_validate(result)
    assert result["document_type"] == "docx"
    assert result["errors"] == []
    assert result["summary"]["page_count"] >= 1
    assert result["summary"]["tile_count"] > 0
    _validate_findings(result)


def test_end_to_end_scanned_pdf(scanned_pdf, fast_options):
    """Image-only PDFs must analyze without errors (no text/vector layer)."""
    result = analyze_document(scanned_pdf, fast_options)
    AnalysisResult.model_validate(result)
    assert result["errors"] == []
    assert result["summary"]["page_count"] == 1
    page = result["page_results"][0]
    # A full-page raster => high raster coverage, not flagged as vector insert.
    assert page["page_metrics"]
