"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.models import AnalysisResult, BatchAnalysisResult, Finding, PageResult


def test_finding_requires_four_element_bbox():
    with pytest.raises(ValidationError):
        Finding(type="x", page=1, bbox=[0, 0, 0])  # too short


def test_finding_confidence_bounds():
    with pytest.raises(ValidationError):
        Finding(type="x", page=1, bbox=[0, 0, 1, 1], confidence=2.0)


def test_analysis_result_roundtrip():
    result = AnalysisResult(
        document_id="abc123",
        document_type="pdf",
        page_results=[PageResult(page=1, width=100, height=200, dpi=400)],
        document_findings=[
            Finding(
                type="blur_anomaly",
                page=1,
                bbox=[10, 20, 30, 40],
                metrics={"blur_score": 12.4, "ocr_confidence": 0.41},
                confidence=0.88,
                explanation="example",
            )
        ],
    )
    dumped = result.model_dump()
    # Required top-level keys must all be present.
    for key in (
        "document_id",
        "document_type",
        "summary",
        "page_results",
        "document_findings",
        "warnings",
        "errors",
    ):
        assert key in dumped

    # Re-validate the dumped dict to prove it is schema-conformant.
    reparsed = AnalysisResult.model_validate(dumped)
    assert reparsed.document_id == "abc123"
    assert reparsed.document_findings[0].metrics["blur_score"] == 12.4


def test_batch_analysis_result_roundtrip():
    single = AnalysisResult(
        document_id="abc123",
        document_type="pdf",
        page_results=[PageResult(page=1, width=100, height=200, dpi=400)],
    )
    batch = BatchAnalysisResult(results=[single])
    dumped = batch.model_dump()
    assert "results" in dumped
    assert len(dumped["results"]) == 1
    reparsed = BatchAnalysisResult.model_validate(dumped)
    assert reparsed.results[0].document_id == "abc123"
