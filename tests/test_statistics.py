"""Tests for factual metric aggregation and the standalone Markdown report."""

from __future__ import annotations

import json
import math

import pytest

from src.report.markdown_report import build_markdown_report, write_markdown_report
from src.report.statistics import (
    aggregate_claim_statistics,
    aggregate_document_statistics,
    summarize_metric,
)
from src.schemas.statistics import ClaimStatistics, DocumentStatistics, MetricStatistics


def _result(document_id: str, blur_values: list[float | None]) -> dict:
    tiles = []
    for index, value in enumerate(blur_values):
        metrics = {"blank": True} if value is None else {"blur_score": value}
        tiles.append(
            {
                "tile_id": index,
                "page": 1,
                "bbox": [0, 0, 10, 10],
                "blank": value is None,
                "metrics": metrics,
            }
        )
    return {
        "document_id": document_id,
        "document_type": "pdf",
        "summary": {"page_count": 1, "tile_count": len(tiles), "finding_count": 2},
        "page_results": [
            {
                "page": 1,
                "width": 100,
                "height": 100,
                "dpi": 200,
                "tiles": tiles,
                "page_metrics": {"blur_score": 2.0, "entropy": 3.5},
                "embedded_images": [
                    {
                        "xref": 81,
                        "original_width": 50,
                        "original_height": 25,
                        "display_width": 100.0,
                        "display_height": 50.0,
                        "scale_x": 2.0,
                        "scale_y": 2.0,
                        "effective_dpi_x": 72.0,
                        "effective_dpi_y": 72.0,
                        "bpc": 8,
                    }
                ],
                "fonts": [{"name": "Body", "span_count": 7, "sizes": [10.0, 12.0]}],
                "findings": [
                    {
                        "type": "stretch_anomaly",
                        "confidence": 0.9,
                        "metrics": {"scale_ratio": 1.4, "context": "embedded image"},
                    }
                ],
            }
        ],
        "document_findings": [
            {
                "type": "font_mismatch",
                "confidence": 0.85,
                "metrics": {"span_count": 2, "font_name": "Rare"},
            }
        ],
    }


def test_summarize_metric_reports_deep_statistics_and_invalid_counts():
    statistics = summarize_metric(
        [1.0, 2.0, 2.0, 4.0, None, math.nan, math.inf, "not numeric"],
        metric_id="tile.test",
        observation_scope="tile",
        metric_name="test",
    )

    assert statistics.observation_count == 8
    assert statistics.finite_count == 4
    assert statistics.missing_count == 1
    assert statistics.nonfinite_count == 2
    assert statistics.non_numeric_count == 1
    assert statistics.mean == pytest.approx(2.25)
    assert statistics.median == pytest.approx(2.0)
    assert statistics.modes == [2.0]
    assert statistics.mode_frequency == 2
    assert statistics.minimum == 1.0
    assert statistics.maximum == 4.0
    assert statistics.range == 3.0
    assert statistics.variance == pytest.approx(1.5833333333)
    assert statistics.standard_deviation == pytest.approx(math.sqrt(1.5833333333))
    assert statistics.quantiles["p25"] == pytest.approx(1.75)
    assert statistics.quantiles["p75"] == pytest.approx(2.5)
    assert statistics.interquartile_range == pytest.approx(0.75)
    assert statistics.median_absolute_deviation == pytest.approx(0.5)
    assert statistics.coefficient_of_variation is not None
    assert statistics.skewness is not None
    assert statistics.excess_kurtosis is not None
    assert statistics.outliers is not None
    assert statistics.outliers.total == 1
    assert sum(item.count for item in statistics.histogram) == 4

    # The model and JSON contract cannot leak NaN or infinity downstream.
    encoded = json.dumps(statistics.model_dump(mode="json"), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_mode_is_not_fabricated_for_unique_continuous_values():
    statistics = summarize_metric(
        [0.11, 0.22, 0.33],
        metric_id="tile.ratio",
        observation_scope="tile",
        metric_name="ratio",
    )
    assert statistics.modes == []
    assert statistics.mode_frequency == 0
    assert statistics.mode_method == "none_all_values_unique"


def test_constant_and_small_samples_have_explicit_unavailable_estimators():
    constant = summarize_metric(
        [5.0, 5.0, 5.0, 5.0],
        metric_id="page.constant",
        observation_scope="page",
        metric_name="constant",
    )
    assert constant.variance == 0.0
    assert constant.standard_deviation == 0.0
    assert constant.skewness is None
    assert constant.excess_kurtosis is None
    assert constant.outliers is not None and constant.outliers.total == 0
    assert len(constant.histogram) == 1

    singleton = summarize_metric(
        [7],
        metric_id="page.single",
        observation_scope="page",
        metric_name="single",
    )
    assert singleton.modes == [7.0]
    assert singleton.variance is None
    assert singleton.standard_deviation is None


def test_document_aggregation_covers_all_measurement_scopes_and_excludes_ids():
    statistics = aggregate_document_statistics(_result("doc-a", [1.0, 3.0, None]), "invoice.pdf")
    parsed = DocumentStatistics.model_validate(statistics)

    assert parsed.document_label == "invoice.pdf"
    assert parsed.metrics["document.page_count"].mean == 1.0
    assert parsed.metrics["document.tile_count"].mean == 3.0
    assert parsed.metrics["document.finding_count"].mean == 2.0
    assert parsed.metrics["page.width"].mean == 100.0
    assert parsed.metrics["page.height"].mean == 100.0
    assert parsed.metrics["page.dpi"].mean == 200.0
    assert parsed.metrics["tile.blur_score"].finite_count == 2
    assert parsed.metrics["tile.blur_score"].missing_count == 1
    assert parsed.metrics["page.entropy"].finite_count == 1
    assert parsed.metrics["embedded_image.effective_dpi_x"].mean == 72.0
    assert parsed.metrics["font.span_count"].mean == 7.0
    assert parsed.metrics["font.size"].finite_count == 2
    assert parsed.metrics["finding.stretch_anomaly.scale_ratio"].mean == 1.4
    assert parsed.metrics["finding.font_mismatch.confidence"].mean == 0.85
    assert not any("xref" in metric_id for metric_id in parsed.metrics)
    assert not any("tile_id" in metric_id for metric_id in parsed.metrics)
    assert not any(metric_id.endswith(".blank") for metric_id in parsed.metrics)


def test_claim_aggregation_reports_pooled_and_document_weighted_distributions():
    results = [_result("doc-a", [1.0, 3.0, None]), _result("doc-b", [5.0])]
    statistics = aggregate_claim_statistics(
        results,
        document_labels={"doc-a": "invoice.pdf", "doc-b": "estimate.pdf"},
    )
    claim = ClaimStatistics.model_validate(statistics)
    blur = claim.overall_metrics["tile.blur_score"]

    assert claim.document_count == 2
    assert blur.pooled.observation_count == 4
    assert blur.pooled.finite_count == 3
    assert blur.pooled.missing_count == 1
    assert blur.pooled.mean == pytest.approx(3.0)
    assert blur.documents_with_observations == 2
    assert blur.documents_with_finite_values == 2
    assert blur.documents_without_finite_values == 0
    assert blur.document_mean_statistics is not None
    assert blur.document_mean_statistics.mean == pytest.approx(3.5)
    json.dumps(statistics, allow_nan=False)


def test_claim_labels_can_follow_input_order_when_content_ids_repeat():
    statistics = aggregate_claim_statistics(
        [_result("same-hash", [1.0]), _result("same-hash", [2.0])],
        document_labels=["first-copy.pdf", "second-copy.pdf"],
    )
    assert [item["document_label"] for item in statistics["documents"]] == [
        "first-copy.pdf",
        "second-copy.pdf",
    ]


def test_markdown_report_is_overall_first_self_contained_and_factual(tmp_path):
    statistics = aggregate_claim_statistics(
        [_result("doc-a", [1.0, 3.0]), _result("doc-b", [5.0])],
        document_labels={"doc-a": "invoice.pdf", "doc-b": "estimate.pdf"},
    )
    report = build_markdown_report(statistics, claim_name="claim_001")

    overall_position = report.index("## Overall claim-set statistics")
    document_position = report.index("## Document —")
    assert overall_position < document_position
    assert "`tile.blur_score`" in report
    assert "Mean" in report
    assert "Median" in report
    assert "Mode" in report
    assert "Sample variance" in report
    assert "Bias-corrected skewness" in report
    assert "<svg" in report
    assert "Per-document intervals" in report
    assert "Histogram" in report
    assert "fraud" not in report.lower()
    assert "risk" not in report.lower()
    assert "recommend" not in report.lower()

    path = write_markdown_report(statistics, tmp_path / "claim_001_result" / "report.md", "claim_001")
    assert path.endswith("report.md")
    assert (tmp_path / "claim_001_result" / "report.md").read_text(encoding="utf-8") == report


def test_statistics_models_reject_negative_counts():
    with pytest.raises(ValueError):
        MetricStatistics(
            metric_id="tile.x",
            observation_scope="tile",
            metric_name="x",
            observation_count=-1,
            finite_count=0,
            missing_count=0,
            nonfinite_count=0,
            non_numeric_count=0,
            unique_count=0,
        )
