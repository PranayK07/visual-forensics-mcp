"""Deterministic descriptive statistics for analyzer measurements.

This module converts the existing analysis JSON into a factual statistics
handoff.  It deliberately reports distributions and estimator metadata only;
it does not compute a risk score, significance label, or document verdict.
"""

from __future__ import annotations

import math
import numbers
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schemas.statistics import (
    ClaimMetricStatistics,
    ClaimStatistics,
    DocumentStatistics,
    HistogramBin,
    MetricStatistics,
    OutlierSummary,
)


_MISSING = object()
_IDENTIFIER_KEYS = {"xref", "xref_a", "xref_b", "tile_id", "page"}


@dataclass
class _Series:
    metric_id: str
    observation_scope: str
    metric_name: str
    values: list[Any]


_METRIC_METADATA: dict[str, tuple[str, str]] = {
    "blur_score": ("squared intensity-gradient units", "Variance of the Laplacian."),
    "sharpness_tenengrad": (
        "squared intensity-gradient units",
        "Mean squared Sobel gradient magnitude (Tenengrad focus measure).",
    ),
    "sharpness_mean_gradient": (
        "intensity-gradient units",
        "Mean Sobel gradient magnitude.",
    ),
    "contrast_rms": ("grayscale intensity levels", "Standard deviation of grayscale intensity."),
    "contrast_michelson": ("ratio", "(maximum - minimum) / (maximum + minimum) intensity."),
    "entropy": ("bits", "Shannon entropy of the grayscale intensity histogram."),
    "edge_density": ("ratio", "Fraction of pixels classified as Canny edges."),
    "noise_sigma": ("grayscale intensity levels", "Laplacian-convolution noise estimate."),
    "noise_median_residual": (
        "grayscale intensity levels",
        "Standard deviation after subtracting a median-filtered image.",
    ),
    "text_density": ("ratio", "Fraction of pixels classified as foreground content."),
    "ocr_confidence": ("ratio", "Mean accepted OCR word confidence on a 0-to-1 scale."),
    "ocr_word_count": ("words", "Accepted OCR word count."),
    "ocr_char_count": ("characters", "Accepted OCR character count."),
    "original_width": ("pixels", "Native embedded-image width."),
    "original_height": ("pixels", "Native embedded-image height."),
    "display_width": ("PDF points", "Embedded-image displayed width on the page."),
    "display_height": ("PDF points", "Embedded-image displayed height on the page."),
    "scale_x": ("PDF points per source pixel", "Horizontal display-to-native image scale."),
    "scale_y": ("PDF points per source pixel", "Vertical display-to-native image scale."),
    "effective_dpi_x": ("dots per inch", "Horizontal effective resolution at displayed size."),
    "effective_dpi_y": ("dots per inch", "Vertical effective resolution at displayed size."),
    "bpc": ("bits per component", "Embedded-image bits per color component."),
    "span_count": ("text spans", "Number of text spans recorded for a font."),
    "size": ("points", "Observed font size."),
    "confidence": ("ratio", "Detector confidence on a 0-to-1 scale."),
    "width": ("pixels", "Rendered page width."),
    "height": ("pixels", "Rendered page height."),
    "dpi": ("dots per inch", "Rendered page resolution."),
    "page_count": ("pages", "Number of rendered document pages."),
    "tile_count": ("tiles", "Number of generated analysis tiles."),
    "finding_count": ("findings", "Number of emitted measured-deviation findings."),
}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _is_numeric(value: Any) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, numbers.Real)


def _metric_metadata(metric_name: str) -> tuple[str | None, str | None]:
    return _METRIC_METADATA.get(metric_name, (None, None))


def _records_to_series(
    records: Sequence[Mapping[str, Any]],
    scope: str,
) -> dict[str, _Series]:
    """Turn same-grain records into series while retaining missing values."""
    keys = sorted({key for record in records for key in record})
    output: dict[str, _Series] = {}
    for key in keys:
        values = [record.get(key, _MISSING) for record in records]
        # Textual labels and booleans are metadata, not measured distributions.
        if not any(_is_numeric(value) for value in values):
            continue
        metric_id = f"{scope}.{key}"
        output[metric_id] = _Series(metric_id, scope, key, values)
    return output


def _flatten_metric_mapping(
    mapping: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Flatten scalar finding metrics; arrays remain contextual metadata."""
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        if str(key).lower() in _IDENTIFIER_KEYS:
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_metric_mapping(value, name))
        elif not isinstance(value, (list, tuple, dict)):
            flattened[name] = value
    return flattened


def _extract_series(result: Mapping[str, Any] | Any) -> dict[str, _Series]:
    data = _mapping(result)
    pages = [_mapping(page) for page in data.get("page_results", [])]
    output: dict[str, _Series] = {}

    document_records: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []
    page_records: list[dict[str, Any]] = []
    embedded_records: list[dict[str, Any]] = []
    font_records: list[dict[str, Any]] = []
    font_size_records: list[dict[str, Any]] = []
    finding_records: dict[str, list[dict[str, Any]]] = {}

    embedded_fields = (
        "original_width",
        "original_height",
        "display_width",
        "display_height",
        "scale_x",
        "scale_y",
        "effective_dpi_x",
        "effective_dpi_y",
        "bpc",
    )

    document_records.append(_mapping(data.get("summary", {})))
    for page in pages:
        page_record = {
            key: page.get(key) for key in ("width", "height", "dpi")
        }
        page_record.update(_mapping(page.get("page_metrics", {})))
        page_records.append(page_record)
        for tile in page.get("tiles", []):
            tile_records.append(_mapping(_mapping(tile).get("metrics", {})))
        for embedded in page.get("embedded_images", []):
            image = _mapping(embedded)
            embedded_records.append({key: image.get(key) for key in embedded_fields})
        for font_value in page.get("fonts", []):
            font = _mapping(font_value)
            font_records.append({"span_count": font.get("span_count")})
            for size in font.get("sizes", []) or []:
                font_size_records.append({"size": size})
        for finding in page.get("findings", []):
            item = _mapping(finding)
            finding_type = str(item.get("type") or "unknown")
            record = {"confidence": item.get("confidence")}
            record.update(_flatten_metric_mapping(_mapping(item.get("metrics", {}))))
            finding_records.setdefault(finding_type, []).append(record)

    for finding in data.get("document_findings", []):
        item = _mapping(finding)
        finding_type = str(item.get("type") or "unknown")
        record = {"confidence": item.get("confidence")}
        record.update(_flatten_metric_mapping(_mapping(item.get("metrics", {}))))
        finding_records.setdefault(finding_type, []).append(record)

    for records, scope in (
        (document_records, "document"),
        (tile_records, "tile"),
        (page_records, "page"),
        (embedded_records, "embedded_image"),
        (font_records, "font"),
        (font_size_records, "font"),
    ):
        output.update(_records_to_series(records, scope))

    for finding_type, records in sorted(finding_records.items()):
        output.update(_records_to_series(records, f"finding.{finding_type}"))
    return output


def _histogram(values: np.ndarray, q1: float, q3: float) -> list[HistogramBin]:
    if values.size == 0:
        return []
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if values.size == 1 or math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=0.0):
        return [
            HistogramBin(
                lower=minimum,
                upper=maximum,
                count=int(values.size),
                upper_inclusive=True,
            )
        ]

    iqr = q3 - q1
    if iqr > 0:
        width = 2.0 * iqr / math.pow(float(values.size), 1.0 / 3.0)
        bin_count = math.ceil((maximum - minimum) / width) if width > 0 else 1
    else:
        bin_count = math.ceil(math.log2(float(values.size)) + 1.0)
    bin_count = max(1, min(20, int(bin_count)))
    counts, edges = np.histogram(values, bins=bin_count, range=(minimum, maximum))
    return [
        HistogramBin(
            lower=float(edges[index]),
            upper=float(edges[index + 1]),
            count=int(count),
            upper_inclusive=index == len(counts) - 1,
        )
        for index, count in enumerate(counts)
    ]


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def summarize_metric(
    values: Sequence[Any],
    *,
    metric_id: str,
    observation_scope: str,
    metric_name: str,
) -> MetricStatistics:
    """Summarize one metric without ever serializing NaN or infinity."""
    finite: list[float] = []
    missing_count = 0
    nonfinite_count = 0
    non_numeric_count = 0
    for value in values:
        if value is _MISSING or value is None:
            missing_count += 1
        elif not _is_numeric(value):
            non_numeric_count += 1
        else:
            numeric = float(value)
            if math.isfinite(numeric):
                finite.append(numeric)
            else:
                nonfinite_count += 1

    unit, definition = _metric_metadata(metric_name)
    base: dict[str, Any] = {
        "metric_id": metric_id,
        "observation_scope": observation_scope,
        "metric_name": metric_name,
        "unit": unit,
        "definition": definition,
        "observation_count": len(values),
        "finite_count": len(finite),
        "missing_count": missing_count,
        "nonfinite_count": nonfinite_count,
        "non_numeric_count": non_numeric_count,
        "unique_count": len(set(finite)),
    }
    if not finite:
        return MetricStatistics(**base)

    array = np.asarray(finite, dtype=np.float64)
    n = int(array.size)
    quantile_points = {
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
    }
    quantiles = {
        name: float(np.quantile(array, probability, method="linear"))
        for name, probability in quantile_points.items()
    }
    mean = float(np.mean(array))
    median = quantiles["p50"]
    q1 = quantiles["p25"]
    q3 = quantiles["p75"]
    iqr = q3 - q1
    minimum = float(np.min(array))
    maximum = float(np.max(array))

    counts = Counter(finite)
    max_frequency = max(counts.values())
    if n == 1:
        modes = [finite[0]]
        mode_method = "single_observation"
    elif max_frequency == 1:
        modes = []
        mode_method = "none_all_values_unique"
    else:
        modes = sorted(value for value, frequency in counts.items() if frequency == max_frequency)
        mode_method = "exact_repeated_value"

    variance: float | None = None
    stddev: float | None = None
    coefficient_of_variation: float | None = None
    skewness: float | None = None
    excess_kurtosis: float | None = None
    if n >= 2:
        variance = _finite_or_none(np.var(array, ddof=1))
        stddev = _finite_or_none(np.std(array, ddof=1))
        zero_tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(array))))
        if stddev is not None and abs(mean) > zero_tolerance:
            coefficient_of_variation = _finite_or_none(stddev / abs(mean))

    deviations = array - mean
    second_moment = float(np.mean(deviations**2))
    if second_moment > 0 and n >= 3:
        third_moment = float(np.mean(deviations**3))
        g1 = third_moment / math.pow(second_moment, 1.5)
        skewness = _finite_or_none(math.sqrt(n * (n - 1)) / (n - 2) * g1)
    if second_moment > 0 and n >= 4:
        fourth_moment = float(np.mean(deviations**4))
        g2 = fourth_moment / (second_moment**2) - 3.0
        excess_kurtosis = _finite_or_none(
            ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
        )

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    below_count = int(np.count_nonzero(array < lower_fence))
    above_count = int(np.count_nonzero(array > upper_fence))

    return MetricStatistics(
        **base,
        mean=mean,
        median=median,
        modes=modes,
        mode_frequency=max_frequency if modes else 0,
        mode_method=mode_method,
        minimum=minimum,
        maximum=maximum,
        range=maximum - minimum,
        variance=variance,
        standard_deviation=stddev,
        quantiles=quantiles,
        interquartile_range=iqr,
        median_absolute_deviation=float(np.median(np.abs(array - median))),
        coefficient_of_variation=coefficient_of_variation,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        zero_count=int(np.count_nonzero(array == 0.0)),
        outliers=OutlierSummary(
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            below_lower_fence=below_count,
            above_upper_fence=above_count,
            total=below_count + above_count,
        ),
        histogram=_histogram(array, q1, q3),
    )


def _document_statistics_model(
    result: Mapping[str, Any] | Any,
    document_label: str | None = None,
) -> tuple[DocumentStatistics, dict[str, _Series]]:
    data = _mapping(result)
    document_id = str(data.get("document_id") or "")
    label = document_label or str(
        data.get("document_label")
        or data.get("document_name")
        or data.get("source_filename")
        or data.get("filename")
        or document_id
        or "unknown document"
    )
    series = _extract_series(data)
    metrics = {
        metric_id: summarize_metric(
            item.values,
            metric_id=item.metric_id,
            observation_scope=item.observation_scope,
            metric_name=item.metric_name,
        )
        for metric_id, item in sorted(series.items())
    }
    return (
        DocumentStatistics(
            document_id=document_id,
            document_label=label,
            document_type=str(data.get("document_type") or "unknown"),
            metrics=metrics,
        ),
        series,
    )


def aggregate_document_statistics(
    result: Mapping[str, Any] | Any,
    document_label: str | None = None,
) -> dict[str, Any]:
    """Return JSON-safe descriptive statistics for one analysis result."""
    model, _ = _document_statistics_model(result, document_label)
    return model.model_dump(mode="json")


def _normalise_results(results: Sequence[Any] | Mapping[str, Any] | Any) -> list[Any]:
    if hasattr(results, "model_dump"):
        results = results.model_dump()
    if isinstance(results, Mapping):
        nested = results.get("results")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return list(nested)
        return [results]
    return list(results)


def aggregate_claim_statistics(
    results: Sequence[Any] | Mapping[str, Any] | Any,
    document_labels: Mapping[str | int, str] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return document and pooled claim statistics as a JSON-safe dictionary."""
    raw_results = _normalise_results(results)
    document_models: list[DocumentStatistics] = []
    document_series: list[dict[str, _Series]] = []
    for index, result in enumerate(raw_results):
        data = _mapping(result)
        document_id = str(data.get("document_id") or "")
        label = None
        if document_labels:
            if isinstance(document_labels, Mapping):
                label = document_labels.get(document_id)
                if label is None:
                    label = document_labels.get(index) or document_labels.get(str(index))
            elif index < len(document_labels):
                label = document_labels[index]
        model, series = _document_statistics_model(data, label)
        document_models.append(model)
        document_series.append(series)

    metric_ids = sorted({key for series in document_series for key in series})
    overall_metrics: dict[str, ClaimMetricStatistics] = {}
    for metric_id in metric_ids:
        available = [series[metric_id] for series in document_series if metric_id in series]
        template = available[0]
        pooled_values = [value for item in available for value in item.values]
        pooled = summarize_metric(
            pooled_values,
            metric_id=metric_id,
            observation_scope=template.observation_scope,
            metric_name=template.metric_name,
        )
        document_metrics = [document.metrics.get(metric_id) for document in document_models]
        finite_documents = [metric for metric in document_metrics if metric and metric.finite_count > 0]
        document_means = [metric.mean for metric in finite_documents if metric.mean is not None]
        mean_distribution = None
        if document_means:
            mean_distribution = summarize_metric(
                document_means,
                metric_id=f"document_mean.{metric_id}",
                observation_scope="document",
                metric_name=template.metric_name,
            )
        overall_metrics[metric_id] = ClaimMetricStatistics(
            pooled=pooled,
            document_count=len(document_models),
            documents_with_observations=len(available),
            documents_with_finite_values=len(finite_documents),
            documents_without_finite_values=len(document_models) - len(finite_documents),
            document_mean_statistics=mean_distribution,
        )

    claim = ClaimStatistics(
        document_count=len(document_models),
        overall_metrics=overall_metrics,
        documents=document_models,
    )
    return claim.model_dump(mode="json")


__all__ = [
    "aggregate_claim_statistics",
    "aggregate_document_statistics",
    "summarize_metric",
]
