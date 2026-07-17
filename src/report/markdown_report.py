"""Self-contained Markdown reporting for claim measurement statistics."""

from __future__ import annotations

import hashlib
import html
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from ..schemas.statistics import ClaimMetricStatistics, ClaimStatistics, MetricStatistics


def _number(value: float | int | None) -> str:
    if value is None:
        return "not available"
    if isinstance(value, int):
        return f"{value:,}"
    numeric = float(value)
    if numeric == 0.0:
        numeric = 0.0
    return f"{numeric:.6g}"


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _svg_token(value: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:48] or "metric"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def _mode(statistics: MetricStatistics) -> str:
    if not statistics.modes:
        if statistics.mode_method == "none_all_values_unique":
            return "none (all finite values are unique)"
        return "not available"
    values = ", ".join(_number(value) for value in statistics.modes)
    return f"{values} (frequency {statistics.mode_frequency:,})"


def _statistics_table(statistics: MetricStatistics) -> str:
    quantiles = statistics.quantiles
    outliers = statistics.outliers
    entries: list[tuple[str, str]] = [
        ("Observation rows", _number(statistics.observation_count)),
        ("Finite numeric values", _number(statistics.finite_count)),
        ("Missing values", _number(statistics.missing_count)),
        ("Non-finite values", _number(statistics.nonfinite_count)),
        ("Non-numeric values", _number(statistics.non_numeric_count)),
        ("Unique finite values", _number(statistics.unique_count)),
        ("Zero values", _number(statistics.zero_count)),
        ("Mean", _number(statistics.mean)),
        ("Median", _number(statistics.median)),
        ("Mode", _mode(statistics)),
        ("Minimum", _number(statistics.minimum)),
        ("Maximum", _number(statistics.maximum)),
        ("Range", _number(statistics.range)),
        ("Sample variance", _number(statistics.variance)),
        ("Sample standard deviation", _number(statistics.standard_deviation)),
        ("5th percentile", _number(quantiles.get("p05"))),
        ("10th percentile", _number(quantiles.get("p10"))),
        ("25th percentile (Q1)", _number(quantiles.get("p25"))),
        ("50th percentile (Q2)", _number(quantiles.get("p50"))),
        ("75th percentile (Q3)", _number(quantiles.get("p75"))),
        ("90th percentile", _number(quantiles.get("p90"))),
        ("95th percentile", _number(quantiles.get("p95"))),
        ("Interquartile range", _number(statistics.interquartile_range)),
        ("Median absolute deviation", _number(statistics.median_absolute_deviation)),
        ("Coefficient of variation", _number(statistics.coefficient_of_variation)),
        ("Bias-corrected skewness", _number(statistics.skewness)),
        ("Excess kurtosis", _number(statistics.excess_kurtosis)),
        ("Tukey lower fence", _number(outliers.lower_fence if outliers else None)),
        ("Tukey upper fence", _number(outliers.upper_fence if outliers else None)),
        ("Values below lower fence", _number(outliers.below_lower_fence if outliers else None)),
        ("Values above upper fence", _number(outliers.above_upper_fence if outliers else None)),
        ("Values outside fences", _number(outliers.total if outliers else None)),
    ]
    rows = ["| Statistic | Value | Statistic | Value |", "|---|---:|---|---:|"]
    for index in range(0, len(entries), 2):
        left = entries[index]
        right = entries[index + 1] if index + 1 < len(entries) else ("", "")
        rows.append(
            f"| {_markdown(left[0])} | {_markdown(left[1])} | "
            f"{_markdown(right[0])} | {_markdown(right[1])} |"
        )
    return "\n".join(rows)


def _scale(value: float, minimum: float, maximum: float, left: float, width: float) -> float:
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=0.0):
        return left + width / 2.0
    return left + (value - minimum) / (maximum - minimum) * width


def _document_interval_svg(
    metric_id: str,
    claim_metric: ClaimMetricStatistics,
    document_statistics: list[tuple[str, MetricStatistics]],
) -> str:
    rows = [
        (label, statistics)
        for label, statistics in document_statistics
        if statistics.finite_count > 0
        and statistics.minimum is not None
        and statistics.maximum is not None
    ]
    if not rows:
        return "_No finite document observations are available for this graph._"

    axis_min = min(float(statistics.minimum) for _, statistics in rows if statistics.minimum is not None)
    axis_max = max(float(statistics.maximum) for _, statistics in rows if statistics.maximum is not None)
    left = 190.0
    plot_width = 500.0
    top = 32.0
    row_height = 32.0
    baseline = top + len(rows) * row_height
    height = baseline + 44.0
    unit = claim_metric.pooled.unit or "measured units"
    title = html.escape(f"Per-document intervals for {metric_id}")
    description = html.escape(
        "Each row shows minimum, first quartile, median, third quartile, and maximum "
        "on a shared scale."
    )
    svg_id = "interval-" + _svg_token(metric_id)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{svg_id}-title {svg_id}-desc" '
        f'viewBox="0 0 720 {height:.0f}" width="100%" style="max-width:760px;background:#fff">',
        f'<title id="{svg_id}-title">{title}</title>',
        f'<desc id="{svg_id}-desc">{description}</desc>',
        '<style>.lbl{font:12px system-ui,sans-serif;fill:#253142}.tick{font:11px ui-monospace,monospace;fill:#536273}'
        '.range{stroke:#536273;stroke-width:2}.box{fill:#dbeafe;stroke:#1e3a5f;stroke-width:1.5}'
        '.median{stroke:#a16207;stroke-width:2}.endpoint{fill:#fff;stroke:#1e3a5f;stroke-width:1.5}'
        '.axis{stroke:#9aa7b5;stroke-width:1}</style>',
        f'<line class="axis" x1="{left:.1f}" y1="{top - 15:.1f}" x2="{left + plot_width:.1f}" y2="{top - 15:.1f}"/>',
    ]
    for index, (label, statistics) in enumerate(rows):
        y = top + index * row_height
        minimum = float(statistics.minimum)
        maximum = float(statistics.maximum)
        q1 = float(statistics.quantiles.get("p25", minimum))
        median = float(statistics.median if statistics.median is not None else q1)
        q3 = float(statistics.quantiles.get("p75", maximum))
        x_min = _scale(minimum, axis_min, axis_max, left, plot_width)
        x_max = _scale(maximum, axis_min, axis_max, left, plot_width)
        x_q1 = _scale(q1, axis_min, axis_max, left, plot_width)
        x_median = _scale(median, axis_min, axis_max, left, plot_width)
        x_q3 = _scale(q3, axis_min, axis_max, left, plot_width)
        safe_label = html.escape(label if len(label) <= 27 else f"{label[:24]}...")
        box_width = max(1.5, x_q3 - x_q1)
        elements.extend(
            [
                f'<text class="lbl" x="4" y="{y + 4:.1f}">{safe_label}</text>',
                f'<line class="range" x1="{x_min:.2f}" y1="{y:.1f}" x2="{x_max:.2f}" y2="{y:.1f}"/>',
                f'<rect class="box" x="{x_q1:.2f}" y="{y - 8:.1f}" width="{box_width:.2f}" height="16"/>',
                f'<line class="median" x1="{x_median:.2f}" y1="{y - 9:.1f}" x2="{x_median:.2f}" y2="{y + 9:.1f}"/>',
                f'<circle class="endpoint" cx="{x_min:.2f}" cy="{y:.1f}" r="3"/>',
                f'<circle class="endpoint" cx="{x_max:.2f}" cy="{y:.1f}" r="3"/>',
            ]
        )
    elements.extend(
        [
            f'<text class="tick" x="{left:.1f}" y="{baseline + 10:.1f}" text-anchor="start">{html.escape(_number(axis_min))}</text>',
            f'<text class="tick" x="{left + plot_width:.1f}" y="{baseline + 10:.1f}" text-anchor="end">{html.escape(_number(axis_max))}</text>',
            f'<text class="tick" x="{left + plot_width / 2:.1f}" y="{baseline + 29:.1f}" text-anchor="middle">{html.escape(unit)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def _histogram_svg(statistics: MetricStatistics, document_key: str) -> str:
    bins = statistics.histogram
    if not bins:
        return "_No finite observations are available for this graph._"
    left = 52.0
    plot_width = 638.0
    top = 18.0
    plot_height = 112.0
    baseline = top + plot_height
    width_per_bin = plot_width / len(bins)
    max_count = max(item.count for item in bins) or 1
    title = html.escape(f"Histogram for {statistics.metric_id}")
    svg_id = "hist-" + _svg_token(f"{document_key}-{statistics.metric_id}")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{svg_id}-title {svg_id}-desc" '
        'viewBox="0 0 720 170" width="100%" style="max-width:760px;background:#fff">',
        f'<title id="{svg_id}-title">{title}</title>',
        f'<desc id="{svg_id}-desc">Histogram counts using the binning method recorded in the methodology section.</desc>',
        '<style>.bar{fill:#dbeafe;stroke:#1e3a5f;stroke-width:1}.axis{stroke:#9aa7b5;stroke-width:1}'
        '.tick{font:11px ui-monospace,monospace;fill:#536273}.median{stroke:#a16207;stroke-width:2}</style>',
        f'<line class="axis" x1="{left}" y1="{baseline}" x2="{left + plot_width}" y2="{baseline}"/>',
    ]
    for index, item in enumerate(bins):
        bar_height = item.count / max_count * plot_height
        x = left + index * width_per_bin + 1.0
        y = baseline - bar_height
        elements.append(
            f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{max(1.0, width_per_bin - 2.0):.2f}" '
            f'height="{bar_height:.2f}"><title>{item.count} observations from {_number(item.lower)} to {_number(item.upper)}</title></rect>'
        )
    if statistics.median is not None and statistics.minimum is not None and statistics.maximum is not None:
        median_x = _scale(
            float(statistics.median),
            float(statistics.minimum),
            float(statistics.maximum),
            left,
            plot_width,
        )
        elements.append(
            f'<line class="median" x1="{median_x:.2f}" y1="{top}" x2="{median_x:.2f}" y2="{baseline}"/>'
        )
        elements.append(
            f'<text class="tick" x="{median_x:.2f}" y="12" text-anchor="middle">median</text>'
        )
    elements.extend(
        [
            f'<text class="tick" x="{left}" y="{baseline + 18}" text-anchor="start">{html.escape(_number(statistics.minimum))}</text>',
            f'<text class="tick" x="{left + plot_width}" y="{baseline + 18}" text-anchor="end">{html.escape(_number(statistics.maximum))}</text>',
            f'<text class="tick" x="8" y="{top + 7}" text-anchor="start">n={statistics.finite_count}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def _definition_line(statistics: MetricStatistics) -> str:
    parts = [f"Observation unit: `{statistics.observation_scope}`"]
    if statistics.unit:
        parts.append(f"measurement unit: {statistics.unit}")
    if statistics.definition:
        parts.append(statistics.definition)
    return ". ".join(parts).rstrip(".") + "."


def _document_comparison_table(
    metric_id: str,
    documents: list[Any],
) -> str:
    rows = [
        "| Document | Finite n | Mean | Median | Q1 | Q3 | Minimum | Maximum | Outside Tukey fences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for document in documents:
        statistics = document.metrics.get(metric_id)
        if statistics is None:
            rows.append(f"| {_markdown(document.document_label)} | 0 | not available | not available | not available | not available | not available | not available | not available |")
            continue
        outlier_count = statistics.outliers.total if statistics.outliers else None
        rows.append(
            f"| {_markdown(document.document_label)} | {statistics.finite_count:,} | "
            f"{_number(statistics.mean)} | {_number(statistics.median)} | "
            f"{_number(statistics.quantiles.get('p25'))} | {_number(statistics.quantiles.get('p75'))} | "
            f"{_number(statistics.minimum)} | {_number(statistics.maximum)} | {_number(outlier_count)} |"
        )
    return "\n".join(rows)


def build_markdown_report(
    claim_statistics: ClaimStatistics | Mapping[str, Any],
    claim_name: str = "Claim set",
) -> str:
    """Build a standalone Markdown report with inline, dependency-free SVG."""
    claim = (
        claim_statistics
        if isinstance(claim_statistics, ClaimStatistics)
        else ClaimStatistics.model_validate(claim_statistics)
    )
    lines: list[str] = [
        f"# Measurement statistics — {_markdown(claim_name)}",
        "",
        "## Overall claim-set statistics",
        "",
        f"This section pools measurements from {claim.document_count:,} document(s). "
        "The tables report descriptive facts only; they do not establish significance, "
        "causation, intent, or authenticity. Claim-level pooled values weight each "
        "observation equally, and the per-document comparison preserves document coverage.",
        "",
    ]
    documents_by_metric: dict[str, list[tuple[str, MetricStatistics]]] = {}
    for metric_id in claim.overall_metrics:
        documents_by_metric[metric_id] = [
            (document.document_label, document.metrics[metric_id])
            for document in claim.documents
            if metric_id in document.metrics
        ]

    if not claim.overall_metrics:
        lines.extend(["No numeric analyzer measurements were available.", ""])
    for metric_id, aggregate in sorted(claim.overall_metrics.items()):
        pooled = aggregate.pooled
        lines.extend(
            [
                f"### `{metric_id}`",
                "",
                _definition_line(pooled),
                "",
                f"Document coverage: {aggregate.documents_with_observations:,} with observations; "
                f"{aggregate.documents_with_finite_values:,} with finite values; "
                f"{aggregate.documents_without_finite_values:,} without finite values.",
                "",
                _document_interval_svg(metric_id, aggregate, documents_by_metric[metric_id]),
                "",
                "The interval graph uses one common scale. Circles mark minimum and maximum, "
                "the box spans Q1 to Q3, and the gold line marks the median.",
                "",
                _document_comparison_table(metric_id, claim.documents),
                "",
                "#### Pooled observations",
                "",
                _statistics_table(pooled),
                "",
            ]
        )
        if aggregate.document_mean_statistics is not None:
            lines.extend(
                [
                    "#### Distribution of per-document means",
                    "",
                    "Each document contributes one mean to this table, independent of its observation count.",
                    "",
                    _statistics_table(aggregate.document_mean_statistics),
                    "",
                ]
            )

    for document_index, document in enumerate(claim.documents):
        lines.extend(
            [
                f"## Document — {_markdown(document.document_label)}",
                "",
                f"Document ID: `{_markdown(document.document_id)}`. Document type: `{_markdown(document.document_type)}`. "
                f"Numeric metric distributions: {len(document.metrics):,}.",
                "",
            ]
        )
        if not document.metrics:
            lines.extend(["No numeric analyzer measurements were available.", ""])
        for metric_id, statistics in sorted(document.metrics.items()):
            lines.extend(
                [
                    f"### `{metric_id}`",
                    "",
                    _definition_line(statistics),
                    "",
                    _histogram_svg(
                        statistics,
                        f"{document_index}-{document.document_id or document.document_label}",
                    ),
                    "",
                    "Histogram bar heights are observation counts; the vertical gold line marks the median.",
                    "",
                    _statistics_table(statistics),
                    "",
                ]
            )

    methodology = claim.methodology
    lines.extend(
        [
            "## Statistical definitions and boundaries",
            "",
            f"- Variance: {methodology.variance}",
            f"- Standard deviation: {methodology.standard_deviation}",
            f"- Quantiles: {methodology.quantiles}",
            f"- Mode: {methodology.mode}",
            f"- Median absolute deviation: {methodology.median_absolute_deviation}",
            f"- Coefficient of variation: {methodology.coefficient_of_variation}",
            f"- Skewness: {methodology.skewness}",
            f"- Excess kurtosis: {methodology.excess_kurtosis}",
            f"- Outlier fences: {methodology.outliers}",
            f"- Histogram bins: {methodology.histogram}",
            f"- Claim pooling: {methodology.claim_pooling}",
            "",
            "Missing counts are relative to observation rows at the named scope. A Tukey-fence "
            "count is a descriptive property of a distribution, not a classification. Statistics "
            "can be unavailable when the finite sample is too small or has zero variance.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(
    claim_statistics: ClaimStatistics | Mapping[str, Any],
    output_path: str | os.PathLike[str],
    claim_name: str = "Claim set",
) -> str:
    """Write ``report.md`` and return its absolute path."""
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_markdown_report(claim_statistics, claim_name=claim_name),
        encoding="utf-8",
    )
    return str(target)


__all__ = ["build_markdown_report", "write_markdown_report"]
