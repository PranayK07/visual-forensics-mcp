"""Schemas for descriptive measurement statistics and claim-level rollups.

The statistics contract intentionally contains measurements and estimator
metadata only.  It does not assign risk, significance, intent, or a verdict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistogramBin(BaseModel):
    """One deterministic histogram interval."""

    lower: float
    upper: float
    count: int = Field(ge=0)
    lower_inclusive: bool = True
    upper_inclusive: bool = False


class OutlierSummary(BaseModel):
    """Observed counts outside Tukey's 1.5-IQR fences."""

    method: Literal["tukey_1_5_iqr"] = "tukey_1_5_iqr"
    lower_fence: float
    upper_fence: float
    below_lower_fence: int = Field(ge=0)
    above_upper_fence: int = Field(ge=0)
    total: int = Field(ge=0)


class MetricStatistics(BaseModel):
    """Descriptive statistics for one source-qualified numeric metric."""

    metric_id: str = Field(description="Source-qualified name, such as tile.blur_score.")
    observation_scope: str = Field(description="The observation grain, such as tile or page.")
    metric_name: str
    unit: str | None = None
    definition: str | None = None

    observation_count: int = Field(ge=0)
    finite_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    nonfinite_count: int = Field(ge=0)
    non_numeric_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)

    mean: float | None = None
    median: float | None = None
    modes: list[float] = Field(default_factory=list)
    mode_frequency: int = Field(0, ge=0)
    mode_method: Literal[
        "exact_repeated_value", "single_observation", "none_all_values_unique", "unavailable"
    ] = "unavailable"

    minimum: float | None = None
    maximum: float | None = None
    range: float | None = None
    variance: float | None = None
    standard_deviation: float | None = None
    quantiles: dict[str, float] = Field(default_factory=dict)
    interquartile_range: float | None = None
    median_absolute_deviation: float | None = None
    coefficient_of_variation: float | None = None
    skewness: float | None = None
    excess_kurtosis: float | None = None
    zero_count: int = Field(0, ge=0)
    outliers: OutlierSummary | None = None
    histogram: list[HistogramBin] = Field(default_factory=list)


class DocumentStatistics(BaseModel):
    """Every available metric distribution for one analyzed document."""

    document_id: str
    document_label: str
    document_type: str = "unknown"
    metrics: dict[str, MetricStatistics] = Field(default_factory=dict)


class ClaimMetricStatistics(BaseModel):
    """A pooled metric plus explicit document-level coverage and variability."""

    pooled: MetricStatistics
    document_count: int = Field(ge=0)
    documents_with_observations: int = Field(ge=0)
    documents_with_finite_values: int = Field(ge=0)
    documents_without_finite_values: int = Field(ge=0)
    pooling_method: Literal["pooled_observations"] = "pooled_observations"
    document_mean_statistics: MetricStatistics | None = None


class StatisticsMethodology(BaseModel):
    """Machine-readable definitions for the estimators in this payload."""

    variance: str = "Sample variance with denominator n - 1; unavailable when n < 2."
    standard_deviation: str = "Square root of sample variance."
    quantiles: str = "Linear interpolation at p05, p10, p25, p50, p75, p90, and p95."
    mode: str = (
        "Exact repeated finite values only; all-unique samples report no mode instead "
        "of imposing an arbitrary histogram binning rule."
    )
    median_absolute_deviation: str = (
        "Median of absolute deviations from the sample median; no consistency scaling."
    )
    coefficient_of_variation: str = (
        "Sample standard deviation divided by the absolute mean; unavailable at a "
        "numerically zero mean."
    )
    skewness: str = (
        "Bias-corrected Fisher-Pearson sample skewness; unavailable when n < 3 or "
        "variance is zero."
    )
    excess_kurtosis: str = (
        "Unbiased Fisher excess kurtosis; unavailable when n < 4 or variance is zero."
    )
    outliers: str = "Observed values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR."
    histogram: str = (
        "Freedman-Diaconis bin width when IQR is positive, otherwise Sturges' rule; "
        "bin count is limited to 20."
    )
    claim_pooling: str = (
        "Claim statistics pool finite observations. Documents with more observations "
        "therefore contribute more; document_mean_statistics separately describes "
        "the distribution of per-document means."
    )


class ClaimStatistics(BaseModel):
    """Complete factual statistics handoff for one claim set."""

    schema_version: Literal["1.0"] = "1.0"
    document_count: int = Field(ge=0)
    overall_metrics: dict[str, ClaimMetricStatistics] = Field(default_factory=dict)
    documents: list[DocumentStatistics] = Field(default_factory=list)
    methodology: StatisticsMethodology = Field(default_factory=StatisticsMethodology)

