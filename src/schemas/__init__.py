"""Pydantic schemas for the analysis result contract."""

from .models import (
    BBox,
    Finding,
    TileResult,
    EmbeddedImageInfo,
    FontInfo,
    PageResult,
    AnalysisSummary,
    AnalysisResult,
    BatchAnalysisResult,
)
from .statistics import (
    ClaimMetricStatistics,
    ClaimStatistics,
    DocumentStatistics,
    HistogramBin,
    MetricStatistics,
    OutlierSummary,
    StatisticsMethodology,
)

__all__ = [
    "BBox",
    "Finding",
    "TileResult",
    "EmbeddedImageInfo",
    "FontInfo",
    "PageResult",
    "AnalysisSummary",
    "AnalysisResult",
    "BatchAnalysisResult",
    "HistogramBin",
    "OutlierSummary",
    "MetricStatistics",
    "DocumentStatistics",
    "ClaimMetricStatistics",
    "StatisticsMethodology",
    "ClaimStatistics",
]
