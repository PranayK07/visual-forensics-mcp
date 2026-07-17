"""Pydantic models that define the deterministic JSON contract.

These models are the single source of truth for the shape of the data returned
by the ``analyze_document`` MCP tool. The top-level :class:`BatchAnalysisResult`
wraps one :class:`AnalysisResult` per input document::

    {
      "results": [
        {
          "document_id": "...",
          "document_type": "...",
          "summary": {},
          "page_results": [],
          "document_findings": [],
          "warnings": [],
          "errors": []
        }
      ]
    }
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .statistics import ClaimStatistics, DocumentStatistics

# A bbox is always [x1, y1, x2, y2] in the coordinate space named by its owner.
BBox = list[float]


class Finding(BaseModel):
    """A single measurable anomaly.

    Findings report a measurable deviation together with the metrics that
    produced it, leaving interpretation to the downstream agent.
    """

    type: str = Field(..., description="Anomaly type, e.g. 'blur_anomaly'.")
    page: int = Field(..., description="1-based page number (0 for document-level).")
    bbox: BBox = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0],
        description="[x1, y1, x2, y2] in rendered pixel coordinates.",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Raw measurements supporting the finding."
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Strength of the measured deviation (0-1)."
    )
    explanation: str = Field("", description="Plain-language description of the deviation.")

    @field_validator("bbox")
    @classmethod
    def _bbox_len(cls, v: BBox) -> BBox:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 elements [x1, y1, x2, y2]")
        return [float(x) for x in v]


class TileResult(BaseModel):
    """Per-tile visual metrics."""

    tile_id: int
    page: int
    bbox: BBox
    blank: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def _bbox_len(cls, v: BBox) -> BBox:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 elements [x1, y1, x2, y2]")
        return [float(x) for x in v]


class EmbeddedImageInfo(BaseModel):
    """A raster image placed on a page (PDF structure analysis)."""

    xref: int
    original_width: int
    original_height: int
    display_width: float
    display_height: float
    bbox: BBox
    scale_x: float
    scale_y: float
    effective_dpi_x: float
    effective_dpi_y: float
    colorspace: Optional[str] = None
    bpc: Optional[int] = None
    image_filter: Optional[str] = None


class FontInfo(BaseModel):
    """A font family observed in the document and how often it appears."""

    name: str
    span_count: int
    sizes: list[float] = Field(default_factory=list)


class PageResult(BaseModel):
    """All analysis output for a single rendered page."""

    page: int
    width: int
    height: int
    dpi: float
    is_vector: bool = False
    tiles: list[TileResult] = Field(default_factory=list)
    page_metrics: dict[str, Any] = Field(default_factory=dict)
    embedded_images: list[EmbeddedImageInfo] = Field(default_factory=list)
    fonts: list[FontInfo] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class AnalysisSummary(BaseModel):
    """High-level rollup across the whole document."""

    page_count: int = 0
    tile_count: int = 0
    finding_count: int = 0
    findings_by_type: dict[str, int] = Field(default_factory=dict)
    ocr_available: bool = False
    pdf_structure_available: bool = False
    config_digest: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Per-document analysis output."""

    document_id: str
    document_name: str = Field(
        "", description="Source filename only; no directory path is exposed."
    )
    document_type: str
    summary: AnalysisSummary = Field(default_factory=AnalysisSummary)
    page_results: list[PageResult] = Field(default_factory=list)
    document_findings: list[Finding] = Field(default_factory=list)
    statistics: DocumentStatistics | None = Field(
        default=None,
        description="Descriptive distributions for every numeric metric measured.",
    )
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BatchAnalysisResult(BaseModel):
    """Top-level response returned by ``analyze_document``."""

    schema_version: Literal["2.0"] = "2.0"
    statistics: ClaimStatistics | None = Field(
        default=None,
        description="Pooled claim-set statistics and per-document distributions.",
    )
    results: list[AnalysisResult] = Field(default_factory=list)
