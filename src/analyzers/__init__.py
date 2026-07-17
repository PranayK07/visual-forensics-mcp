"""Visual and structural analyzers.

Each analyzer computes a single, deterministic, measurable metric. Analyzers
only produce numbers and leave decisions to downstream consumers. Detectors
consume those numbers and flag deviations.
"""

from . import (
    blur,
    sharpness,
    contrast,
    entropy,
    edge_density,
    noise,
    ocr,
    font_analysis,
    image_scaling,
    pdf_structure,
)
from .visual_engine import compute_tile_metrics, compute_page_metrics

__all__ = [
    "blur",
    "sharpness",
    "contrast",
    "entropy",
    "edge_density",
    "noise",
    "ocr",
    "font_analysis",
    "image_scaling",
    "pdf_structure",
    "compute_tile_metrics",
    "compute_page_metrics",
]
