"""Anomaly detectors.

Detectors translate analyzer measurements into :class:`Finding` objects. They
report measurable deviations only -- never fraud judgements. Every threshold is
read from configuration.
"""

from . import (
    blur_detector,
    dpi_detector,
    font_detector,
    stretch_detector,
    ocr_confidence_detector,
    compression_detector,
    structure_detector,
)

__all__ = [
    "blur_detector",
    "dpi_detector",
    "font_detector",
    "stretch_detector",
    "ocr_confidence_detector",
    "compression_detector",
    "structure_detector",
]
