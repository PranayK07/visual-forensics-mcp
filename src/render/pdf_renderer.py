"""Page rendering via PyMuPDF.

Renders every page of a fitz document to a deterministic raster at a
configurable DPI (default 400). Works for any PyMuPDF-openable document,
including PDF and DOCX, so the downstream pipeline is identical for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import fitz  # PyMuPDF

from ..utils.config import Config
from ..utils.image_ops import ensure_gray
from ..utils.logging import get_logger

logger = get_logger("renderer")


@dataclass
class RenderedPage:
    """A single rendered page raster plus its provenance."""

    page_number: int  # 1-based
    image_gray: np.ndarray  # 2D uint8 grayscale used for analysis
    image_rgb: np.ndarray  # HxWx3 uint8, used for OCR / colour metrics
    width: int  # pixels
    height: int  # pixels
    dpi: float
    pdf_width: float  # page width in PDF points (1/72 inch)
    pdf_height: float  # page height in PDF points


def _matrix_for_dpi(dpi: float) -> "fitz.Matrix":
    zoom = dpi / 72.0
    return fitz.Matrix(zoom, zoom)


def render_document(doc: "fitz.Document", config: Config) -> list[RenderedPage]:
    """Render the document pages described by ``config`` into rasters."""
    dpi = float(config.get("render.dpi", 400))
    max_pages = int(config.get("render.max_pages", 0) or 0)
    max_pixels = int(config.get("render.max_pixels", 40_000_000) or 0)

    page_count = doc.page_count
    if max_pages > 0:
        page_count = min(page_count, max_pages)

    matrix = _matrix_for_dpi(dpi)
    pages: list[RenderedPage] = []

    for index in range(page_count):
        page = doc[index]
        rect = page.rect

        effective_dpi = dpi
        if max_pixels > 0:
            est_pixels = (rect.width / 72.0 * dpi) * (rect.height / 72.0 * dpi)
            if est_pixels > max_pixels:
                scale = (max_pixels / est_pixels) ** 0.5
                effective_dpi = max(72.0, dpi * scale)
                matrix = _matrix_for_dpi(effective_dpi)
                logger.warning(
                    "Page %d exceeds max_pixels; reducing DPI to %.1f",
                    index + 1,
                    effective_dpi,
                )
            else:
                matrix = _matrix_for_dpi(dpi)
                effective_dpi = dpi

        pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            rgb = rgb[:, :, :3]
        rgb = np.ascontiguousarray(rgb)
        gray = ensure_gray(rgb)

        pages.append(
            RenderedPage(
                page_number=index + 1,
                image_gray=gray,
                image_rgb=rgb,
                width=pix.width,
                height=pix.height,
                dpi=float(effective_dpi),
                pdf_width=float(rect.width),
                pdf_height=float(rect.height),
            )
        )

    logger.info("Rendered %d page(s) at ~%.0f DPI", len(pages), dpi)
    return pages
