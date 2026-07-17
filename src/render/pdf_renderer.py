"""Page rendering via PyMuPDF.

Renders every page of an input to a deterministic analysis raster. PDF and
DOCX pages use the configured DPI (default 400). Submitted raster images keep
their native pixels so the renderer does not introduce interpolation artifacts
that could affect visual-forensics metrics. Multi-frame TIFFs render one frame
per page.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageOps

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


def _positive_dpi(value: object) -> tuple[float, float] | None:
    """Normalize Pillow's scalar/tuple DPI metadata, rejecting bad values."""
    try:
        if isinstance(value, (tuple, list)):
            x = float(value[0])
            y = float(value[1] if len(value) > 1 else value[0])
        else:
            x = y = float(value)
        if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y):
            return x, y
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _rgb_on_white(frame: "Image.Image") -> "Image.Image":
    """Convert a Pillow frame to RGB, compositing transparency onto white."""
    has_alpha = frame.mode in {"RGBA", "LA"} or (
        frame.mode == "P" and "transparency" in frame.info
    )
    if not has_alpha:
        return frame.convert("RGB")
    rgba = frame.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def _render_raster_document(doc: object, config: Config) -> list[RenderedPage]:
    """Render a Pillow-backed raster source without upsampling its pixels."""
    fallback_dpi = float(config.get("render.dpi", 400))
    max_pages = int(config.get("render.max_pages", 0) or 0)
    max_pixels = int(config.get("render.max_pixels", 40_000_000) or 0)
    page_count = int(doc.page_count)
    if max_pages > 0:
        page_count = min(page_count, max_pages)

    pages: list[RenderedPage] = []
    for index in range(page_count):
        frame, info = doc.get_frame(index)
        try:
            # Apply camera/scanner orientation metadata before analysis. This is
            # a lossless transpose/rotation, not an interpolating resize.
            frame = ImageOps.exif_transpose(frame)
            rgb_image = _rgb_on_white(frame)
            source_width, source_height = rgb_image.size

            dpi_pair = _positive_dpi(info.get("dpi"))
            dpi_x, dpi_y = dpi_pair or (fallback_dpi, fallback_dpi)

            effective_scale = 1.0
            if max_pixels > 0 and source_width * source_height > max_pixels:
                effective_scale = math.sqrt(
                    max_pixels / float(source_width * source_height)
                )
                target = (
                    max(1, int(round(source_width * effective_scale))),
                    max(1, int(round(source_height * effective_scale))),
                )
                rgb_image = rgb_image.resize(target, resample=Image.Resampling.LANCZOS)
                logger.warning(
                    "Raster page %d exceeds max_pixels; reducing dimensions "
                    "from %dx%d to %dx%d",
                    index + 1,
                    source_width,
                    source_height,
                    target[0],
                    target[1],
                )

            rgb = np.ascontiguousarray(np.asarray(rgb_image, dtype=np.uint8))
            gray = ensure_gray(rgb)
            width, height = rgb_image.size
            effective_dpi_x = dpi_x * effective_scale
            effective_dpi_y = dpi_y * effective_scale
            # PageResult currently exposes one DPI value. The geometric mean
            # preserves area when uncommon anisotropic DPI metadata is present.
            effective_dpi = math.sqrt(effective_dpi_x * effective_dpi_y)

            pages.append(
                RenderedPage(
                    page_number=index + 1,
                    image_gray=gray,
                    image_rgb=rgb,
                    width=width,
                    height=height,
                    dpi=float(effective_dpi),
                    pdf_width=float(width / effective_dpi_x * 72.0),
                    pdf_height=float(height / effective_dpi_y * 72.0),
                )
            )
        finally:
            frame.close()

    logger.info("Rendered %d native raster page(s)", len(pages))
    return pages


def render_document(doc: object, config: Config) -> list[RenderedPage]:
    """Render the document pages described by ``config`` into rasters."""
    if getattr(doc, "is_raster_document", False):
        return _render_raster_document(doc, config)

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
