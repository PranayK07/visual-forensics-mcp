"""Conversion helpers for producing annotatable PDF copies of evidence.

Analysis always runs against the original input. These helpers exist only for
the visual-annotation stage, which needs a PDF canvas on which to draw located
findings. PyMuPDF is used when it supports the source directly; Pillow provides
a deterministic bridge for WebP and multi-frame raster files.
"""

from __future__ import annotations

import io
import math
import os

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from .loader import RasterDocument, detect_type

_RASTER_TYPES = {"png", "jpeg", "tiff", "bmp", "webp", "gif"}


def _positive_dpi(value: object, fallback: float = 96.0) -> tuple[float, float]:
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
    return fallback, fallback


def _rgb_on_white(frame: "Image.Image") -> "Image.Image":
    has_alpha = frame.mode in {"RGBA", "LA"} or (
        frame.mode == "P" and "transparency" in frame.info
    )
    if not has_alpha:
        return frame.convert("RGB")
    rgba = frame.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def _raster_to_pdf_bytes(path: str) -> bytes:
    """Build a PDF with one full-page lossless image per raster frame."""
    source = RasterDocument(path)
    output = fitz.open()
    try:
        for index in range(source.page_count):
            frame, info = source.get_frame(index)
            try:
                frame = ImageOps.exif_transpose(frame)
                rgb = _rgb_on_white(frame)
                dpi_x, dpi_y = _positive_dpi(info.get("dpi"))
                width_pt = max(1.0, rgb.width / dpi_x * 72.0)
                height_pt = max(1.0, rgb.height / dpi_y * 72.0)

                encoded = io.BytesIO()
                rgb.save(encoded, format="PNG")
                page = output.new_page(width=width_pt, height=height_pt)
                page.insert_image(page.rect, stream=encoded.getvalue())
            finally:
                frame.close()
        return output.tobytes(garbage=4, deflate=True)
    finally:
        output.close()
        source.close()


def convert_to_pdf_bytes(document_path: str) -> bytes:
    """Return an annotatable PDF representation of a supported input.

    PDF bytes are returned unchanged. DOCX and directly supported single-frame
    raster inputs use PyMuPDF conversion. WebP, multi-frame TIFF/GIF, and any
    raster that PyMuPDF cannot open fall back to Pillow without dropping pages.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path!r}")
    if not os.path.isfile(document_path):
        raise ValueError(f"Not a file: {document_path!r}")

    doc_type = detect_type(document_path)
    if doc_type == "pdf":
        with open(document_path, "rb") as source:
            return source.read()

    if doc_type == "docx":
        with fitz.open(document_path) as source:
            return source.convert_to_pdf()

    if doc_type not in _RASTER_TYPES:
        raise ValueError(f"Unsupported evidence type: {doc_type!r}")

    raster = RasterDocument(document_path)
    try:
        needs_pillow_bridge = doc_type == "webp" or raster.page_count > 1
    finally:
        raster.close()

    if not needs_pillow_bridge:
        try:
            with fitz.open(document_path) as source:
                return source.convert_to_pdf()
        except Exception:
            # Content-signature detection may identify a format even when the
            # filename extension confuses MuPDF. Pillow already validated it.
            pass
    return _raster_to_pdf_bytes(document_path)


def convert_to_pdf_path(document_path: str, output_path: str) -> str:
    """Write :func:`convert_to_pdf_bytes` to ``output_path`` and return it."""
    pdf_bytes = convert_to_pdf_bytes(document_path)
    with open(output_path, "wb") as output:
        output.write(pdf_bytes)
    return output_path
