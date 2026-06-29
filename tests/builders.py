"""Deterministic synthetic document builders for tests and examples.

These create small PDF/DOCX files that exercise the full pipeline and reliably
trigger several structural findings (stretch, low DPI, raster-in-vector, font
mismatch) without relying on any external resources.
"""

from __future__ import annotations

import io

import numpy as np

import fitz  # PyMuPDF


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to PNG bytes via PyMuPDF (no cv2 dep)."""
    pix = fitz.Pixmap(
        fitz.csRGB, fitz.IRect(0, 0, rgb.shape[1], rgb.shape[0]), False
    )
    pix.set_rect(pix.irect, bytes(np.ascontiguousarray(rgb).tobytes()))
    return pix.tobytes("png")


def _checker(width: int, height: int) -> np.ndarray:
    """A small deterministic checkerboard image (low native resolution)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cell = max(2, width // 8)
    for y in range(height):
        for x in range(width):
            img[y, x] = 230 if ((x // cell + y // cell) % 2 == 0) else 40
    return img


def build_sample_pdf(path: str) -> str:
    """Create a native PDF with text, fonts, and a stretched low-res image."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter, points

    # Body text in a single dominant font -> many spans.
    y = 80
    for i in range(24):
        page.insert_text(
            (72, y),
            f"Line {i + 1}: deterministic forensic sample content for analysis.",
            fontname="helv",
            fontsize=11,
        )
        y += 18

    # A few rare-font spans -> font_mismatch.
    page.insert_text((72, y + 10), "Rare italic insertion line.", fontname="tiro", fontsize=11)

    # Low-resolution image (40x40) displayed large and non-uniformly
    # (200 wide x 120 tall points) -> low effective DPI + stretch + raster.
    small = _checker(40, 40)
    rect = fitz.Rect(320, 300, 520, 420)
    page.insert_image(rect, stream=_png_bytes(small))

    doc.save(path)
    doc.close()
    return path


def build_scanned_like_pdf(path: str) -> str:
    """Create an image-only (scanned-like) PDF: a full-page raster, no text."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    raster = _checker(300, 388)  # ~ page aspect
    page.insert_image(page.rect, stream=_png_bytes(raster))
    doc.save(path)
    doc.close()
    return path


def build_sample_docx(path: str) -> str:
    """Create a DOCX with headings, paragraphs, a rare font run, and an image."""
    import docx
    from docx.shared import Pt

    d = docx.Document()
    d.add_heading("Visual Forensics Sample", level=0)
    for i in range(6):
        d.add_paragraph(
            f"Paragraph {i + 1}: deterministic content used to validate the "
            "DOCX analysis pipeline end to end."
        )

    # A run in an unusual font family.
    p = d.add_paragraph()
    run = p.add_run("Monospaced inserted snippet.")
    run.font.name = "Courier New"
    run.font.size = Pt(11)

    # Embed an image so the converted-PDF structure analysis sees a raster.
    img = _checker(60, 60)
    d.add_picture(io.BytesIO(_png_bytes(img)))

    d.save(path)
    return path
