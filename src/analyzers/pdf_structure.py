"""PDF structure analysis.

Inspects the structural (vector/raster/font/object) layer of a page using
PyMuPDF, with an optional pikepdf cross-check of image XObjects at the document
level. Produces measurements only; detectors apply thresholds.

For each page we extract:

* embedded images (original dims, displayed dims, location, scaling, DPI)
* raster coverage of the page (union area, not double-counted)
* whether the page is predominantly vector or raster
* overlapping / stacked image regions
* font usage (delegated to :mod:`font_analysis`)
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

import fitz  # PyMuPDF

from ..utils.config import Config
from ..utils.geometry import iou
from ..utils.logging import get_logger
from . import font_analysis, image_scaling

logger = get_logger("pdf_structure")

POINTS_PER_INCH = 72.0


def _union_coverage(
    rects_pts: list[tuple[float, float, float, float]],
    page_w: float,
    page_h: float,
    grid: int = 120,
) -> float:
    """Union area fraction of ``rects`` over the page, via a coarse grid mask."""
    if page_w <= 0 or page_h <= 0 or not rects_pts:
        return 0.0
    mask = np.zeros((grid, grid), dtype=bool)
    sx = grid / page_w
    sy = grid / page_h
    for x0, y0, x1, y1 in rects_pts:
        gx0 = max(0, min(grid, int(np.floor(x0 * sx))))
        gx1 = max(0, min(grid, int(np.ceil(x1 * sx))))
        gy0 = max(0, min(grid, int(np.floor(y0 * sy))))
        gy1 = max(0, min(grid, int(np.ceil(y1 * sy))))
        if gx1 > gx0 and gy1 > gy0:
            mask[gy0:gy1, gx0:gx1] = True
    return float(mask.sum()) / float(grid * grid)


def analyze_page_structure(
    page: "fitz.Page",
    render_dpi: float,
    config: Config | None = None,
) -> dict[str, Any]:
    """Analyze one page's structure. ``bbox`` values are in render pixels."""
    pts_to_px = render_dpi / POINTS_PER_INCH
    rect = page.rect
    page_w, page_h = float(rect.width), float(rect.height)

    embedded: list[dict[str, Any]] = []
    raster_rects: list[tuple[float, float, float, float]] = []

    try:
        infos = page.get_image_info(xrefs=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("get_image_info failed: %s", exc)
        infos = []

    for info in infos:
        bbox_pts = info.get("bbox")
        if not bbox_pts:
            continue
        x0, y0, x1, y1 = bbox_pts
        disp_w_pts = max(0.0, float(x1) - float(x0))
        disp_h_pts = max(0.0, float(y1) - float(y0))
        orig_w = int(info.get("width", 0) or 0)
        orig_h = int(info.get("height", 0) or 0)
        if orig_w <= 0 or orig_h <= 0:
            continue

        scaling = image_scaling.compute_scaling(
            orig_w, orig_h, disp_w_pts, disp_h_pts
        )
        raster_rects.append((float(x0), float(y0), float(x1), float(y1)))

        embedded.append(
            {
                "xref": int(info.get("xref", 0) or 0),
                "original_width": orig_w,
                "original_height": orig_h,
                "display_width": round(disp_w_pts, 4),
                "display_height": round(disp_h_pts, 4),
                "bbox": [
                    float(x0) * pts_to_px,
                    float(y0) * pts_to_px,
                    float(x1) * pts_to_px,
                    float(y1) * pts_to_px,
                ],
                "scale_x": scaling["scale_x"],
                "scale_y": scaling["scale_y"],
                "effective_dpi_x": scaling["effective_dpi_x"],
                "effective_dpi_y": scaling["effective_dpi_y"],
                "colorspace": info.get("cs-name"),
                "bpc": info.get("bpc"),
                "image_filter": None,
            }
        )

    # Overlap / stacking detection between placed images.
    overlaps: list[dict[str, Any]] = []
    for a, b in combinations(range(len(embedded)), 2):
        score = iou(embedded[a]["bbox"], embedded[b]["bbox"])
        if score > 0:
            overlaps.append(
                {
                    "xref_a": embedded[a]["xref"],
                    "xref_b": embedded[b]["xref"],
                    "iou": round(score, 4),
                    "bbox": embedded[a]["bbox"],
                }
            )

    raster_coverage = _union_coverage(raster_rects, page_w, page_h)

    # Vector / text presence.
    try:
        drawings = page.get_drawings()
        drawing_count = len(drawings)
    except Exception:
        drawing_count = 0

    fonts = font_analysis.analyze_page_fonts(page)
    span_count = fonts.get("span_count", 0)

    has_vector_text = span_count > 0
    has_vector_graphics = drawing_count > 0
    is_vector = (has_vector_text or has_vector_graphics) and raster_coverage < 0.85

    return {
        "embedded_images": embedded,
        "image_count": len(embedded),
        "raster_coverage": round(raster_coverage, 4),
        "drawing_count": drawing_count,
        "span_count": span_count,
        "fonts": fonts.get("fonts", []),
        "is_vector": bool(is_vector),
        "overlaps": overlaps,
        "page_width_pts": page_w,
        "page_height_pts": page_h,
    }


def pikepdf_object_summary(path: str) -> dict[str, Any]:
    """Document-level image XObject cross-check using pikepdf.

    Returns counts of image XObjects and the set of compression filters used,
    which corroborates the PyMuPDF view. Best-effort; returns empty on failure.
    """
    summary: dict[str, Any] = {"image_xobject_count": 0, "filters": {}}
    try:
        import pikepdf

        with pikepdf.open(path) as pdf:
            filters: dict[str, int] = {}
            count = 0
            for page in pdf.pages:
                resources = page.get("/Resources")
                if resources is None:
                    continue
                xobjects = resources.get("/XObject")
                if xobjects is None:
                    continue
                for obj in xobjects.values():
                    try:
                        if obj.get("/Subtype") == "/Image":
                            count += 1
                            filt = obj.get("/Filter")
                            if filt is not None:
                                key = str(filt)
                                filters[key] = filters.get(key, 0) + 1
                    except Exception:
                        continue
            summary["image_xobject_count"] = count
            summary["filters"] = filters
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("pikepdf summary failed: %s", exc)
    return summary
