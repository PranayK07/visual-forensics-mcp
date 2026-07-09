"""Analysis orchestrator: the deterministic forensic pipeline.

    Document -> Loader -> Renderer -> Tiler -> Visual Engine ->
    PDF Structure Engine -> Detectors -> Evidence Aggregator -> JSON

This module is transport-agnostic: it returns a plain dict matching the
:class:`AnalysisResult` schema and is called by both the MCP server and tests.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from ..analyzers import font_analysis, pdf_structure
from ..analyzers.ocr import OCREngine
from ..analyzers.visual_engine import compute_page_metrics, compute_tile_metrics
from ..detectors import (
    blur_detector,
    compression_detector,
    dpi_detector,
    font_detector,
    font_outlier_detector,
    ocr_confidence_detector,
    stretch_detector,
    structure_detector,
)
from ..render import load_document
from ..render.docx_renderer import extract_docx_formatting
from ..render.pdf_renderer import render_document
from ..schemas.models import (
    AnalysisResult,
    AnalysisSummary,
    EmbeddedImageInfo,
    Finding,
    FontInfo,
    PageResult,
    TileResult,
)
from ..tiling import generate_tiles
from ..utils.config import Config, load_config
from ..utils.logging import get_logger

logger = get_logger("analyze")


def _document_id(path: str) -> str:
    """Deterministic id derived from file content (sha1, first 16 hex chars)."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        h.update(os.path.abspath(path).encode("utf-8"))
    return h.hexdigest()[:16]


def analyze_document(
    document_paths: list[str] | str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full forensic pipeline for each document and return results."""
    if isinstance(document_paths, str):
        document_paths = [document_paths]
    if not isinstance(document_paths, list) or not all(isinstance(p, str) for p in document_paths):
        raise TypeError("document_paths must be a list of file path strings")
    results = [_analyze_single_document(path, options) for path in document_paths]
    return {"results": results}


def _analyze_single_document(
    document_path: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full forensic pipeline for one document and return a schema-valid dict."""
    warnings: list[str] = []
    errors: list[str] = []

    # ---- configuration -----------------------------------------------------
    config_path = (options or {}).get("config_path") if options else None
    overrides = dict(options or {})
    overrides.pop("config_path", None)
    try:
        config: Config = load_config(path=config_path, overrides=overrides)
    except Exception as exc:
        return AnalysisResult(
            document_id="",
            document_type="unknown",
            errors=[f"Failed to load configuration: {exc}"],
        ).model_dump()

    doc_id = _document_id(document_path)

    # ---- load --------------------------------------------------------------
    try:
        loaded = load_document(document_path, config)
    except Exception as exc:
        logger.exception("Document load failed")
        return AnalysisResult(
            document_id=doc_id,
            document_type="unknown",
            errors=[f"Failed to load document: {exc}"],
        ).model_dump()

    warnings.extend(loaded.warnings)
    enable_visual = bool(config.get("features.enable_visual_analysis", True))
    enable_pdf = bool(config.get("features.enable_pdf_analysis", True))
    enable_ocr = bool(config.get("features.enable_ocr", True))
    enable_detectors = bool(config.get("features.enable_detectors", True))
    include_tile_metrics = bool(config.get("output.include_tile_metrics", True))
    max_findings = int(config.get("output.max_findings_per_page", 200))

    # ---- OCR engine (optional, graceful) -----------------------------------
    ocr_engine = None
    if enable_ocr:
        ocr_engine = OCREngine(config)
        if not ocr_engine.available:
            warnings.append(
                "Tesseract OCR binary not available; OCR metrics and the "
                "OCR-confidence detector are disabled for this run."
            )

    page_results: list[PageResult] = []
    document_findings: list[Finding] = []
    all_page_fonts: list[list[dict[str, Any]]] = []
    total_tiles = 0

    # ---- render ------------------------------------------------------------
    try:
        rendered_pages = render_document(loaded.render_doc, config)
    except Exception as exc:
        logger.exception("Rendering failed")
        loaded.close()
        return AnalysisResult(
            document_id=doc_id,
            document_type=loaded.doc_type,
            warnings=warnings,
            errors=[f"Failed to render document: {exc}"],
        ).model_dump()

    pdf_doc = loaded.pdf_doc if enable_pdf else None

    # ---- per-page pipeline -------------------------------------------------
    for rp in rendered_pages:
        try:
            page_result = _process_page(
                rp=rp,
                pdf_doc=pdf_doc,
                config=config,
                ocr_engine=ocr_engine,
                enable_visual=enable_visual,
                enable_pdf=enable_pdf,
                enable_detectors=enable_detectors,
                include_tile_metrics=include_tile_metrics,
                max_findings=max_findings,
                warnings=warnings,
            )
            all_page_fonts.append([f.model_dump() for f in page_result.fonts])
            total_tiles += len(page_result.tiles)
            page_results.append(page_result)
        except Exception as exc:  # pragma: no cover - defensive per-page guard
            logger.exception("Page %d processing failed", rp.page_number)
            errors.append(f"Page {rp.page_number} processing failed: {exc}")

    # ---- document-level structure / fonts ----------------------------------
    pdf_structure_available = pdf_doc is not None
    if loaded.doc_type == "docx" and loaded.docx_obj is not None:
        try:
            docx_fmt = extract_docx_formatting(loaded.docx_obj)
            if docx_fmt.get("fonts"):
                all_page_fonts.append(docx_fmt["fonts"])
        except Exception as exc:
            warnings.append(f"DOCX formatting extraction failed: {exc}")

    if enable_pdf and loaded.doc_type == "pdf":
        try:
            pike = pdf_structure.pikepdf_object_summary(document_path)
            if pike.get("image_xobject_count") is not None:
                logger.info(
                    "pikepdf cross-check: %d image XObject(s)",
                    pike.get("image_xobject_count", 0),
                )
        except Exception as exc:
            warnings.append(f"pikepdf cross-check failed: {exc}")

    aggregated_fonts = font_analysis.aggregate_fonts(all_page_fonts)
    if enable_detectors:
        try:
            document_findings.extend(
                font_detector.detect(aggregated_fonts, config)
            )
        except Exception as exc:
            warnings.append(f"Font detector failed: {exc}")

        # Located font outliers need the document-wide dominant font, so they
        # run as a second pass once every page's fonts have been aggregated.
        if pdf_doc is not None:
            outlier_fonts = aggregated_fonts
            # For DOCX, aggregated_fonts may include python-docx formatting (original
            # font names) in addition to the converted-PDF rendition used for span
            # locations. Base dominance for located outliers on the converted-PDF
            # fonts to avoid false positives from font substitution.
            if loaded.doc_type == "docx":
                outlier_fonts = font_analysis.aggregate_fonts(
                    [[f.model_dump() for f in p.fonts] for p in page_results]
                )

            for page_result in page_results:
                idx = page_result.page - 1
                if idx < 0 or idx >= pdf_doc.page_count:
                    continue
                try:
                    pts_to_px = page_result.dpi / 72.0
                    spans = font_analysis.extract_font_spans(pdf_doc[idx], pts_to_px)
                    page_result.findings.extend(
                        font_outlier_detector.detect(
                            spans,
                            outlier_fonts,
                            page_result.page,
                            pts_to_px,
                            config,
                        )
                    )
                    page_result.findings = page_result.findings[:max_findings]
                except Exception as exc:
                    warnings.append(
                        f"Font outlier detection failed on page "
                        f"{page_result.page}: {exc}"
                    )

    # ---- summary -----------------------------------------------------------
    finding_count = sum(len(p.findings) for p in page_results) + len(document_findings)
    by_type: dict[str, int] = {}
    for p in page_results:
        for f in p.findings:
            by_type[f.type] = by_type.get(f.type, 0) + 1
    for f in document_findings:
        by_type[f.type] = by_type.get(f.type, 0) + 1

    summary = AnalysisSummary(
        page_count=len(page_results),
        tile_count=total_tiles,
        finding_count=finding_count,
        findings_by_type=by_type,
        ocr_available=bool(ocr_engine and ocr_engine.available),
        pdf_structure_available=pdf_structure_available,
        config_digest=config.digest(),
    )

    loaded.close()

    result = AnalysisResult(
        document_id=doc_id,
        document_type=loaded.doc_type,
        summary=summary,
        page_results=page_results,
        document_findings=document_findings,
        warnings=warnings,
        errors=errors,
    )
    return result.model_dump()


def _process_page(
    *,
    rp,
    pdf_doc,
    config: Config,
    ocr_engine,
    enable_visual: bool,
    enable_pdf: bool,
    enable_detectors: bool,
    include_tile_metrics: bool,
    max_findings: int,
    warnings: list[str],
) -> PageResult:
    """Run the full per-page pipeline and return a populated PageResult."""
    page_no = rp.page_number
    page_result = PageResult(
        page=page_no,
        width=rp.width,
        height=rp.height,
        dpi=rp.dpi,
    )

    # --- visual: tiles ---
    tile_results: list[TileResult] = []
    if enable_visual:
        tiles = generate_tiles(page_no, rp.image_gray, rp.image_rgb, config)
        for tile in tiles:
            if tile.blank:
                metrics: dict[str, Any] = {"blank": True}
            else:
                metrics = compute_tile_metrics(
                    tile.image, tile.image_rgb, config, ocr_engine, tile.blank
                )
            tile_results.append(
                TileResult(
                    tile_id=tile.tile_id,
                    page=page_no,
                    bbox=tile.bbox,
                    blank=tile.blank,
                    metrics=metrics,
                )
            )
        page_result.page_metrics = compute_page_metrics(rp.image_gray, config)

    # --- pdf structure ---
    page_structure: dict[str, Any] = {}
    if enable_pdf and pdf_doc is not None and page_no - 1 < pdf_doc.page_count:
        try:
            page_structure = pdf_structure.analyze_page_structure(
                pdf_doc[page_no - 1], rp.dpi, config
            )
            page_result.is_vector = bool(page_structure.get("is_vector", False))
            page_result.embedded_images = [
                EmbeddedImageInfo(**img)
                for img in page_structure.get("embedded_images", [])
            ]
            page_result.fonts = [
                FontInfo(**f) for f in page_structure.get("fonts", [])
            ]
        except Exception as exc:
            warnings.append(f"Page {page_no} structure analysis failed: {exc}")

    # --- detectors ---
    findings: list[Finding] = []
    if enable_detectors:
        if enable_visual and tile_results:
            findings.extend(blur_detector.detect(tile_results, page_no, config))
            findings.extend(
                compression_detector.detect(tile_results, page_no, config)
            )
            if ocr_engine and ocr_engine.available:
                findings.extend(
                    ocr_confidence_detector.detect(tile_results, page_no, config)
                )
        if page_structure:
            embedded = page_structure.get("embedded_images", [])
            findings.extend(
                dpi_detector.detect(embedded, page_no, rp.dpi, config)
            )
            findings.extend(stretch_detector.detect(embedded, page_no, config))
            findings.extend(
                structure_detector.detect(page_structure, page_no, config)
            )

    page_result.findings = findings[:max_findings]
    if include_tile_metrics:
        page_result.tiles = tile_results
    else:
        page_result.tiles = []
    return page_result
