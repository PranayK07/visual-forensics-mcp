"""DOCX loading and formatting-extraction tests."""

from __future__ import annotations

from src.render import load_document, render_document
from src.render.docx_renderer import extract_docx_formatting
from src.utils.config import load_config


def test_docx_loads_and_renders(sample_docx):
    config = load_config(overrides={"render": {"dpi": 120}})
    loaded = load_document(sample_docx, config)
    try:
        assert loaded.doc_type == "docx"
        assert loaded.docx_obj is not None
        pages = render_document(loaded.render_doc, config)
        assert len(pages) >= 1
        assert pages[0].image_gray.ndim == 2
    finally:
        loaded.close()


def test_docx_font_extraction(sample_docx):
    config = load_config()
    loaded = load_document(sample_docx, config)
    try:
        fmt = extract_docx_formatting(loaded.docx_obj)
    finally:
        loaded.close()
    assert fmt["run_count"] > 0
    names = {f["name"] for f in fmt["fonts"]}
    # The rare run uses Courier New.
    assert "Courier New" in names
    assert fmt["embedded_image_count"] >= 1


def test_docx_converts_to_pdf_for_structure(sample_docx):
    config = load_config()
    loaded = load_document(sample_docx, config)
    try:
        # convert_to_pdf path populates pdf_doc for structure analysis.
        assert loaded.pdf_doc is not None
        assert loaded.pdf_doc.page_count >= 1
    finally:
        loaded.close()
