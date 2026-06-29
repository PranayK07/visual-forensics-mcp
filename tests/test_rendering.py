"""PDF rendering tests."""

from __future__ import annotations

import numpy as np

from src.render import load_document, render_document
from src.utils.config import load_config


def test_pdf_renders_pages(sample_pdf):
    config = load_config(overrides={"render": {"dpi": 150}})
    loaded = load_document(sample_pdf, config)
    try:
        pages = render_document(loaded.render_doc, config)
    finally:
        loaded.close()

    assert len(pages) >= 1
    page = pages[0]
    assert page.page_number == 1
    assert page.width > 0 and page.height > 0
    assert page.dpi == 150
    assert page.image_gray.ndim == 2
    assert page.image_rgb.shape[2] == 3
    assert page.image_gray.dtype == np.uint8
    # US Letter @150 dpi -> ~1275 x 1650 px
    assert 1000 < page.width < 1400
    assert 1400 < page.height < 1800


def test_dpi_scales_resolution(sample_pdf):
    cfg_low = load_config(overrides={"render": {"dpi": 100}})
    cfg_high = load_config(overrides={"render": {"dpi": 200}})
    low = load_document(sample_pdf, cfg_low)
    high = load_document(sample_pdf, cfg_high)
    try:
        p_low = render_document(low.render_doc, cfg_low)[0]
        p_high = render_document(high.render_doc, cfg_high)[0]
    finally:
        low.close()
        high.close()
    assert p_high.width > p_low.width
    assert p_high.height > p_low.height
