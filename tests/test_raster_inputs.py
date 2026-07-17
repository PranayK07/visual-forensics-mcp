"""Raster evidence loading, rendering, discovery, and pipeline tests."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

import fitz

from src.render import (
    convert_to_pdf_bytes,
    convert_to_pdf_path,
    detect_type,
    load_document,
    render_document,
)
from src.schemas.models import AnalysisResult
from src.tools.analyze import analyze_document
from src.utils.claim_batch import discover_claim_sets, list_documents_in_dir
from src.utils.config import load_config


RASTER_FORMATS = [
    ("png", "PNG", "png"),
    ("jpg", "JPEG", "jpeg"),
    ("jpeg", "JPEG", "jpeg"),
    ("tif", "TIFF", "tiff"),
    ("tiff", "TIFF", "tiff"),
    ("bmp", "BMP", "bmp"),
    ("webp", "WEBP", "webp"),
    ("gif", "GIF", "gif"),
]


def _pattern(width: int = 96, height: int = 64) -> Image.Image:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        [
            (xx * 3 + yy) % 256,
            (yy * 5) % 256,
            ((xx // 8 + yy // 8) % 2) * 220,
        ],
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


@pytest.mark.parametrize("extension,pillow_format,expected_type", RASTER_FORMATS)
def test_raster_formats_load_and_preserve_native_pixels(
    tmp_path, extension, pillow_format, expected_type
):
    path = tmp_path / f"evidence.{extension}"
    _pattern().save(path, format=pillow_format, dpi=(200, 200))
    config = load_config(overrides={"render": {"dpi": 72}})

    assert detect_type(str(path)) == expected_type
    loaded = load_document(str(path), config)
    try:
        pages = render_document(loaded.render_doc, config)
    finally:
        loaded.close()

    assert loaded.doc_type == expected_type
    assert loaded.pdf_doc is None
    assert len(pages) == 1
    page = pages[0]
    # Raster evidence is analyzed at its submitted resolution, not enlarged to
    # the generic document-render DPI.
    assert (page.width, page.height) == (96, 64)
    assert page.image_rgb.shape == (64, 96, 3)
    assert page.image_gray.shape == (64, 96)
    assert page.image_rgb.dtype == np.uint8


def test_content_signature_identifies_mislabeled_raster(tmp_path):
    path = tmp_path / "scanner_export.jpg"
    _pattern().save(path, format="PNG")
    assert detect_type(str(path)) == "png"


def test_extensionless_png_is_detected_by_content(tmp_path):
    path = tmp_path / "scanner_export"
    _pattern().save(path, format="PNG")
    assert detect_type(str(path)) == "png"


def test_transparent_pixels_composite_onto_white(tmp_path):
    path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (8, 6), (10, 20, 30, 0))
    image.putpixel((3, 2), (100, 110, 120, 255))
    image.save(path)
    config = load_config(overrides={"render": {"dpi": 100}})

    loaded = load_document(str(path), config)
    try:
        page = render_document(loaded.render_doc, config)[0]
    finally:
        loaded.close()

    assert page.image_rgb[0, 0].tolist() == [255, 255, 255]
    assert page.image_rgb[2, 3].tolist() == [100, 110, 120]


def test_exif_orientation_is_applied_without_resampling(tmp_path):
    path = tmp_path / "phone_photo.jpg"
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise for display
    _pattern(width=80, height=40).save(path, format="JPEG", exif=exif)
    config = load_config(overrides={"render": {"dpi": 100}})

    loaded = load_document(str(path), config)
    try:
        page = render_document(loaded.render_doc, config)[0]
    finally:
        loaded.close()

    assert (page.width, page.height) == (40, 80)


def test_multipage_tiff_renders_each_frame(tmp_path):
    path = tmp_path / "fax.tiff"
    first = Image.new("RGB", (40, 30), "white")
    second = Image.new("RGB", (40, 30), "black")
    first.save(path, save_all=True, append_images=[second], format="TIFF", dpi=(204, 196))
    config = load_config(overrides={"render": {"dpi": 100}})

    loaded = load_document(str(path), config)
    try:
        pages = render_document(loaded.render_doc, config)
    finally:
        loaded.close()

    assert [p.page_number for p in pages] == [1, 2]
    assert all((p.width, p.height) == (40, 30) for p in pages)
    assert pages[0].image_gray.mean() == 255
    assert pages[1].image_gray.mean() == 0


def test_raster_max_pages_limits_multiframe_input(tmp_path):
    path = tmp_path / "fax.tiff"
    first = Image.new("RGB", (40, 30), "white")
    second = Image.new("RGB", (40, 30), "black")
    first.save(path, save_all=True, append_images=[second], format="TIFF")
    config = load_config(overrides={"render": {"max_pages": 1}})

    loaded = load_document(str(path), config)
    try:
        pages = render_document(loaded.render_doc, config)
    finally:
        loaded.close()

    assert len(pages) == 1
    assert pages[0].page_number == 1


def test_raster_max_pixels_downsamples_without_aspect_change(tmp_path):
    path = tmp_path / "large.png"
    _pattern(200, 100).save(path)
    config = load_config(overrides={"render": {"dpi": 100, "max_pixels": 5_000}})

    loaded = load_document(str(path), config)
    try:
        page = render_document(loaded.render_doc, config)[0]
    finally:
        loaded.close()

    assert page.width * page.height <= 5_100  # rounding tolerance
    assert page.width / page.height == pytest.approx(2.0, rel=0.02)


def test_raster_end_to_end_is_schema_valid_and_skips_pdf_structure(tmp_path, fast_options):
    path = tmp_path / "claim_photo.png"
    _pattern(640, 640).save(path)

    result = analyze_document(str(path), fast_options)["results"][0]
    AnalysisResult.model_validate(result)

    assert result["document_type"] == "png"
    assert result["errors"] == []
    assert result["summary"]["page_count"] == 1
    assert result["summary"]["tile_count"] > 0
    assert result["summary"]["pdf_structure_available"] is False
    assert result["page_results"][0]["page_metrics"]
    assert result["page_results"][0]["fonts"] == []
    assert result["page_results"][0]["embedded_images"] == []


def test_claim_folder_discovers_all_raster_extensions(tmp_path):
    for extension, pillow_format, _ in RASTER_FORMATS:
        _pattern().save(tmp_path / f"evidence.{extension}", format=pillow_format)
    (tmp_path / "ignore.txt").write_text("not evidence")

    docs = list_documents_in_dir(str(tmp_path))
    assert {os.path.splitext(path)[1].lower() for path in docs} == {
        f".{extension}" for extension, _, _ in RASTER_FORMATS
    }
    claim_sets = discover_claim_sets(str(tmp_path))
    assert len(claim_sets) == 1
    assert len(claim_sets[0].documents) == len(RASTER_FORMATS)


def test_corrupt_recognized_image_returns_factual_load_error(tmp_path, fast_options):
    path = tmp_path / "damaged.jpg"
    path.write_bytes(b"not a jpeg")

    result = analyze_document(str(path), fast_options)["results"][0]
    assert result["document_type"] == "unknown"
    assert result["summary"]["page_count"] == 0
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("Failed to load document:")
    assert "fraud" not in result["errors"][0].lower()


@pytest.mark.parametrize("extension,pillow_format,_expected_type", RASTER_FORMATS)
def test_every_raster_format_converts_to_annotatable_pdf(
    tmp_path, extension, pillow_format, _expected_type
):
    path = tmp_path / f"evidence.{extension}"
    _pattern().save(path, format=pillow_format)

    pdf_bytes = convert_to_pdf_bytes(str(path))
    with fitz.open("pdf", pdf_bytes) as converted:
        assert converted.page_count == 1
        assert converted[0].rect.width > 0
        assert converted[0].rect.height > 0


def test_multiframe_raster_conversion_preserves_all_pages(tmp_path):
    path = tmp_path / "fax.tiff"
    first = Image.new("RGB", (40, 30), "white")
    second = Image.new("RGB", (40, 30), "black")
    first.save(path, save_all=True, append_images=[second], format="TIFF")

    pdf_bytes = convert_to_pdf_bytes(str(path))
    with fitz.open("pdf", pdf_bytes) as converted:
        assert converted.page_count == 2


def test_pdf_conversion_helper_preserves_existing_pdf_bytes(sample_pdf):
    with open(sample_pdf, "rb") as source:
        original = source.read()
    assert convert_to_pdf_bytes(sample_pdf) == original


def test_docx_conversion_helper_produces_pdf(sample_docx):
    pdf_bytes = convert_to_pdf_bytes(sample_docx)
    with fitz.open("pdf", pdf_bytes) as converted:
        assert converted.page_count >= 1


def test_convert_to_pdf_path_writes_requested_file(tmp_path):
    source = tmp_path / "source.webp"
    output = tmp_path / "converted.pdf"
    _pattern().save(source, format="WEBP")

    returned = convert_to_pdf_path(str(source), str(output))
    assert returned == str(output)
    assert output.read_bytes().startswith(b"%PDF")
