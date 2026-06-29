"""Pytest fixtures and path setup."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the project root (containing the `src` package) is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.builders import (  # noqa: E402
    build_sample_docx,
    build_sample_pdf,
    build_scanned_like_pdf,
)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("docs") / "sample.pdf"
    return build_sample_pdf(str(path))


@pytest.fixture(scope="session")
def scanned_pdf(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("docs") / "scanned.pdf"
    return build_scanned_like_pdf(str(path))


@pytest.fixture(scope="session")
def sample_docx(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("docs") / "sample.docx"
    return build_sample_docx(str(path))


@pytest.fixture(scope="session")
def fast_options() -> dict:
    """Lower DPI + OCR off keeps the suite quick and binary-independent."""
    return {"render": {"dpi": 150}, "features": {"enable_ocr": False}}
