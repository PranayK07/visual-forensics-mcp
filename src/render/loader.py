"""Document loader.

Detects the document type and opens the resources needed downstream:

* PDF  -> a PyMuPDF document used for both rendering and structure analysis.
* DOCX -> a PyMuPDF document (opened directly) for rendering, a converted
          in-memory PDF for structure analysis, and a python-docx handle for
          formatting/font extraction.

All loading is local and deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger("loader")

_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx"}


@dataclass
class LoadedDocument:
    """Container for everything the pipeline needs from one input file."""

    path: str
    doc_type: str  # "pdf" or "docx"
    render_doc: "fitz.Document"  # used by the renderer (any fitz-openable doc)
    pdf_doc: Optional["fitz.Document"] = None  # PDF view for structure analysis
    docx_obj: object = None  # python-docx Document, for DOCX only
    warnings: list[str] = field(default_factory=list)

    def close(self) -> None:
        for doc in (self.render_doc, self.pdf_doc):
            try:
                if doc is not None:
                    doc.close()
            except Exception:  # pragma: no cover - defensive
                pass


def detect_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _PDF_EXTS:
        return "pdf"
    if ext in _DOCX_EXTS:
        return "docx"
    # Fall back to content sniffing.
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
        if head.startswith(b"%PDF"):
            return "pdf"
        if head.startswith(b"PK\x03\x04"):
            # zip container -> assume modern OOXML docx
            return "docx"
    except OSError:
        pass
    raise ValueError(f"Unsupported or unrecognised document type: {path!r}")


def load_document(path: str, config: Config) -> LoadedDocument:
    """Open ``path`` and return a :class:`LoadedDocument`.

    Raises ``FileNotFoundError`` / ``ValueError`` for invalid inputs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found: {path!r}")
    if not os.path.isfile(path):
        raise ValueError(f"Not a file: {path!r}")

    doc_type = detect_type(path)
    logger.info("Loading %s document: %s", doc_type, path)

    if doc_type == "pdf":
        render_doc = fitz.open(path)
        return LoadedDocument(
            path=path,
            doc_type="pdf",
            render_doc=render_doc,
            pdf_doc=render_doc,  # same handle; structure analyzer reads only
        )

    # DOCX
    render_doc = fitz.open(path)
    loaded = LoadedDocument(path=path, doc_type="docx", render_doc=render_doc)

    # Converted PDF view enables PDF-structure analysis (images, fonts, objects).
    if config.get("features.enable_pdf_analysis", True):
        try:
            pdf_bytes = render_doc.convert_to_pdf()
            loaded.pdf_doc = fitz.open("pdf", pdf_bytes)
        except Exception as exc:  # pragma: no cover - depends on libmupdf build
            loaded.warnings.append(
                f"DOCX->PDF conversion failed; structure analysis limited: {exc}"
            )
            logger.warning("DOCX->PDF conversion failed: %s", exc)

    # python-docx handle for native formatting/font extraction.
    try:
        import docx  # local import keeps PDF path lightweight

        loaded.docx_obj = docx.Document(path)
    except Exception as exc:  # pragma: no cover - defensive
        loaded.warnings.append(f"python-docx load failed: {exc}")
        logger.warning("python-docx load failed: %s", exc)

    return loaded
