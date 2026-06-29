"""DOCX-specific helpers.

Rendering of DOCX pages is handled by :mod:`pdf_renderer` (PyMuPDF opens DOCX
directly), so this module focuses on native formatting extraction that is only
available from the OOXML package itself.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("docx_renderer")


def extract_docx_formatting(docx_obj: Any) -> dict[str, Any]:
    """Extract font families, counts, and sizes from a python-docx Document.

    Returns a dict shaped like::

        {
          "fonts": [{"name": "Calibri", "span_count": 12, "sizes": [11.0]}],
          "run_count": 40,
          "embedded_image_count": 2,
        }
    """
    result: dict[str, Any] = {
        "fonts": [],
        "run_count": 0,
        "embedded_image_count": 0,
    }
    if docx_obj is None:
        return result

    font_spans: dict[str, int] = defaultdict(int)
    font_sizes: dict[str, set] = defaultdict(set)
    run_count = 0

    try:
        default_font = None
        try:
            default_font = docx_obj.styles["Normal"].font.name
        except Exception:
            default_font = None

        for paragraph in docx_obj.paragraphs:
            for run in paragraph.runs:
                run_count += 1
                name = run.font.name or default_font or "(default)"
                font_spans[name] += 1
                if run.font.size is not None:
                    # size is an EMU/Pt object; .pt gives points
                    try:
                        font_sizes[name].add(round(float(run.font.size.pt), 2))
                    except Exception:
                        pass

        # Tables contain runs too.
        for table in docx_obj.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run_count += 1
                            name = run.font.name or default_font or "(default)"
                            font_spans[name] += 1
                            if run.font.size is not None:
                                try:
                                    font_sizes[name].add(
                                        round(float(run.font.size.pt), 2)
                                    )
                                except Exception:
                                    pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DOCX formatting extraction failed: %s", exc)

    # Embedded images live in the document part's related parts.
    try:
        rels = docx_obj.part.rels
        result["embedded_image_count"] = sum(
            1 for r in rels.values() if "image" in r.reltype
        )
    except Exception:
        pass

    result["run_count"] = run_count
    result["fonts"] = [
        {
            "name": name,
            "span_count": count,
            "sizes": sorted(font_sizes.get(name, set())),
        }
        for name, count in sorted(
            font_spans.items(), key=lambda kv: kv[1], reverse=True
        )
    ]
    return result
