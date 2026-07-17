"""Font analysis for PDF (and converted DOCX) pages.

Extracts font families, per-font span counts, and the set of sizes used, both
per page and aggregated across the document. The detector layer decides whether
the distribution is anomalous; this module only measures.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

import fitz  # PyMuPDF

from ..utils.logging import get_logger

logger = get_logger("font_analysis")


_STYLE_TOKENS = {
    "black",
    "bold",
    "book",
    "condensed",
    "demi",
    "extrabold",
    "heavy",
    "italic",
    "light",
    "medium",
    "oblique",
    "regular",
    "roman",
    "semibold",
    "thin",
}
_SPACE_STYLE_TOKENS = _STYLE_TOKENS - {"book", "medium", "roman"}


def _is_style_suffix(value: str) -> bool:
    """Return whether ``value`` consists only of common font-style words."""
    compact = re.sub(r"[^a-z]", "", value.lower())
    if not compact:
        return False
    if compact.endswith("mt") and len(compact) > 2:
        compact = compact[:-2]
    if compact in _STYLE_TOKENS:
        return True
    # Handles combined PostScript suffixes such as BoldItalic and SemiBold.
    remaining = compact
    for token in sorted(_STYLE_TOKENS, key=len, reverse=True):
        remaining = remaining.replace(token, "")
    return not remaining


def _normalise_font_name(raw: str) -> str:
    """Return a stable font-family name, ignoring subset and style variants.

    PDF producers commonly expose one family as names such as ``ArialMT``,
    ``Arial-BoldMT`` and ``TimesNewRomanPS-ItalicMT``. Treating those as
    different families makes ordinary bold or italic text look anomalous.
    This normalisation deliberately removes only recognised style suffixes;
    unrelated hyphenated family names remain distinct.
    """
    if not raw:
        return "(unknown)"
    name = raw.strip()
    if "+" in name and len(name.split("+", 1)[0]) == 6:
        name = name.split("+", 1)[1]

    # A comma or hyphen normally separates a PostScript family from its style.
    for separator in (",", "-"):
        if separator in name:
            family, suffix = name.rsplit(separator, 1)
            if family and _is_style_suffix(suffix):
                name = family

    # Some extractors use a space: ``Times New Roman Bold``.
    parts = name.split()
    while (
        len(parts) > 1
        and re.sub(r"[^a-z]", "", parts[-1].lower()) in _SPACE_STYLE_TOKENS
    ):
        parts.pop()
    name = " ".join(parts)

    # PostScript foundry suffixes appear with and without an explicit style.
    if name.lower().endswith("psmt") and len(name) > 4:
        name = name[:-4]
    elif name.lower().endswith("ps") and len(name) > 2:
        name = name[:-2]
    elif name.lower().endswith("mt") and len(name) > 2:
        name = name[:-2]

    return name or "(unknown)"


def analyze_page_fonts(page: "fitz.Page") -> dict[str, Any]:
    """Return font usage for a single page.

    Shape::

        {
          "fonts": [{"name": "Arial", "span_count": 10, "sizes": [11.0, 12.0]}],
          "span_count": 42,
        }
    """
    spans = 0
    counts: dict[str, int] = defaultdict(int)
    sizes: dict[str, set] = defaultdict(set)

    try:
        data = page.get_text("dict")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("get_text failed on page: %s", exc)
        return {"fonts": [], "span_count": 0}

    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                name = _normalise_font_name(span.get("font", ""))
                counts[name] += 1
                spans += 1
                size = span.get("size")
                if size is not None:
                    sizes[name].add(round(float(size), 2))

    fonts = [
        {
            "name": name,
            "span_count": count,
            "sizes": sorted(sizes.get(name, set())),
        }
        for name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {"fonts": fonts, "span_count": spans}


def extract_font_spans(
    page: "fitz.Page",
    pts_to_px: float = 1.0,
) -> list[dict[str, Any]]:
    """Return every non-empty text span on the page with its font and location.

    Each entry::

        {
          "font": "Times-Roman",
          "size": 11.0,
          "bbox": [x1, y1, x2, y2],   # rendered pixel coordinates
          "text": "the span text",
        }

    ``bbox`` values are converted from PDF points to rendered pixels using
    ``pts_to_px`` (= render_dpi / 72) so they line up with every other finding
    coordinate in the pipeline. Whitespace-only spans are skipped.
    """
    out: list[dict[str, Any]] = []
    try:
        data = page.get_text("dict")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("get_text failed on page: %s", exc)
        return out

    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                bbox = span.get("bbox")
                if not text.strip() or not bbox or len(bbox) != 4:
                    continue
                out.append(
                    {
                        "font": _normalise_font_name(span.get("font", "")),
                        "size": round(float(span.get("size", 0.0)), 2),
                        "bbox": [float(v) * pts_to_px for v in bbox],
                        "text": text,
                    }
                )
    return out


def aggregate_fonts(page_font_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge per-page font lists into a document-level list."""
    counts: dict[str, int] = defaultdict(int)
    sizes: dict[str, set] = defaultdict(set)
    for page_fonts in page_font_lists:
        for font in page_fonts:
            counts[font["name"]] += int(font.get("span_count", 0))
            for s in font.get("sizes", []):
                sizes[font["name"]].add(s)
    return [
        {"name": name, "span_count": count, "sizes": sorted(sizes.get(name, set()))}
        for name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
