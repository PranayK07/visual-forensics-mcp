"""Font consistency agent: box every place a document departs from its font.

Analyzes a PDF (or DOCX) and produces an annotated copy in which:

* the document's most common (dominant) font is stated in a banner at the top
  of every page and on the cover page;
* every other font detected is listed in that same banner with its usage
  share, detection confidence, and the pages it occurs on; and
* each region of text set in a non-dominant font gets a bounding box drawn
  exactly where it sits on the page, labelled with the font name and the
  detection confidence.

Only font analysis runs (visual/OCR/image detectors are switched off), so the
agent is fast and its output contains font evidence exclusively.

Usage:
    python font_agent.py "path/to/document.pdf"
    python font_agent.py "path/to/document.pdf" --out "path/to/out.pdf"
    python font_agent.py "path/to/document.pdf" --result existing_result.json

Produces:
    "<document> - fonts annotated.pdf"  (a copy of the input, marked up)
    "<document> - fonts result.json"    (the raw analysis, unless --result)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.report.annotator import collect_font_summary, annotate_document  # noqa: E402
from src.tools.analyze import analyze_document  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Run only the font detectors; skip rendering-heavy visual analysis entirely.
FONT_ONLY_OPTIONS: dict = {
    "features": {
        "enable_visual_analysis": False,
        "enable_ocr": False,
    },
    "detectors": {
        "blur": {"enabled": False},
        "ocr_confidence": {"enabled": False},
        "dpi": {"enabled": False},
        "stretch": {"enabled": False},
        "compression": {"enabled": False},
        "structure": {"enabled": False},
    },
    "report": {
        "draw": {
            "add_font_banner": True,
            "add_risk_badge": False,
        },
    },
}


def build_font_report(result: dict) -> dict:
    """Condense an analysis result into the agent's font report."""
    summary = collect_font_summary(result)
    outliers = [
        f
        for page in result.get("page_results", [])
        for f in page.get("findings", [])
        if f.get("type") == "font_outlier"
    ]
    return {
        "dominant_font": summary["dominant"] if summary else None,
        "other_fonts": summary["others"] if summary else [],
        "outlier_regions": outliers,
    }


def print_font_report(report: dict) -> None:
    print("\n=== Font Report ===")
    dom = report["dominant_font"]
    if dom is None:
        print("No text spans found; nothing to report.")
        return
    print(
        f"Dominant font : {dom['name']}  "
        f"({dom['share']:.1%} of text, {dom['span_count']} spans)"
    )
    others = report["other_fonts"]
    if not others:
        print("Other fonts   : none — the document uses a single font family.")
    else:
        print("Other fonts   :")
        for other in others:
            conf = other.get("confidence")
            conf_txt = f"confidence {conf:.2f}" if conf is not None else "not boxed"
            pages = other.get("pages") or []
            pages_txt = (
                "pages " + ", ".join(str(p) for p in pages) if pages else "-"
            )
            print(
                f"  - {other['name']:<24s} {other['share']:>6.1%} of text   "
                f"{conf_txt}   {pages_txt}"
            )
    regions = report["outlier_regions"]
    print(f"Boxed regions : {len(regions)}")
    for f in regions:
        m = f.get("metrics", {})
        bbox = ", ".join(f"{v:.0f}" for v in f.get("bbox", []))
        print(
            f"  - page {f.get('page')}  font '{m.get('font')}'  "
            f"conf {f.get('confidence'):.2f}  bbox [{bbox}] px  "
            f"text: {m.get('text', '')!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the dominant font and draw a labelled bounding box around "
            "every region of text set in a different font."
        )
    )
    parser.add_argument("document", help="Path to the input PDF or DOCX.")
    parser.add_argument("--out", default=None, help="Output annotated PDF path.")
    parser.add_argument("--result", default=None,
                        help="Use an existing result JSON instead of re-analyzing.")
    args = parser.parse_args()

    if not os.path.exists(args.document):
        raise SystemExit(f"File not found: {args.document}")

    if args.result:
        with open(args.result, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        print(f"Loaded existing result: {args.result}")
    else:
        print(f"Analyzing fonts: {args.document}")
        result = analyze_document(args.document, FONT_ONLY_OPTIONS)
        result_path = os.path.splitext(args.document)[0] + " - fonts result.json"
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote analysis: {result_path}")

    if result.get("errors"):
        print("WARNING: analysis reported errors:", result["errors"])

    out = args.out
    if out is None:
        base, ext = os.path.splitext(args.document)
        out = f"{base} - fonts annotated{ext if ext.lower() == '.pdf' else '.pdf'}"

    # DOCX inputs are annotated on their converted-PDF rendition.
    input_pdf = args.document
    tmp_converted: str | None = None
    try:
        if os.path.splitext(args.document)[1].lower() != ".pdf":
            import fitz  # PyMuPDF
            import tempfile

            fd, tmp_converted = tempfile.mkstemp(
                suffix=".pdf", prefix="font_agent_converted_"
            )
            os.close(fd)
            with fitz.open(args.document) as doc:
                pdf_bytes = doc.convert_to_pdf()
            with open(tmp_converted, "wb") as fh:
                fh.write(pdf_bytes)
            input_pdf = tmp_converted
            print(f"Converted to PDF for annotation: {tmp_converted}")

        config = load_config(overrides=FONT_ONLY_OPTIONS)
        out = annotate_document(input_pdf, result, out, config)
    finally:
        if tmp_converted:
            try:
                os.remove(tmp_converted)
            except OSError:
                pass
    report = build_font_report(result)
    print_font_report(report)
    print(f"\nAnnotated PDF : {out}")


if __name__ == "__main__":
    main()
