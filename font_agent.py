"""Font consistency agent: box every place a document departs from its font.

Analyzes PDF/DOCX files and produces annotated copies in which:

* the document's most common (dominant) font is stated in a banner at the top
  of every page and on the cover page;
* every other font detected is listed in that same banner with its usage
  share, detection confidence, and the pages it occurs on; and
* each region of text set in a non-dominant font gets a bounding box drawn
  exactly where it sits on the page, labelled with the font name and the
  detection confidence.

Accepts a single file, a claim-set folder, or a claims root whose subfolders
are claim sets. Outputs mirror claim-set folders under ``--out-dir``.

Only font analysis runs (visual/OCR/image detectors are switched off), so the
agent is fast and its output contains font evidence exclusively.

Usage:
    python font_agent.py "path/to/document.pdf"
    python font_agent.py "path/to/claim_set/"
    python font_agent.py "path/to/claims_root/" --out-dir "path/to/out/"
    python font_agent.py "path/to/document.pdf" --out "path/to/out.pdf"
    python font_agent.py "path/to/document.pdf" --result existing_result.json

Produces (per document):
    "<stem> - fonts annotated.pdf"
    "<stem> - fonts result.json"    (unless --result was given for a single file)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.report.annotator import collect_font_summary, annotate_document  # noqa: E402
from src.tools.analyze import analyze_document  # noqa: E402
from src.utils.claim_batch import (  # noqa: E402
    annotated_pdf_path,
    claim_output_dir,
    default_out_dir,
    discover_claim_sets,
    ensure_dir,
    result_json_path,
)
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


def _convert_to_pdf(document_path: str) -> tuple[str, str | None]:
    if os.path.splitext(document_path)[1].lower() == ".pdf":
        return document_path, None

    import fitz  # PyMuPDF

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="font_agent_converted_")
    os.close(fd)
    with fitz.open(document_path) as doc:
        pdf_bytes = doc.convert_to_pdf()
    with open(tmp_path, "wb") as fh:
        fh.write(pdf_bytes)
    return tmp_path, tmp_path


def _process_document(
    document_path: str,
    result: dict,
    output_dir: str,
    out_override: str | None = None,
) -> str:
    if result.get("errors"):
        print(f"WARNING: {os.path.basename(document_path)} errors:", result["errors"])

    out = out_override or annotated_pdf_path(
        document_path, output_dir, suffix="fonts annotated"
    )

    input_pdf, tmp_converted = _convert_to_pdf(document_path)
    try:
        if tmp_converted:
            print(f"Converted to PDF for annotation: {document_path}")
        config = load_config(overrides=FONT_ONLY_OPTIONS)
        out = annotate_document(input_pdf, result, out, config)
    finally:
        if tmp_converted:
            try:
                os.remove(tmp_converted)
            except OSError:
                pass

    report = build_font_report(result)
    print(f"\n--- {os.path.basename(document_path)} ---")
    print_font_report(report)
    print(f"Annotated PDF : {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the dominant font and draw a labelled bounding box around "
            "every region of text set in a different font. Pass a file, a "
            "claim-set folder, or a parent folder of claim sets."
        )
    )
    parser.add_argument(
        "input_path",
        help=(
            "Path to a PDF/DOCX file, a claim-set folder of documents, "
            "or a claims root whose subfolders are claim sets."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory for annotated outputs. Defaults to the input file's "
            "folder, or to '<folder>_annotated' beside a claim-set / claims root."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output annotated PDF path (single-file mode only).",
    )
    parser.add_argument(
        "--result",
        default=None,
        help="Use an existing result JSON instead of re-analyzing (single-file only).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise SystemExit(f"Path not found: {args.input_path}")

    claim_sets = discover_claim_sets(args.input_path)
    total_docs = sum(len(cs.documents) for cs in claim_sets)
    is_single_file = (
        len(claim_sets) == 1
        and len(claim_sets[0].documents) == 1
        and claim_sets[0].source_dir is None
    )

    if args.result and not is_single_file:
        raise SystemExit("--result is only supported for a single input file.")
    if args.out and not is_single_file:
        raise SystemExit("--out is only supported for a single input file; use --out-dir.")

    out_root = os.path.abspath(args.out_dir) if args.out_dir else default_out_dir(args.input_path)

    print(
        f"Found {len(claim_sets)} claim set(s), {total_docs} document(s). "
        f"Output root: {out_root}"
    )

    for claim in claim_sets:
        claim_dir = ensure_dir(claim_output_dir(claim, out_root, args.input_path))
        print(f"\n--- Claim set: {claim.name} ({len(claim.documents)} doc(s)) → {claim_dir}")

        if args.result and is_single_file:
            with open(args.result, "r", encoding="utf-8") as fh:
                result = json.load(fh)
            print(f"Loaded existing result: {args.result}")
            _process_document(
                claim.documents[0], result, claim_dir, out_override=args.out
            )
            continue

        print(f"Analyzing fonts for {len(claim.documents)} document(s)…")
        batch = analyze_document(list(claim.documents), FONT_ONLY_OPTIONS)
        results = batch["results"]
        if len(results) != len(claim.documents):
            raise SystemExit(
                f"Expected {len(claim.documents)} results, got {len(results)}"
            )

        for document_path, result in zip(claim.documents, results):
            result_path = result_json_path(
                document_path, claim_dir, suffix="fonts result"
            )
            with open(result_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
            print(f"Wrote analysis: {result_path}")
            _process_document(
                document_path,
                result,
                claim_dir,
                out_override=args.out if is_single_file else None,
            )


if __name__ == "__main__":
    main()
