"""Analyze PDF/DOCX documents and produce annotated, marked-up copies.

Accepts a single file, a claim-set folder, or a claims root whose subfolders
are claim sets. Outputs mirror claim-set folders under ``--out-dir``.

Usage:
    python annotate_report.py "path/to/document.pdf"
    python annotate_report.py "path/to/claim_set/"
    python annotate_report.py "path/to/claims_root/" --out-dir "path/to/out/"
    python annotate_report.py "path/to/document.pdf" --out "path/to/out.pdf"
    python annotate_report.py "path/to/document.pdf" --result existing_result.json

Produces (per document):
    "<stem> - annotated.pdf"
    "<stem> - result.json"   (unless --result was given for a single file)
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

from src.report.annotator import annotate_document, compute_fraud_risk  # noqa: E402
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


def _convert_to_pdf(document_path: str) -> tuple[str, str | None]:
    """Return (pdf_path, temp_path_to_cleanup). DOCX is converted in a temp file."""
    if os.path.splitext(document_path)[1].lower() == ".pdf":
        return document_path, None

    import fitz  # PyMuPDF

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="annotate_converted_")
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
    config,
    out_override: str | None = None,
) -> str:
    """Annotate one document; write PDF (and optionally reuse existing result)."""
    if result.get("errors"):
        print(f"WARNING: {os.path.basename(document_path)} errors:", result["errors"])

    risk = compute_fraud_risk(result, config)
    out = out_override or annotated_pdf_path(document_path, output_dir)

    input_pdf, tmp_converted = _convert_to_pdf(document_path)
    try:
        if tmp_converted:
            print(f"Converted to PDF for annotation: {document_path}")
        out = annotate_document(input_pdf, result, out, config)
    finally:
        if tmp_converted:
            try:
                os.remove(tmp_converted)
            except OSError:
                pass

    print(f"\n=== {os.path.basename(document_path)} ===")
    print(f"Fraud risk : {risk['level']} (score {risk['score']}/100)")
    print(f"Findings   : {result.get('summary', {}).get('findings_by_type', {})}")
    print(f"Annotated  : {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate PDF/DOCX documents with forensic findings. "
            "Pass a file, a claim-set folder, or a parent folder of claim sets."
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
    parser.add_argument("--dpi", type=int, default=None, help="Override render DPI.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise SystemExit(f"Path not found: {args.input_path}")

    options: dict = {}
    if args.dpi:
        options["render"] = {"dpi": args.dpi}

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
    config = load_config(overrides=options or None)

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
                claim.documents[0], result, claim_dir, config, out_override=args.out
            )
            continue

        print(f"Analyzing {len(claim.documents)} document(s)…")
        batch = analyze_document(list(claim.documents), options or None)
        results = batch["results"]
        if len(results) != len(claim.documents):
            raise SystemExit(
                f"Expected {len(claim.documents)} results, got {len(results)}"
            )

        for document_path, result in zip(claim.documents, results):
            result_path = result_json_path(document_path, claim_dir)
            with open(result_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
            print(f"Wrote analysis: {result_path}")
            _process_document(
                document_path,
                result,
                claim_dir,
                config,
                out_override=args.out if is_single_file else None,
            )


if __name__ == "__main__":
    main()
