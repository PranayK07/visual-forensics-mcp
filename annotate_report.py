"""Analyze claim evidence and write one factual result bundle per claim set.

Accepts a single file, a claim-set folder, or a claims root whose subfolders
are claim sets. Outputs mirror claim-set folders under ``--out-dir``.

Usage:
    python annotate_report.py "path/to/document.pdf"
    python annotate_report.py "path/to/claim_set/"
    python annotate_report.py "path/to/claims_root/" --out-dir "path/to/out/"
    python annotate_report.py "path/to/document.pdf" --out "path/to/out.pdf"
    python annotate_report.py "path/to/document.pdf" --result existing_result.json

Produces (per claim set):
    report.md
    json_results/claim_result.json
    json_results/<stem> - result.json
    annotated_visuals/<stem> - annotated.pdf
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

from src.render import convert_to_pdf_path, detect_type  # noqa: E402
from src.report.annotator import annotate_document  # noqa: E402
from src.report.markdown_report import write_markdown_report  # noqa: E402
from src.report.statistics import aggregate_claim_statistics  # noqa: E402
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
    """Return an annotatable PDF path and an optional temporary path."""
    if detect_type(document_path) == "pdf":
        return document_path, None

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="annotate_converted_")
    os.close(fd)
    try:
        convert_to_pdf_path(document_path, tmp_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
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
    print(f"Findings   : {result.get('summary', {}).get('findings_by_type', {})}")
    print(f"Annotated  : {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze PDF, DOCX, and raster-image evidence and write factual "
            "JSON, one claim-level statistics report, and annotated copies. "
            "Pass a file, a claim-set folder, or a parent folder of claim sets."
        )
    )
    parser.add_argument(
        "input_path",
        help=(
            "Path to a supported evidence file, a claim-set folder, "
            "or a claims root whose subfolders are claim sets."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Result-bundle directory. Defaults to '<input-name>_result' beside "
            "the input file, claim set, or claims root."
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
                loaded_result = json.load(fh)
            if isinstance(loaded_result, dict) and "results" in loaded_result:
                result = loaded_result["results"][0]
            else:
                result = loaded_result
            print(f"Loaded existing result: {args.result}")
            statistics = aggregate_claim_statistics(
                [result], document_labels=[os.path.basename(claim.documents[0])]
            )
            result["statistics"] = statistics["documents"][0]
            batch = {
                "schema_version": "2.0",
                "statistics": statistics,
                "results": [result],
            }
        else:
            print(f"Analyzing {len(claim.documents)} document(s)…")
            batch = analyze_document(list(claim.documents), options or None)

        results = batch["results"]
        if len(results) != len(claim.documents):
            raise SystemExit(
                f"Expected {len(claim.documents)} results, got {len(results)}"
            )

        report_path = write_markdown_report(
            batch["statistics"],
            os.path.join(claim_dir, "report.md"),
            claim_name=claim.name,
        )
        print(f"Wrote claim statistics: {report_path}")

        json_dir = ensure_dir(os.path.join(claim_dir, "json_results"))
        annotated_dir = ensure_dir(os.path.join(claim_dir, "annotated_visuals"))
        claim_json_path = os.path.join(json_dir, "claim_result.json")
        with open(claim_json_path, "w", encoding="utf-8") as fh:
            json.dump(batch, fh, indent=2, allow_nan=False)
        print(f"Wrote claim JSON: {claim_json_path}")

        for document_path, result in zip(claim.documents, results):
            result_path = result_json_path(document_path, json_dir)
            with open(result_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, allow_nan=False)
            print(f"Wrote analysis: {result_path}")
            _process_document(
                document_path,
                result,
                annotated_dir,
                config,
                out_override=args.out if is_single_file else None,
            )


if __name__ == "__main__":
    main()
