"""Analyze a PDF and produce an annotated, marked-up copy.

Usage:
    python annotate_report.py "path\\to\\document.pdf"
    python annotate_report.py "path\\to\\document.pdf" --out "path\\to\\out.pdf"
    python annotate_report.py "path\\to\\document.pdf" --result existing_result.json

Produces:
    "<document> - annotated.pdf"   (a copy of the input, marked up)
    "<document> - result.json"     (the raw analysis, unless --result was given)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.report.annotator import annotate_document, compute_fraud_risk  # noqa: E402
from src.tools.analyze import analyze_document  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a PDF with forensic findings.")
    parser.add_argument("document", help="Path to the input PDF.")
    parser.add_argument("--out", default=None, help="Output annotated PDF path.")
    parser.add_argument("--result", default=None,
                        help="Use an existing result JSON instead of re-analyzing.")
    parser.add_argument("--dpi", type=int, default=None, help="Override render DPI.")
    args = parser.parse_args()

    if not os.path.exists(args.document):
        raise SystemExit(f"File not found: {args.document}")

    options = {}
    if args.dpi:
        options["render"] = {"dpi": args.dpi}

    if args.result:
        with open(args.result, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        print(f"Loaded existing result: {args.result}")
    else:
        print(f"Analyzing: {args.document}")
        batch = analyze_document([args.document], options or None)
        result = batch["results"][0]
        result_path = os.path.splitext(args.document)[0] + " - result.json"
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote analysis: {result_path}")

    if result.get("errors"):
        print("WARNING: analysis reported errors:", result["errors"])

    config = load_config(overrides=options or None)
    risk = compute_fraud_risk(result, config)
    out = annotate_document(args.document, result, args.out, config)

    print("\n=== Summary ===")
    print(f"Fraud risk : {risk['level']} (score {risk['score']}/100)")
    print(f"Findings   : {result.get('summary', {}).get('findings_by_type', {})}")
    print(f"Annotated  : {out}")


if __name__ == "__main__":
    main()
