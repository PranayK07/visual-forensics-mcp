"""Generate example documents and a sample analysis response.

Run from the project root:

    python -m examples.generate_examples

Produces, under ``examples/``:

* ``sample.pdf``           -- native PDF with a stretched, low-res raster insert
* ``sample.docx``          -- DOCX with a rare-font run and an embedded image
* ``sample_response.json`` -- the analysis result for ``sample.pdf``
* ``example_request.json`` -- a sample MCP tool-call payload
"""

from __future__ import annotations

import json
import os
import sys

# Allow running as a plain script as well as a module.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.builders import build_sample_docx, build_sample_pdf  # noqa: E402
from src.tools.analyze import analyze_document  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    pdf_path = os.path.join(HERE, "sample.pdf")
    docx_path = os.path.join(HERE, "sample.docx")
    build_sample_pdf(pdf_path)
    build_sample_docx(docx_path)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {docx_path}")

    options = {"render": {"dpi": 200}, "features": {"enable_ocr": True}}
    batch = analyze_document([pdf_path], options)
    result = batch["results"][0]

    response_path = os.path.join(HERE, "sample_response.json")
    with open(response_path, "w", encoding="utf-8") as fh:
        json.dump(batch, fh, indent=2, allow_nan=False)
    print(f"Wrote {response_path}")

    request = {
        "tool": "analyze_document",
        "arguments": {
            "document_paths": ["examples/sample.pdf"],
            "options": options,
        },
    }
    request_path = os.path.join(HERE, "example_request.json")
    with open(request_path, "w", encoding="utf-8") as fh:
        json.dump(request, fh, indent=2)
    print(f"Wrote {request_path}")

    print(
        f"\nSummary: {result['summary']['finding_count']} findings "
        f"across {result['summary']['page_count']} page(s); "
        f"types={result['summary']['findings_by_type']}"
    )


if __name__ == "__main__":
    main()
