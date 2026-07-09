"""FastMCP server exposing the ``analyze_document`` tool.

Run with stdio transport (the default), which is what UiPath and other MCP
clients use to launch and talk to a local server:

    python -m src.server.app

All work is local and deterministic. No network access is performed.
"""

from __future__ import annotations

from typing import Any, Optional

from fastmcp import FastMCP

from ..tools.analyze import analyze_document as _analyze_document
from ..utils.logging import get_logger

logger = get_logger("server")

mcp = FastMCP(
    name="visual-forensics-mcp",
    instructions=(
        "Deterministic visual and structural forensics for PDF and DOCX "
        "documents. Call `analyze_document` with one or more local file paths "
        "to obtain measurable visual evidence (blur, OCR confidence, effective "
        "DPI, image stretch, font usage, structural anomalies). This server "
        "never makes fraud judgements; it returns metrics and measurable "
        "anomalies for an agent to reason over."
    ),
)


@mcp.tool(
    name="analyze_document",
    description=(
        "Analyze one or more local PDF or DOCX files and return deterministic "
        "visual and structural forensic evidence as JSON. `document_paths` is "
        "an array of absolute or relative paths on the machine running this "
        "server. `options` optionally overrides configuration (e.g. "
        "{'render': {'dpi': 300}, 'features': {'enable_ocr': false}}). Returns "
        "a `results` array; each element has document_id, document_type, "
        "summary, page_results, document_findings, warnings, and errors."
    ),
)
def analyze_document(
    document_paths: list[str] | str,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """MCP tool entry point. Delegates to the deterministic pipeline."""
    if isinstance(document_paths, str):
        document_paths = [document_paths]
    if not isinstance(document_paths, list) or not all(
        isinstance(path, str) for path in document_paths
    ):
        raise TypeError("document_paths must be a list of file path strings")
    logger.info("analyze_document called: %d document(s)", len(document_paths))
    return _analyze_document(document_paths, options)


def build_server() -> FastMCP:
    """Return the configured FastMCP instance (useful for tests/embedding)."""
    return mcp


def main() -> None:
    """Console entry point: run the server over stdio."""
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)



if __name__ == "__main__":
    main()
