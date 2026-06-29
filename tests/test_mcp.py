"""MCP invocation tests using an in-memory FastMCP client."""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client

from src.server.app import mcp


def _extract_payload(call_result):
    """Pull the structured dict out of a FastMCP CallToolResult."""
    data = getattr(call_result, "data", None)
    if isinstance(data, dict):
        return data
    structured = getattr(call_result, "structured_content", None)
    if isinstance(structured, dict):
        # FastMCP may wrap non-dict returns under "result"; ours is a dict.
        return structured.get("result", structured)
    content = getattr(call_result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            return json.loads(text)
    raise AssertionError("Could not extract payload from tool result")


def test_tool_registered():
    async def _run():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            return [t.name for t in tools]

    names = asyncio.run(_run())
    assert "analyze_document" in names


def test_mcp_analyze_document_invocation(sample_pdf, fast_options):
    async def _run():
        async with Client(mcp) as client:
            result = await client.call_tool(
                "analyze_document",
                {"document_path": sample_pdf, "options": fast_options},
            )
            return _extract_payload(result)

    payload = asyncio.run(_run())
    assert payload["document_type"] == "pdf"
    assert payload["document_id"]
    assert payload["summary"]["page_count"] >= 1
    assert "page_results" in payload
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["errors"], list)
    assert payload["errors"] == []
