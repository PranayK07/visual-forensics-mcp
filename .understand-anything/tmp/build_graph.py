"""Build knowledge-graph batch files from AST + import map (deterministic)."""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTER = ROOT / ".understand-anything" / "intermediate"

COMPLEXITY = {
    "simple": lambda n: n < 80,
    "moderate": lambda n: 80 <= n < 200,
    "complex": lambda n: n >= 200,
}


def complexity_for(lines: int) -> str:
    if lines < 80:
        return "simple"
    if lines < 200:
        return "moderate"
    return "complex"


def node_id_for_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".yaml", ".yml", ".json", ".toml"}:
        return f"config:{path}"
    if ext in {".md", ".txt"}:
        return f"document:{path}"
    return f"file:{path}"


def summarize_file(path: str, text: str) -> str:
    first = ""
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith('"""'):
            first = s[:120]
            break
    if "server" in path:
        return "FastMCP server entry exposing the analyze_document tool over stdio/SSE."
    if "analyze.py" in path:
        return "Orchestrates the full deterministic forensic pipeline from load to JSON response."
    if "annotator" in path:
        return "Draws bounding boxes and fraud-risk badges on annotated PDF copies."
    if "detector" in path:
        return "Flags measurable visual/structural anomalies using configurable thresholds."
    if "analyzers" in path:
        return "Computes deterministic per-tile or per-page visual/structure metrics."
    if path.startswith("tests/"):
        return f"Automated test module validating pipeline behavior ({Path(path).name})."
    if path == "README.md":
        return "Project documentation: architecture, installation, MCP usage, UiPath integration."
    if path == "configs/default.yaml":
        return "Central YAML configuration for DPI, tiling, analyzers, detectors, and report output."
    return first or f"Module {path}"


def tags_for(path: str) -> list[str]:
    tags = []
    if path.startswith("src/server"):
        tags += ["mcp", "entry-point", "fastmcp"]
    if path.startswith("src/tools"):
        tags += ["orchestrator", "pipeline"]
    if path.startswith("src/analyzers"):
        tags += ["analyzer", "metrics"]
    if path.startswith("src/detectors"):
        tags += ["detector", "anomaly"]
    if path.startswith("src/render"):
        tags += ["render", "loader"]
    if path.startswith("src/report"):
        tags += ["report", "pdf-annotation"]
    if path.startswith("src/schemas"):
        tags += ["schema", "pydantic"]
    if path.startswith("tests/"):
        tags += ["test", "pytest"]
    if path.endswith(".yaml"):
        tags += ["config", "yaml"]
    if path.endswith(".md"):
        tags += ["docs"]
    return tags or ["module"]


def parse_python(path: Path) -> tuple[list[dict], list[dict]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], []
    funcs, classes = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_") and node.name not in {"__init__"}:
                continue
            funcs.append(
                {
                    "name": node.name,
                    "startLine": node.lineno,
                    "endLine": getattr(node, "end_lineno", node.lineno),
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
            ]
            classes.append(
                {
                    "name": node.name,
                    "startLine": node.lineno,
                    "endLine": getattr(node, "end_lineno", node.lineno),
                    "methods": methods,
                }
            )
    return funcs, classes


def build_batch(batch: dict, import_map: dict) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    rel = ROOT

    for f in batch["files"]:
        path = f["path"]
        abs_path = rel / path
        lines = f.get("sizeLines", 0)
        cat = f.get("fileCategory", "code")
        lang = f.get("language", "unknown")
        nid = node_id_for_path(path)
        text = abs_path.read_text(encoding="utf-8", errors="replace") if abs_path.exists() else ""

        ntype = "file"
        if cat == "config":
            ntype = "config"
        elif cat == "docs":
            ntype = "document"

        nodes.append(
            {
                "id": nid,
                "type": ntype,
                "name": Path(path).name,
                "filePath": path,
                "summary": summarize_file(path, text),
                "tags": tags_for(path),
                "complexity": complexity_for(lines),
                "language": lang if lang != "unknown" else None,
            }
        )

        if path.endswith(".py") and abs_path.exists():
            funcs, classes = parse_python(abs_path)
            for fn in funcs:
                fid = f"function:{path}:{fn['name']}"
                nodes.append(
                    {
                        "id": fid,
                        "type": "function",
                        "name": fn["name"],
                        "filePath": path,
                        "summary": f"Function `{fn['name']}` in {path}.",
                        "tags": tags_for(path) + ["function"],
                        "complexity": "simple",
                        "startLine": fn["startLine"],
                        "endLine": fn["endLine"],
                    }
                )
                edges.append(
                    {
                        "source": nid,
                        "target": fid,
                        "type": "contains",
                        "weight": 1.0,
                    }
                )
            for cls in classes:
                cid = f"class:{path}:{cls['name']}"
                nodes.append(
                    {
                        "id": cid,
                        "type": "class",
                        "name": cls["name"],
                        "filePath": path,
                        "summary": f"Class `{cls['name']}` in {path}.",
                        "tags": tags_for(path) + ["class"],
                        "complexity": "moderate",
                        "startLine": cls["startLine"],
                        "endLine": cls["endLine"],
                    }
                )
                edges.append(
                    {"source": nid, "target": cid, "type": "contains", "weight": 1.0}
                )

        for target in import_map.get(path, []):
            edges.append(
                {
                    "source": nid,
                    "target": node_id_for_path(target),
                    "type": "imports",
                    "weight": 0.7,
                }
            )

    return {"nodes": nodes, "edges": edges, "batchIndex": batch["batchIndex"]}


def main() -> None:
    batches_data = json.loads((INTER / "batches.json").read_text(encoding="utf-8"))
    scan = json.loads((INTER / "scan-result.json").read_text(encoding="utf-8"))
    import_map = scan.get("importMap", {})

    for batch in batches_data["batches"]:
        idx = batch["batchIndex"]
        out = build_batch(batch, import_map)
        (INTER / f"batch-{idx}.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        print(f"wrote batch-{idx}.json nodes={len(out['nodes'])} edges={len(out['edges'])}")


if __name__ == "__main__":
    main()
