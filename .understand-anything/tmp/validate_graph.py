import json
from pathlib import Path

graph = json.loads(Path(".understand-anything/knowledge-graph.json").read_text(encoding="utf-8"))
issues, warnings = [], []
node_ids = {n["id"] for n in graph.get("nodes", []) if n.get("id")}
for e in graph.get("edges", []):
    if e["source"] not in node_ids:
        issues.append(f"edge source {e['source']}")
    if e["target"] not in node_ids:
        issues.append(f"edge target {e['target']}")
file_level = {
    n["id"] for n in graph["nodes"] if n["type"] in ("file", "config", "document")
}
assigned = set()
for layer in graph.get("layers", []):
    for nid in layer.get("nodeIds", []):
        if nid not in node_ids:
            issues.append(f"layer {layer['id']} missing {nid}")
        assigned.add(nid)
for nid in file_level:
    if nid not in assigned:
        issues.append(f"unassigned file {nid}")
review = {
    "issues": issues,
    "warnings": warnings,
    "stats": {
        "totalNodes": len(graph["nodes"]),
        "totalEdges": len(graph["edges"]),
        "totalLayers": len(graph["layers"]),
        "tourSteps": len(graph["tour"]),
    },
}
Path(".understand-anything/intermediate/review.json").write_text(
    json.dumps(review, indent=2), encoding="utf-8"
)
print("issues", len(issues))
