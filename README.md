# Visual Document Forensics MCP Server

A production-ready **Python MCP server** that performs **deterministic visual and
structural analysis** of PDF and DOCX documents.

It is designed to be called by a **UiPath agent**. The agent does the reasoning
and decision-making; this server does only one thing: extract **measurable
visual evidence** that *UiPath Analyze File* cannot reliably provide.

> The server **never** determines fraud. It reports metrics and *measurable
> anomalies* (blur, OCR confidence, effective DPI, image stretch, font usage,
> structural inconsistencies) and leaves all interpretation to the agent.

## Guarantees

All processing is **local and deterministic**. The server does **not** use:

- LLMs or VLMs
- External APIs or API keys
- Embeddings, vector databases, or retrieval systems
- Any cloud service or network access

The only optional external component is the local **Tesseract OCR binary**. If
it is absent, the server degrades gracefully (OCR metrics are skipped and a
warning is emitted) — it never fails because of it.

---

## Architecture

```text
Document
    ↓
MCP Tool Call            src/server/app.py        (FastMCP, stdio transport)
    ↓
Document Loader          src/render/loader.py     (detect PDF/DOCX, open handles)
    ↓
Page Renderer            src/render/pdf_renderer.py (PyMuPDF, configurable DPI)
    ↓
Tile Generator           src/tiling/tiler.py      (overlapping tiles)
    ↓
Visual Analysis Engine   src/analyzers/*          (blur, sharpness, contrast,
    ↓                                              entropy, edge density, noise,
    ↓                                              OCR confidence, text density)
PDF Structure Engine     src/analyzers/pdf_structure.py (images, scaling, DPI,
    ↓                                              fonts, objects; pikepdf check)
Evidence Aggregator      src/tools/analyze.py     (detectors + rollup)
    ↓
JSON Response            src/schemas/models.py    (pydantic-validated contract)
    ↓
UiPath Agent
```

### Project layout

```text
visual-forensics-mcp/
├── src/
│   ├── server/      # FastMCP server (analyze_document tool)
│   ├── tools/       # analysis orchestrator (the pipeline)
│   ├── render/      # document loader + PDF/DOCX rendering
│   ├── tiling/      # overlapping tile generation
│   ├── analyzers/   # deterministic metric computation
│   ├── detectors/   # anomaly detection from metrics
│   ├── schemas/     # pydantic output contract
│   └── utils/       # config, logging, geometry, image ops
├── tests/           # pytest suite (unit + end-to-end)
├── configs/         # default.yaml (all thresholds live here)
├── examples/        # sample docs + example request/response
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Analyzer & detector modules

| Analyzers (`src/analyzers/`) | Detectors (`src/detectors/`) |
| --- | --- |
| `blur.py` — Laplacian variance | `blur_detector.py` — blur anomaly |
| `sharpness.py` — Tenengrad / gradient | `dpi_detector.py` — resolution anomaly |
| `contrast.py` — RMS / Michelson | `font_detector.py` — font mismatch |
| `entropy.py` — Shannon entropy | `font_outlier_detector.py` — located non-dominant-font text |
| `edge_density.py` — Canny edge ratio | `stretch_detector.py` — non-uniform scaling |
| `noise.py` — Immerkaer sigma / residual | `ocr_confidence_detector.py` — low OCR confidence |
| `ocr.py` — Tesseract confidence | `compression_detector.py` — compression artifact |
| `font_analysis.py` — font families/sizes/span locations | `structure_detector.py` — raster-in-vector / overlap |
| `image_scaling.py` — scale/effective DPI | |
| `pdf_structure.py` — images/objects/fonts | |

---

## Installation

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Tesseract OCR (optional but recommended)

OCR-based metrics (`ocr_confidence`) and the OCR-confidence detector require the
Tesseract binary. The Python wheel (`pytesseract`) only wraps it.

- **Windows:** install the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki),
  then either add it to `PATH` or set `ocr.tesseract_cmd` in `configs/default.yaml`
  (or via request options) to e.g. `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`.
- **macOS:** `brew install tesseract`
- **Debian/Ubuntu:** `sudo apt-get install tesseract-ocr`

Without Tesseract, everything else still runs; the response simply includes a
warning and `summary.ocr_available = false`.

---

## Configuration

All tunables — including **every threshold** — live in
[`configs/default.yaml`](configs/default.yaml). There are **no hardcoded
thresholds** anywhere in the analysis or detection code.

Key settings:

```yaml
render:
  dpi: 400                 # default render resolution
tiling:
  tile_size: 512
  tile_overlap: 0.20
features:
  enable_ocr: true
  enable_pdf_analysis: true
```

Three ways to configure, in increasing precedence:

1. Edit `configs/default.yaml`.
2. Point the server at another file via the `VISUAL_FORENSICS_CONFIG`
   environment variable.
3. Pass an `options` dict per request (deep-merged over the file), e.g.
   `{"render": {"dpi": 300}, "features": {"enable_ocr": false}}`.

---

## Running the MCP server

```bash
python -m src.server.app
```

This starts a **FastMCP** server on the **stdio** transport — the standard way
MCP clients (including UiPath) launch and communicate with a local server.

A console script is also installed:

```bash
visual-forensics-mcp
```

---

## MCP usage

The server exposes a single primary tool:

```python
analyze_document(document_paths: list[str], options: dict | None = None)
```

It returns a JSON object with a `results` array. Each element has this shape:

```json
{
  "results": [
    {
      "document_id": "...",
      "document_type": "...",
      "summary": {},
      "page_results": [],
      "document_findings": [],
      "warnings": [],
      "errors": []
    }
  ]
}
```

Every **finding** (in `results[].page_results[].findings` and
`results[].document_findings`) has the required schema:

```json
{
  "type": "...",
  "page": 0,
  "bbox": [0, 0, 0, 0],
  "metrics": {},
  "confidence": 0.0,
  "explanation": "..."
}
```

`bbox` values are in **rendered pixel coordinates** at the page's render DPI.
Document-level findings (e.g. `font_mismatch`) use `page: 0` and a zero bbox.

### Finding types produced

| Type | Trigger (configurable) |
| --- | --- |
| `blur_anomaly` | Tile sharpness far below the page distribution |
| `ocr_confidence_anomaly` | Tile OCR confidence far below page average |
| `resolution_anomaly` | Embedded image effective DPI far below render DPI |
| `image_stretch` | Embedded image scaled non-uniformly (`scale_x` ≠ `scale_y`) |
| `compression_artifact_anomaly` | Tile noise signature unlike its neighbours |
| `font_mismatch` | Rare fonts / too many font families (document-level) |
| `font_outlier` | Text set in a font other than the document's dominant font, with an exact bounding box, the font name, and a confidence |
| `raster_in_vector_anomaly` | Raster region inside an otherwise vector page |
| `object_overlap_anomaly` | Strongly overlapping / stacked images |

---

## UiPath integration

UiPath agents can call MCP tools directly. Configure this server as a **local
stdio MCP server** in your UiPath agent / MCP client configuration:

```jsonc
{
  "mcpServers": {
    "visual-forensics": {
      "command": "C:/path/to/visual-forensics-mcp/.venv/Scripts/python.exe",
      "args": ["-m", "src.server.app"],
      "cwd": "C:/path/to/visual-forensics-mcp",
      "env": {
        // optional: point at a custom config
        "VISUAL_FORENSICS_CONFIG": "C:/path/to/custom.yaml"
      }
    }
  }
}
```

Typical agent flow:

1. UiPath downloads / locates the document(s) and resolves local file paths.
2. The agent calls `analyze_document` with those paths (and optional `options`).
3. The agent reads each result's `summary`, `page_results[].findings`, and
   `document_findings`, and applies its own business rules / human-in-the-loop
   logic to decide what to do next.

Because the server is deterministic and offline, the same document always yields
the same evidence — ideal for auditable, repeatable RPA workflows.

### Example request

```json
{
  "tool": "analyze_document",
  "arguments": {
    "document_paths": ["C:/docs/invoice.pdf", "C:/docs/receipt.pdf"],
    "options": { "render": { "dpi": 200 }, "features": { "enable_ocr": true } }
  }
}
```

### Example response (abridged)

For the bundled `examples/sample.pdf` (a text page with a stretched, low-resolution
raster insert and one rare-font line):

```json
{
  "results": [
    {
      "document_id": "5f34550e05450828",
      "document_type": "pdf",
      "summary": {
        "page_count": 1,
        "tile_count": 20,
        "finding_count": 4,
        "findings_by_type": {
          "resolution_anomaly": 1,
          "image_stretch": 1,
          "raster_in_vector_anomaly": 1,
          "font_mismatch": 1
        },
        "ocr_available": false,
        "pdf_structure_available": true
      },
      "page_results": [
        {
          "page": 1,
          "width": 1700,
          "height": 2200,
          "dpi": 200.0,
          "is_vector": true,
          "findings": [
            {
              "type": "image_stretch",
              "page": 1,
              "bbox": [888.89, 833.33, 1444.44, 1166.67],
              "metrics": { "scale_x": 5.0, "scale_y": 3.0, "stretch_ratio": 0.4, "xref": 32 },
              "confidence": 0.4,
              "explanation": "Embedded image is scaled non-uniformly (horizontal and vertical scale factors differ), distorting the image."
            }
          ]
        }
      ],
      "document_findings": [
        {
          "type": "font_mismatch",
          "page": 0,
          "bbox": [0, 0, 0, 0],
          "metrics": { "font": "Times-Roman", "span_count": 1, "share": 0.04 },
          "confidence": 0.6,
          "explanation": "A font family is used on only a small fraction of text spans, an unusual change versus the dominant fonts."
        }
      ],
      "warnings": ["Tesseract OCR binary not available; OCR metrics and the OCR-confidence detector are disabled for this run."],
      "errors": []
    }
  ]
}
```

Regenerate the full sample with:

```bash
python -m examples.generate_examples
```

---

## Font consistency agent (`font_agent.py`)

A standalone command-line agent that answers one question: **where does this
document depart from its own font?**

```bash
python font_agent.py "path/to/document.pdf"
```

It determines the document's **dominant font** (the most common family by text
span count) and produces `"<document> - fonts annotated.pdf"` in which:

- a banner at the **top of every page** states the dominant font, plus every
  other font detected with its usage share, detection confidence, and the
  pages it appears on;
- every region of text set in a non-dominant font gets a **crimson bounding
  box exactly where it sits on the page**, labelled with the font name and the
  detection confidence;
- the cover page carries the same font summary for the whole document.

The console output mirrors the annotation — dominant font, every other font,
and each boxed region with its page, confidence, pixel bbox, and text snippet:

```text
=== Font Report ===
Dominant font : Calibri  (37.2% of text, 16 spans)
Other fonts   :
  - TimesLTPro-Bold           23.3% of text   confidence 0.62   pages 1, 2
  - Alegreya-Regular           2.3% of text   confidence 0.94   pages 1
Boxed regions : 20
  - page 1  font 'Alegreya-Regular'  conf 0.94  bbox [810, 1755, 1205, 1831] px  text: 'NH-2026-001847'
  ...
```

Only the font detectors run (visual/OCR/image analysis is switched off), so the
agent is fast. The same evidence is available programmatically: the pipeline
emits `font_outlier` findings in `page_results[].findings`, each carrying
`bbox`, `confidence`, and `metrics.font` / `metrics.dominant_font`. Confidence
is deterministic: `dominant_spans / (dominant_spans + font_spans)`, so a single
inserted line in a foreign font scores near 1.0. Thresholds live under
`detectors.font_outlier` in `configs/default.yaml`; the banner and badge are
toggled via `report.draw.add_font_banner` / `report.draw.add_risk_badge`.

DOCX inputs work too: they are converted to PDF in memory for analysis, and
the annotated copy is written from that rendition. Note that PyMuPDF's DOCX
conversion may substitute font families, so span-level font locations are
exact for PDF inputs but approximate for DOCX (document-level font usage from
python-docx still feeds the `font_mismatch` detector).

The general-purpose `annotate_report.py` (all finding types) also draws the
font banner and the compact font-outlier boxes by default.

---

## Testing

```bash
pytest -q
```

The suite covers PDF rendering, DOCX loading, tile generation, blur computation,
OCR confidence extraction (graceful path), image-scaling detection, font
extraction, font-outlier location and the font agent, schema validation, MCP
invocation, and a complete end-to-end run for PDF, DOCX, and scanned/image-only
PDF.

---

## How metrics are computed (deterministic definitions)

- **Blur** — variance of the Laplacian (lower ⇒ blurrier).
- **Sharpness** — Tenengrad (mean squared Sobel gradient) and mean gradient.
- **Contrast** — RMS (intensity std) and Michelson `(max−min)/(max+min)`.
- **Entropy** — Shannon entropy (bits) of the 256-bin intensity histogram.
- **Edge density** — fraction of Canny edge pixels.
- **Noise** — Immerkaer fast σ estimate and median-residual std.
- **OCR confidence** — mean Tesseract word confidence, normalised to 0–1.
- **Text density** — foreground fraction after adaptive binarisation.
- **Effective DPI** — `original_px × 72 / displayed_points` per axis.
- **Scale / stretch** — `displayed_points / original_px` per axis; stretch is
  `|scale_x − scale_y| / max(scale_x, scale_y)`.

Anomalies are flagged using per-page z-scores and/or absolute floors, all read
from `configs/default.yaml`. Confidence is a monotonic function of the measured
deviation, not a probability of fraud.
```
