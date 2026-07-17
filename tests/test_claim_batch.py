"""Tests for claim-set folder discovery and batch CLI output layout."""

from __future__ import annotations

import os
import shutil
import json

import fitz
from PIL import Image

import annotate_report
import font_agent
from src.utils.claim_batch import (
    ClaimSet,
    annotated_pdf_path,
    claim_output_dir,
    default_out_dir,
    discover_claim_sets,
    list_documents_in_dir,
    result_json_path,
)


def test_discover_single_file(sample_pdf):
    sets = discover_claim_sets(sample_pdf)
    assert len(sets) == 1
    assert sets[0].source_dir is None
    assert sets[0].documents == (os.path.abspath(sample_pdf),)


def test_discover_claim_set_folder(sample_pdf, sample_docx, tmp_path):
    claim = tmp_path / "claim_001"
    claim.mkdir()
    shutil.copy(sample_pdf, claim / "invoice.pdf")
    shutil.copy(sample_docx, claim / "notes.docx")
    (claim / "readme.txt").write_text("ignore me")

    sets = discover_claim_sets(str(claim))
    assert len(sets) == 1
    assert sets[0].name == "claim_001"
    assert len(sets[0].documents) == 2
    assert all(os.path.basename(p) in {"invoice.pdf", "notes.docx"} for p in sets[0].documents)


def test_discover_claims_root(sample_pdf, sample_docx, tmp_path):
    root = tmp_path / "claims"
    a = root / "claim_a"
    b = root / "claim_b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    shutil.copy(sample_pdf, a / "a.pdf")
    shutil.copy(sample_docx, b / "b.docx")
    (root / "empty").mkdir()
    (root / "noise.txt").write_text("skip")

    sets = discover_claim_sets(str(root))
    names = {s.name for s in sets}
    assert names == {"claim_a", "claim_b"}
    assert all(len(s.documents) == 1 for s in sets)


def test_claim_output_dir_nesting(tmp_path):
    root = str(tmp_path / "claims")
    claim = ClaimSet(
        name="claim_a",
        documents=(str(tmp_path / "claims" / "claim_a" / "a.pdf"),),
        source_dir=str(tmp_path / "claims" / "claim_a"),
    )
    out = claim_output_dir(claim, str(tmp_path / "out"), root)
    assert out == str(tmp_path / "out" / "claim_a")


def test_default_out_dir_for_folder(tmp_path):
    claim = tmp_path / "claim_001"
    claim.mkdir()
    assert default_out_dir(str(claim)) == str(claim) + "_result"


def test_default_out_dir_for_file(sample_pdf):
    expected = os.path.join(
        os.path.dirname(sample_pdf),
        f"{os.path.splitext(os.path.basename(sample_pdf))[0]}_result",
    )
    assert default_out_dir(sample_pdf) == expected


def test_list_documents_ignores_unsupported(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.txt").write_text("x")
    docs = list_documents_in_dir(str(tmp_path))
    assert [os.path.basename(p) for p in docs] == ["a.pdf"]


def test_path_helpers():
    assert annotated_pdf_path(r"C:\docs\inv.pdf", r"C:\out") == os.path.join(
        r"C:\out", "inv - annotated.pdf"
    )
    assert result_json_path(r"C:\docs\inv.pdf", r"C:\out", "fonts result") == os.path.join(
        r"C:\out", "inv - fonts result.json"
    )


def test_annotate_report_claim_set_folder(sample_pdf, tmp_path, monkeypatch, capsys):
    claim = tmp_path / "claim_001"
    claim.mkdir()
    shutil.copy(sample_pdf, claim / "doc1.pdf")
    shutil.copy(sample_pdf, claim / "doc2.pdf")
    out_dir = tmp_path / "annotated_out"

    monkeypatch.setattr(
        "sys.argv",
        ["annotate_report.py", str(claim), "--out-dir", str(out_dir), "--dpi", "72"],
    )
    annotate_report.main()

    # One claim bundle: report plus exactly the two requested artifact folders.
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "annotated_visuals" / "doc1 - annotated.pdf").is_file()
    assert (out_dir / "annotated_visuals" / "doc2 - annotated.pdf").is_file()
    assert (out_dir / "json_results" / "doc1 - result.json").is_file()
    assert (out_dir / "json_results" / "doc2 - result.json").is_file()
    claim_json = out_dir / "json_results" / "claim_result.json"
    assert claim_json.is_file()
    payload = json.loads(claim_json.read_text())
    assert payload["schema_version"] == "2.0"
    assert payload["statistics"]["document_count"] == 2
    assert "tile.blur_score" in payload["statistics"]["overall_metrics"]
    report = (out_dir / "report.md").read_text()
    assert "## Overall claim-set statistics" in report
    assert "Mean" in report and "Median" in report and "Mode" in report
    assert "<svg" in report

    # Annotated visuals contain only the source pages—no repeated cover report.
    with fitz.open(sample_pdf) as source, fitz.open(
        out_dir / "annotated_visuals" / "doc1 - annotated.pdf"
    ) as annotated:
        assert annotated.page_count == source.page_count
    console = capsys.readouterr().out
    assert "1 claim set" in console


def test_annotate_report_claims_root(sample_pdf, tmp_path, monkeypatch):
    root = tmp_path / "claims"
    (root / "c1").mkdir(parents=True)
    (root / "c2").mkdir(parents=True)
    shutil.copy(sample_pdf, root / "c1" / "a.pdf")
    shutil.copy(sample_pdf, root / "c2" / "b.pdf")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        ["annotate_report.py", str(root), "--out-dir", str(out_dir), "--dpi", "72"],
    )
    annotate_report.main()

    for claim_name, stem in (("c1", "a"), ("c2", "b")):
        claim_dir = out_dir / claim_name
        assert (claim_dir / "report.md").is_file()
        assert (claim_dir / "annotated_visuals" / f"{stem} - annotated.pdf").is_file()
        assert (claim_dir / "json_results" / f"{stem} - result.json").is_file()
        assert (claim_dir / "json_results" / "claim_result.json").is_file()


def test_annotate_report_mixed_pdf_and_webp_bundle(
    sample_pdf, tmp_path, monkeypatch
):
    claim = tmp_path / "mixed_claim"
    claim.mkdir()
    shutil.copy(sample_pdf, claim / "document.pdf")
    Image.new("RGB", (640, 480), "white").save(claim / "photo.webp", "WEBP")

    monkeypatch.setattr(
        "sys.argv",
        ["annotate_report.py", str(claim), "--dpi", "72"],
    )
    annotate_report.main()

    result_dir = tmp_path / "mixed_claim_result"
    assert (result_dir / "report.md").is_file()
    assert (result_dir / "json_results" / "document - result.json").is_file()
    assert (result_dir / "json_results" / "photo - result.json").is_file()
    assert (result_dir / "annotated_visuals" / "document - annotated.pdf").is_file()
    image_annotation = result_dir / "annotated_visuals" / "photo - annotated.pdf"
    assert image_annotation.is_file()
    with fitz.open(image_annotation) as visual:
        assert visual.page_count == 1

    claim_payload = json.loads(
        (result_dir / "json_results" / "claim_result.json").read_text()
    )
    assert {result["document_type"] for result in claim_payload["results"]} == {
        "pdf",
        "webp",
    }


def test_font_agent_claim_set_folder(sample_pdf, tmp_path, monkeypatch, capsys):
    claim = tmp_path / "claim_fonts"
    claim.mkdir()
    shutil.copy(sample_pdf, claim / "page.pdf")
    out_dir = tmp_path / "font_out"

    monkeypatch.setattr(
        "sys.argv",
        ["font_agent.py", str(claim), "--out-dir", str(out_dir)],
    )
    font_agent.main()

    assert (out_dir / "report.md").is_file()
    assert (out_dir / "annotated_visuals" / "page - fonts annotated.pdf").is_file()
    assert (out_dir / "json_results" / "page - fonts result.json").is_file()
    assert (out_dir / "json_results" / "claim_result.json").is_file()
    assert {p.name for p in out_dir.iterdir() if p.is_dir()} == {
        "annotated_visuals",
        "json_results",
    }
    assert "Dominant font" in capsys.readouterr().out
