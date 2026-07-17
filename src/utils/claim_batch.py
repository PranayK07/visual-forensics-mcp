"""Claim-set folder discovery and output-path helpers for CLI batch runs.

A *claim set* is a folder of related documents (e.g. one insurance claim).
Callers may pass:

* a single file,
* one claim-set folder (documents directly inside), or
* a claims root whose immediate subfolders are claim sets.

Outputs mirror the claim-set name under ``--out-dir`` (or a sibling
``<name>_result`` folder when ``--out-dir`` is omitted).
"""

from __future__ import annotations

import os
import ntpath
from dataclasses import dataclass

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".gif",
}


def supported_extensions_label() -> str:
    """Human-readable extension list for discovery errors and CLI help."""
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


@dataclass(frozen=True)
class ClaimSet:
    """One group of related documents to analyze together."""

    name: str
    documents: tuple[str, ...]
    source_dir: str | None  # None when the input was a lone file


def is_supported_document(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def list_documents_in_dir(directory: str) -> list[str]:
    """Return sorted supported document paths directly inside ``directory``."""
    names = []
    try:
        entries = os.listdir(directory)
    except OSError as exc:
        raise SystemExit(f"Cannot read directory: {directory} ({exc})") from exc
    for name in sorted(entries):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and is_supported_document(path):
            names.append(path)
    return names


def discover_claim_sets(path: str) -> list[ClaimSet]:
    """Resolve ``path`` into one or more claim sets.

    * File → one claim set containing that file.
    * Directory with supported docs and no claim-set subfolders → one claim set.
    * Directory whose subfolders contain docs → one claim set per such subfolder.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise SystemExit(f"Path not found: {path}")

    if os.path.isfile(path):
        if not is_supported_document(path):
            raise SystemExit(
                f"Unsupported file type: {path} "
                f"(expected {supported_extensions_label()})"
            )
        stem = os.path.splitext(os.path.basename(path))[0]
        return [ClaimSet(name=stem, documents=(path,), source_dir=None)]

    if not os.path.isdir(path):
        raise SystemExit(f"Not a file or directory: {path}")

    subdirs = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
        and not name.startswith(".")
        and not name.endswith("_annotated")
        and not name.endswith("_result")
    )
    claim_subdirs = [
        (d, list_documents_in_dir(d)) for d in subdirs if list_documents_in_dir(d)
    ]
    top_docs = list_documents_in_dir(path)

    if claim_subdirs:
        # Claims root: each populated subfolder is a claim set.
        # Loose top-level docs (if any) form an extra set named after the root.
        sets = [
            ClaimSet(
                name=os.path.basename(d),
                documents=tuple(docs),
                source_dir=d,
            )
            for d, docs in claim_subdirs
        ]
        if top_docs:
            sets.insert(
                0,
                ClaimSet(
                    name=os.path.basename(path),
                    documents=tuple(top_docs),
                    source_dir=path,
                ),
            )
        return sets

    if not top_docs:
        raise SystemExit(
            f"No supported evidence files found under: {path}\n"
            "Pass a file, a claim-set folder of documents, or a parent folder "
            "whose subfolders are claim sets."
        )

    return [
        ClaimSet(
            name=os.path.basename(path),
            documents=tuple(top_docs),
            source_dir=path,
        )
    ]


def default_out_dir(input_path: str) -> str:
    """Sibling ``<name>_result`` directory for file or folder inputs."""
    path = os.path.abspath(input_path)
    if os.path.isfile(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(os.path.dirname(path), f"{stem}_result")
    return path.rstrip("/\\") + "_result"


def claim_output_dir(claim: ClaimSet, out_dir: str, input_path: str) -> str:
    """Directory where one claim set's annotated files should be written.

    * Lone file → ``out_dir`` directly.
    * Single claim-set folder as input → ``out_dir`` directly (no extra nest).
    * Claims root → ``out_dir/<claim_name>/`` so each claim set stays separate.
    """
    out_dir = os.path.abspath(out_dir)
    input_path = os.path.abspath(input_path)

    if claim.source_dir is None and os.path.isfile(input_path):
        return out_dir

    # Input was exactly this claim-set folder (not a parent claims root).
    if claim.source_dir is not None and os.path.abspath(claim.source_dir) == input_path:
        return out_dir

    return os.path.join(out_dir, claim.name)


def annotated_pdf_path(document_path: str, output_dir: str, suffix: str = "annotated") -> str:
    # ntpath.basename also handles POSIX separators and prevents a Windows
    # source directory from leaking into the output name on Linux/macOS.
    stem = os.path.splitext(ntpath.basename(document_path))[0]
    return os.path.join(output_dir, f"{stem} - {suffix}.pdf")


def result_json_path(document_path: str, output_dir: str, suffix: str = "result") -> str:
    stem = os.path.splitext(ntpath.basename(document_path))[0]
    return os.path.join(output_dir, f"{stem} - {suffix}.json")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
