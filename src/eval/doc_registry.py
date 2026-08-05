"""Mapping of friendly source_document names to actual source_file paths.

Centralizes the name → path mapping used by the eval scripts and the dataset
builder, so a per-question ``doc_filter`` can restrict retrieval to one manual.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# friendly name -> PDF filename in data/raw_docs
_DOC_FILENAMES: dict[str, str] = {
    "Roborock G10S": "Roborock G10S Auto-Empty Wet and Dry Robot Vacuum User Manual_v1.0.pdf",
    "Ecovacs DEEBOT T30C": "Ecovacs DEEBOT T30C Manual.pdf",
}


def source_document_map() -> dict[str, str]:
    """Map friendly source_document name -> absolute source_file path.

    Only includes documents whose PDF is actually present.
    """
    out: dict[str, str] = {}
    for name, filename in _DOC_FILENAMES.items():
        p = PROJECT_ROOT / "data" / "raw_docs" / filename
        if p.exists():
            out[name] = str(p)
    return out


def resolve_doc_filter(source_document: str | None) -> str | None:
    """Resolve a question's source_document to a source_file filter path."""
    if not source_document:
        return None
    return source_document_map().get(source_document)
