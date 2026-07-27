"""PDF text parser — extracts page text via pymupdf."""

from pathlib import Path
from src.ingestion.document import Document


def parse_pdf(file_path: str | Path) -> Document:
    """Parse a PDF file into a Document with page-level text."""
    return Document.from_pdf(str(file_path))
