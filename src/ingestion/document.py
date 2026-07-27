"""Document and Chunk data models following DATA_CONTRACTS.md."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _make_chunk_id(document_id: str, page: int, seq: int) -> str:
    """Generate a stable chunk_id."""
    raw = f"{document_id}|p{page}|s{seq}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_document_id(source_file: str) -> str:
    """Generate a stable document_id from the source filename."""
    return hashlib.sha256(source_file.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A knowledge unit — matches the Data Contract."""

    document_id: str
    document_version: str
    page_number: int
    content: str
    content_type: str = "text"
    source_file: str = ""
    source_bbox: tuple[float, ...] | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Generated fields
    chunk_id: str = ""
    seq: int = 0

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = _make_chunk_id(self.document_id, self.page_number, self.seq)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "page_number": self.page_number,
            "content": self.content,
            "content_type": self.content_type,
            "source_file": self.source_file,
            "source_bbox": list(self.source_bbox) if self.source_bbox else None,
            "image_path": self.image_path,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """A parsed document ready for chunking."""

    document_id: str
    source_file: str
    version: str
    pages: list[str] = field(default_factory=list)  # one string per page
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pdf(cls, file_path: str) -> "Document":
        """Create a Document from a PDF file, computing version hash."""
        import fitz  # pymupdf

        path = file_path
        with open(path, "rb") as f:
            raw = f.read()
        version = hashlib.sha256(raw).hexdigest()[:16]
        doc_id = _make_document_id(path)

        pages: list[str] = []
        page_nums: list[int] = []

        pdf = fitz.open(path)
        total_pages = len(pdf)
        try:
            for i, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():
                    pages.append(text.strip())
                    page_nums.append(i)
        finally:
            pdf.close()

        return cls(
            document_id=doc_id,
            source_file=path,
            version=version,
            pages=pages,
            page_numbers=page_nums,
            metadata={
                "total_pages": total_pages,
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def chunk(self, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
        """Split document into fixed-size overlapping chunks."""
        chunks: list[Chunk] = []
        seq = 0

        for page_idx, (text, page_num) in enumerate(zip(self.pages, self.page_numbers)):
            # Simple fixed-size sliding window
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]

                if len(chunk_text.strip()) < 20:  # skip tiny fragments
                    start = end
                    continue

                chunk = Chunk(
                    document_id=self.document_id,
                    document_version=self.version,
                    page_number=page_num,
                    content=chunk_text,
                    content_type="text",
                    source_file=self.source_file,
                    seq=seq,
                )
                chunks.append(chunk)
                seq += 1

                if end >= len(text):
                    break
                start = end - overlap

        return chunks
