"""Fixed-rule text chunking."""

from src.ingestion.document import Document, Chunk


def chunk_document(
    doc: Document,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a Document into fixed-size overlapping Chunks."""
    return doc.chunk(chunk_size=chunk_size, overlap=overlap)
