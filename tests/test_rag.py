"""Phase 1 integration tests for the Naive RAG pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ──


@pytest.fixture(scope="module")
def eval_questions() -> list[dict]:
    path = PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Tests ──


def test_eval_dataset_exists() -> None:
    path = PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json"
    assert path.exists(), f"Eval dataset not found: {path}"


def test_eval_dataset_has_20_questions(eval_questions: list[dict]) -> None:
    assert len(eval_questions) >= 20, f"Expected >=20 questions, got {len(eval_questions)}"


def test_eval_dataset_schema(eval_questions: list[dict]) -> None:
    for i, q in enumerate(eval_questions):
        assert "question" in q, f"Q{i}: missing 'question'"
        assert "question_type" in q, f"Q{i}: missing 'question_type'"
        assert "difficulty" in q, f"Q{i}: missing 'difficulty'"


def test_pdf_exists() -> None:
    raw_docs = PROJECT_ROOT / "data" / "raw_docs"
    pdfs = list(raw_docs.glob("*.pdf"))
    assert len(pdfs) > 0, f"No PDF found in {raw_docs}"


def test_parse_pdf() -> None:
    """PDF should parse into a Document with pages."""
    from src.ingestion.pdf_parser import parse_pdf

    raw_docs = PROJECT_ROOT / "data" / "raw_docs"
    pdfs = list(raw_docs.glob("*.pdf"))
    doc = parse_pdf(str(pdfs[0]))

    assert doc.document_id, "Document ID should be set"
    assert doc.version, "Document version should be set"
    assert len(doc.pages) > 0, "Document should have pages with text"
    assert doc.metadata["total_pages"] > 0


def test_chunk_document() -> None:
    """Document should produce chunks with correct metadata."""
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.chunker import chunk_document

    raw_docs = PROJECT_ROOT / "data" / "raw_docs"
    pdfs = list(raw_docs.glob("*.pdf"))
    doc = parse_pdf(str(pdfs[0]))
    chunks = chunk_document(doc, chunk_size=500, overlap=50)

    assert len(chunks) > 0, "Should produce chunks"
    for c in chunks:
        assert c.chunk_id, "Chunk should have chunk_id"
        assert c.document_id, "Chunk should have document_id"
        assert c.document_version, "Chunk should have document_version"
        assert c.page_number > 0, "Chunk should have page_number"
        assert c.content, "Chunk should have content"
        assert c.content_type == "text", "Chunk content_type should be 'text'"
        assert c.source_file, "Chunk should have source_file"


def test_chunk_content_not_empty() -> None:
    """All chunks should have non-trivial content."""
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.chunker import chunk_document

    raw_docs = PROJECT_ROOT / "data" / "raw_docs"
    pdfs = list(raw_docs.glob("*.pdf"))
    doc = parse_pdf(str(pdfs[0]))
    chunks = chunk_document(doc, chunk_size=500, overlap=50)

    empty = [c for c in chunks if len(c.content.strip()) < 10]
    assert len(empty) == 0, f"Found {len(empty)} empty/near-empty chunks"


def test_generator_refuses_without_context() -> None:
    """LLM should refuse to answer when given no retrieved chunks."""
    from src.generation.generator import generate_answer

    result = generate_answer("这个问题在说明书中肯定找不到答案。", [])

    assert result["chunks_used"] == 0
    assert len(result["citations"]) == 0
    # Should contain refusal language
    refusal_keywords = ["无法回答", "说明书", "没有"]
    assert any(kw in result["answer"] for kw in refusal_keywords), (
        f"Expected refusal language, got: {result['answer'][:100]}"
    )


def test_milvus_collection_available() -> None:
    """After ingestion, Milvus collection should exist and be queryable."""
    from pymilvus import MilvusClient
    from dotenv import dotenv_values

    # Read real Milvus URI from .env (not the env-var override)
    env_vals = dotenv_values(str(PROJECT_ROOT / ".env"))
    milvus_uri = env_vals.get("MILVUS_URI", "milvus.db")
    if not milvus_uri.startswith("http"):
        milvus_uri = str(PROJECT_ROOT / milvus_uri)

    client = MilvusClient(milvus_uri)
    exists = client.has_collection("v0_naive_rag")
    assert exists, "Collection v0_naive_rag not found — run scripts/ingest.py first"
    client.close()
