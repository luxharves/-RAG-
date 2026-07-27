"""Phase 3 tests — V1 multimodal ingestion and V0/V1 comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _get_v1_collection() -> str:
    import os as _os
    _os.environ["MILVUS_URI"] = "http://localhost:19530"
    from pymilvus import MilvusClient
    from dotenv import dotenv_values
    env = dotenv_values(str(PROJECT_ROOT / ".env"))
    path = str(PROJECT_ROOT / env.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(path)
    cols = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    client.close()
    assert cols, "No V1 collection found"
    return cols[-1]


# ── V1 Ingestion metadata ──


def test_v1_ingest_metadata_exists() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v1_ingest" / "metadata.json"
    assert path.exists(), "v1_ingest/metadata.json not found"


def test_v1_has_image_chunks() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v1_ingest" / "metadata.json"
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["image_chunks"] >= 1, "V1 should have image chunks"


def test_v1_has_table_chunks() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v1_ingest" / "metadata.json"
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["table_chunks"] >= 1, "V1 should have table chunks"


def test_v1_total_chunks() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v1_ingest" / "metadata.json"
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    total = meta["total_chunks"]
    assert total > meta["text_chunks"], "V1 total should exceed text-only count"


def test_v1_collection_exists() -> None:
    col = _get_v1_collection()
    assert col, "V1 collection should exist"


# ── V0/V1 comparison ──


def test_v0_v1_comparison_exists() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v0_v1_comparison.json"
    assert path.exists(), "v0_v1_comparison.json not found"


def test_v0_collection_not_overwritten() -> None:
    """V0 collection must still exist and be independent."""
    import os as _os
    _os.environ["MILVUS_URI"] = "http://localhost:19530"
    from pymilvus import MilvusClient
    from dotenv import dotenv_values
    env = dotenv_values(str(PROJECT_ROOT / ".env"))
    path = str(PROJECT_ROOT / env.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(path)
    assert client.has_collection("v0_naive_rag"), "V0 collection must still exist"
    # V0 must only have text chunks
    client.load_collection("v0_naive_rag")
    # query all content types
    results = client.query(
        collection_name="v0_naive_rag",
        filter="content_type != \"text\"",
        limit=1,
        output_fields=["content_type"],
    )
    assert len(results) == 0, "V0 collection should not have non-text chunks"
    client.close()


def test_table_chunk_retrievable() -> None:
    """At least one table chunk should be searchable in V1."""
    import os as _os
    _os.environ["MILVUS_URI"] = "http://localhost:19530"
    from pymilvus import MilvusClient
    from dotenv import dotenv_values
    env = dotenv_values(str(PROJECT_ROOT / ".env"))
    path = str(PROJECT_ROOT / env.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(path)
    v1_kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    v1_ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    v1_all = v1_ts + v1_kw
    v1_col = v1_all[-1]

    from src.retrieval.retriever import DenseRetriever
    retriever = DenseRetriever(collection_name=v1_col)
    results = retriever.search("产品有害物质含量表", top_k=5)
    retriever.close()
    client.close()

    types = [r.get("content_type", "") for r in results]
    assert "table" in types, f"No table chunk in top-5: types={types}"


def test_q18_hits_image_chunk_in_v1() -> None:
    """Q18 must hit at least one image chunk in V1 top-5."""
    import os as _os
    _os.environ["MILVUS_URI"] = "http://localhost:19530"
    from pymilvus import MilvusClient
    from dotenv import dotenv_values
    env = dotenv_values(str(PROJECT_ROOT / ".env"))
    path = str(PROJECT_ROOT / env.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(path)
    v1_kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    v1_ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    v1_all = v1_ts + v1_kw
    v1_col = v1_all[-1]

    from src.retrieval.retriever import DenseRetriever
    retriever = DenseRetriever(collection_name=v1_col)
    results = retriever.search("机器人会不会从楼梯摔下去？", top_k=5)
    retriever.close()
    client.close()

    types = [r.get("content_type", "") for r in results]
    pages = [r.get("page_number", 0) for r in results]
    assert "image" in types, f"No image chunk in top-5: types={types}"
    assert 6 in pages, f"Page 6 not in top-5: pages={pages}"


def test_q18_remains_image_modality() -> None:
    """Q18 must be modality_required=image in the dataset."""
    with open(PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json", encoding="utf-8") as f:
        qs = json.load(f)
    q18 = qs[17]  # index 17 = Q18
    assert q18["modality_required"] == "image", f"Q18 must be image, got {q18['modality_required']}"
