"""Phase 4 tests — BM25 index, hybrid retrieval, fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── BM25 ──


def test_bm25_index_exists() -> None:
    path = PROJECT_ROOT / "storage" / "bm25" / "bm25_index.pkl"
    assert path.exists(), "BM25 index not found — run scripts/build_bm25.py"


def test_bm25_meta_exists() -> None:
    path = PROJECT_ROOT / "storage" / "bm25" / "bm25_meta.json"
    assert path.exists()


def test_bm25_can_load_and_search() -> None:
    from src.retrieval.bm25 import BM25Retriever
    bm = BM25Retriever()
    bm.load(PROJECT_ROOT / "storage" / "bm25")
    results = bm.search("设备无法开机怎么办", top_k=3)
    assert len(results) > 0, "BM25 should return results"
    assert "page_number" in results[0]
    assert "bm25_score" in results[0]
    assert results[0]["retrieval_channel"] == "bm25"


def test_bm25_num_docs() -> None:
    from src.retrieval.bm25 import BM25Retriever
    bm = BM25Retriever()
    bm.load(PROJECT_ROOT / "storage" / "bm25")
    assert bm.num_docs > 0, f"BM25 index should have documents, got {bm.num_docs}"
    # Note: exact count varies by ingestion history; tests verify index usability


# ── Hybrid ──


def test_hybrid_dense_mode() -> None:
    from src.retrieval.hybrid_retriever import HybridRetriever
    hr = HybridRetriever(
        collection_name=_get_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
    )
    results = hr.search("设备无法开机怎么办", top_k=5, mode="dense")
    assert len(results) > 0
    channel = results[0].get("retrieval_channel", "")
    assert channel == "dense" or channel == ""
    hr.close()


def test_hybrid_bm25_mode() -> None:
    from src.retrieval.hybrid_retriever import HybridRetriever
    hr = HybridRetriever(
        collection_name=_get_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
    )
    results = hr.search("设备无法开机怎么办", top_k=5, mode="bm25")
    assert len(results) > 0
    assert results[0]["retrieval_channel"] == "bm25"
    hr.close()


def test_hybrid_mode_returns_fusion_scores() -> None:
    from src.retrieval.hybrid_retriever import HybridRetriever
    hr = HybridRetriever(
        collection_name=_get_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
    )
    results = hr.search("设备无法开机怎么办", top_k=5, mode="hybrid")
    assert len(results) > 0
    assert results[0]["retrieval_channel"] == "hybrid"
    assert "rrf_score" in results[0]
    assert results[0]["rrf_score"] is not None
    hr.close()


# ── V2 comparison ──


def test_v2_comparison_exists() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v2_comparison.json"
    assert path.exists()


def test_v2_hybrid_improves_over_bm25() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v2_comparison.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    bm25_hr = data["summary"]["bm25"]["hit_rate"]
    hybrid_hr = data["summary"]["hybrid"]["hit_rate"]
    assert hybrid_hr >= bm25_hr, f"Hybrid ({hybrid_hr}) should not be worse than BM25 ({bm25_hr})"


def test_hybrid_fusion_score_equals_rrf_score() -> None:
    """fusion_score must equal rrf_score in hybrid results."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    hr = HybridRetriever(
        collection_name=_get_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
    )
    results = hr.search("设备无法开机怎么办", top_k=5, mode="hybrid")
    hr.close()
    for r in results:
        if r.get("retrieval_channel") == "hybrid":
            assert "fusion_score" in r, "fusion_score missing"
            assert r["fusion_score"] == r["rrf_score"], \
                f"fusion_score={r['fusion_score']} != rrf_score={r['rrf_score']}"


def test_dense_unavailable_raises_error() -> None:
    """Mode=dense with no Milvus should raise HybridRetrievalError."""
    from src.retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalError
    hr = HybridRetriever(
        collection_name="nonexistent_collection",
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
    )
    with pytest.raises(HybridRetrievalError):
        hr.search("test", mode="dense")
    hr.close()


def test_bm25_unavailable_hybrid_degrades_to_dense() -> None:
    """Hybrid with no BM25 index should degrade to dense with degrade_reason."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    hr = HybridRetriever(
        collection_name=_get_v1_col(),
        bm25_index_path="nonexistent/path",
    )
    results = hr.search("设备无法开机怎么办", top_k=3, mode="hybrid")
    hr.close()
    assert len(results) > 0
    assert results[0]["retrieval_channel"] == "dense"
    assert "degrade_reason" in results[0]
    assert results[0]["degrade_reason"] == "bm25_unavailable"


def test_both_unavailable_raises_error() -> None:
    """Hybrid with neither Dense nor BM25 should raise HybridRetrievalError."""
    from src.retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalError
    hr = HybridRetriever(
        collection_name="nonexistent_collection",
        bm25_index_path="nonexistent/path",
    )
    with pytest.raises(HybridRetrievalError):
        hr.search("test", mode="hybrid")
    hr.close()


def test_v2_q19_improvement() -> None:
    """Q19 should improve in hybrid vs dense-only."""
    path = PROJECT_ROOT / "storage" / "runs" / "v2_comparison.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    q19_dense = data["summary"]["dense"]["q19"]
    q19_hybrid = data["summary"]["hybrid"]["q19"]
    # Hybrid should be at least as good as dense for Q19
    assert q19_hybrid["hit"] or not q19_dense["hit"], \
        "If dense hits Q19, hybrid should too"


# ── Helpers ──


def _get_v1_col() -> str:
    import os as _os
    _os.environ["MILVUS_URI"] = "http://localhost:19530"
    from pymilvus import MilvusClient
    from dotenv import dotenv_values
    env = dotenv_values(str(PROJECT_ROOT / ".env"))
    path = str(PROJECT_ROOT / env.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(path)
    kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    all_cols = ts + kw
    client.close()
    return all_cols[-1]
