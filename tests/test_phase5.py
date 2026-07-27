"""Phase 5 tests — Reranker, V2/V3 comparison, fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _v1_col() -> str:
    import os as _os
    _os.environ["MILVUS_URI"] = "http://localhost:19530"
    from pymilvus import MilvusClient
    from dotenv import dotenv_values
    env = dotenv_values(str(PROJECT_ROOT / ".env"))
    c = MilvusClient(str(PROJECT_ROOT / env.get("MILVUS_URI", "milvus.db")))
    kw = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_kw_")])
    ts = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_2")])
    c.close()
    return (ts + kw)[-1]


# ── Reranker load + score ──


def test_reranker_loads_and_scores() -> None:
    from src.infra.reranker import Reranker
    r = Reranker()
    r.load()
    scores = r.score("设备无法开机", [
        "电池电量不足，请先靠上基座充电后再使用",
        "设备屏幕亮度可以在设置中调节",
    ])
    assert len(scores) == 2
    assert scores[0] > scores[1], "Relevant doc should score higher"


# ── V3 retrieval ──


def test_v3_candidates_more_than_final() -> None:
    from src.retrieval.reranked_retriever import RerankedRetriever
    rr = RerankedRetriever(
        collection_name=_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
        candidate_top_k=20, final_top_k=5,
    )
    results = rr.search("设备无法开机怎么办", mode="reranked")
    rr.close()
    assert len(results) <= 5


def test_v3_has_rerank_score_and_rank() -> None:
    from src.retrieval.reranked_retriever import RerankedRetriever
    rr = RerankedRetriever(
        collection_name=_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
        candidate_top_k=20, final_top_k=5,
    )
    results = rr.search("设备无法开机怎么办", mode="reranked")
    rr.close()
    for r in results:
        assert "rerank_score" in r
        assert "rerank_rank" in r
        assert "ranking_changed" in r


def test_v3_rankings_differ_from_v2() -> None:
    """At least some questions should have ranking changes."""
    path = PROJECT_ROOT / "storage" / "runs" / "v3_rerank" / "v2_v3_comparison.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changes = sum(
        1 for v2, v3 in zip(data["v2_results"], data["v3_results"])
        if v2["pages"][:3] != v3["pages"][:3]
    )
    assert changes > 0, "Reranker should change rankings for at least some questions"


# ── V2 not overwritten ──


def test_v2_results_not_overwritten() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v2_comparison.json"
    assert path.exists(), "V2 results should still exist"


# ── Fallback: reranker unavailable ──


def test_reranker_unavailable_fallback() -> None:
    """If reranker model path is invalid, should return results with degrade_reason."""
    from src.retrieval.reranked_retriever import RerankedRetriever
    import src.infra.reranker as rk_mod

    # Monkey-patch Reranker.load to simulate failure
    orig_load = rk_mod.Reranker.load
    def _fail_load(self):
        raise RuntimeError("simulated load failure")
    rk_mod.Reranker.load = _fail_load

    try:
        rr = RerankedRetriever(
            collection_name=_v1_col(),
            bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
            candidate_top_k=20, final_top_k=5,
        )
        results = rr.search("设备无法开机怎么办", mode="reranked")
        rr.close()
        assert len(results) >= 1
        # All should have degrade_reason set
        for r in results:
            assert r.get("degrade_reason") == "reranker_unavailable"
    finally:
        rk_mod.Reranker.load = orig_load


# ── V3 output files ──


def test_v3_output_files_exist() -> None:
    out = PROJECT_ROOT / "storage" / "runs" / "v3_rerank"
    for name in ["v3_results.json", "v2_v3_comparison.json", "metadata.json",
                 "ranking_changes.json"]:
        assert (out / name).exists(), f"Missing: {name}"


def test_v3_results_have_full_chunk_details() -> None:
    """v3_results.json must have candidate_count, final_results with chunk details."""
    path = PROJECT_ROOT / "storage" / "runs" / "v3_rerank" / "v3_results.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        assert r.get("candidate_count") == 20
        assert r.get("final_count") == 5
        assert "rerank_scores" in r
        assert "rerank_ranks" in r
        assert "original_hybrid_ranks" in r
        assert "final_results" in r
        for chunk in r["final_results"]:
            for field in ["chunk_id", "page_number", "content_type",
                          "rerank_score", "rerank_rank"]:
                assert field in chunk, f"final_results missing {field}"


def test_ranking_changes_have_details() -> None:
    """ranking_changes.json must have chunk-level before/after."""
    path = PROJECT_ROOT / "storage" / "runs" / "v3_rerank" / "ranking_changes.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) > 0, "Should have ranking changes"
    for entry in data:
        assert "question" in entry
        assert "changes" in entry
        if entry["changes"]:
            c = entry["changes"][0]
            for field in ["chunk_id", "page_number", "original_hybrid_rank",
                          "rerank_rank", "fusion_score", "rerank_score"]:
                assert field in c, f"change missing {field}"


def test_q18_q19_in_comparison() -> None:
    path = PROJECT_ROOT / "storage" / "runs" / "v3_rerank" / "v3_results.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    q18 = any("楼梯" in r["question"] for r in data)
    q19 = any("不动" in r["question"] for r in data)
    assert q18 and q19, "Q18/Q19 must be in V3 results"


def test_v3_compare_script_no_llm_vlm_imports() -> None:
    """compare_v3.py must not import LLM/VLM/RAGAS modules."""
    script = PROJECT_ROOT / "scripts" / "compare_v3.py"
    text = script.read_text(encoding="utf-8")
    forbidden = ["ragas", "LLMClient", "VLMClient", "generate_answer"]
    for kw in forbidden:
        assert kw not in text, f"compare_v3.py should not import/use {kw}"
