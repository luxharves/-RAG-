#!/usr/bin/env python
"""Phase 5 — V2 Hybrid vs V3 Hybrid+Reranker comparison on 20 fixed questions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os as _os
from dotenv import dotenv_values as _dv
_os.environ["MILVUS_URI"] = "http://localhost:19530"
_ENV = _dv(str(PROJECT_ROOT / ".env"))


def _get_v1_col(client) -> str:
    kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    return (ts + kw)[-1] if (ts or kw) else "v1_multimodal_kw"


def _compute_mrr(gold, pages):
    for r, p in enumerate(pages, 1):
        if p in gold:
            return 1.0 / r
    return 0.0


def _compute_recall(gold, pages, k=5):
    return len(set(gold) & set(pages[:k])) / len(gold) if gold else 1.0


def main() -> None:
    from pymilvus import MilvusClient
    from src.retrieval.reranked_retriever import RerankedRetriever

    client = MilvusClient(str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db")))
    v1_col = _get_v1_col(client)
    client.close()

    bm25_path = str(PROJECT_ROOT / "storage" / "bm25")
    retriever = RerankedRetriever(
        collection_name=v1_col,
        bm25_index_path=bm25_path,
        candidate_top_k=20,
        final_top_k=5,
    )

    with open(PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"V3 Comparison: {len(questions)} questions, candidate=20, final=5")

    v2_results, v3_results = [], []
    ranking_changes = []
    total_start = time.perf_counter()
    v2_times, v3_times = [], []

    for q in questions:
        q_text = q["question"]
        gold = q.get("gold_pages", [])
        modality = q.get("modality_required", "text")

        # V2
        t0 = time.perf_counter()
        v2_ret = retriever.search(q_text, top_k=5, mode="v2_hybrid")
        v2_times.append(time.perf_counter() - t0)
        v2_pages = [r["page_number"] for r in v2_ret]
        v2_types = [r.get("content_type", "") for r in v2_ret]
        v2_results.append({
            "question": q_text, "modality": modality, "gold_pages": gold,
            "pages": v2_pages, "types": v2_types,
            "recall_at_5": round(_compute_recall(gold, v2_pages), 4),
            "mrr": round(_compute_mrr(gold, v2_pages), 4),
            "hit": bool(set(gold) & set(v2_pages[:5])),
            "top1_hit": v2_pages[0] in gold if v2_pages else False,
        })

        # V3
        t0 = time.perf_counter()
        v3_ret = retriever.search(q_text, top_k=5, mode="reranked")
        v3_times.append(time.perf_counter() - t0)
        v3_pages = [r["page_number"] for r in v3_ret]
        v3_types = [r.get("content_type", "") for r in v3_ret]
        changed = any(r.get("ranking_changed", False) for r in v3_ret)

        # Full per-chunk details
        final_chunks = []
        for r in v3_ret:
            final_chunks.append({
                "chunk_id": r.get("chunk_id", ""),
                "page_number": r.get("page_number", 0),
                "content_type": r.get("content_type", ""),
                "fusion_score": r.get("fusion_score"),
                "rrf_score": r.get("rrf_score"),
                "rerank_score": r.get("rerank_score"),
                "rerank_rank": r.get("rerank_rank"),
                "original_hybrid_rank": r.get("original_hybrid_rank"),
                "dense_rank": r.get("dense_rank"),
                "bm25_rank": r.get("bm25_rank"),
                "content_preview": r.get("content", "")[:80],
            })

        v3_results.append({
            "question": q_text, "modality": modality, "gold_pages": gold,
            "candidate_count": 20, "final_count": 5,
            "pages": v3_pages, "types": v3_types,
            "rerank_scores": [r.get("rerank_score") for r in v3_ret],
            "rerank_ranks": [r.get("rerank_rank") for r in v3_ret],
            "original_hybrid_ranks": [r.get("original_hybrid_rank") for r in v3_ret],
            "ranking_changed": changed,
            "recall_at_5": round(_compute_recall(gold, v3_pages), 4),
            "mrr": round(_compute_mrr(gold, v3_pages), 4),
            "hit": bool(set(gold) & set(v3_pages[:5])),
            "top1_hit": v3_pages[0] in gold if v3_pages else False,
            "final_results": final_chunks,
        })
        if changed:
            rc_entry = {
                "question": q_text,
                "v2_top3_pages": v2_pages[:3],
                "v3_top3_pages": v3_pages[:3],
                "changes": []
            }
            for r in v3_ret:
                old = r.get("original_hybrid_rank", 0)
                new = r.get("rerank_rank", 0)
                if old != new:
                    rc_entry["changes"].append({
                        "chunk_id": r.get("chunk_id", ""),
                        "page_number": r.get("page_number", 0),
                        "content_type": r.get("content_type", ""),
                        "original_hybrid_rank": old,
                        "rerank_rank": new,
                        "fusion_score": r.get("fusion_score"),
                        "rerank_score": r.get("rerank_score"),
                    })
            ranking_changes.append(rc_entry)

    retriever.close()
    total_time = time.perf_counter() - total_start

    # ── Summaries ──
    def summarize(results, times):
        return {
            "hit_rate": round(sum(r["hit"] for r in results) / len(results), 4),
            "top1_hit_rate": round(sum(r["top1_hit"] for r in results) / len(results), 4),
            "recall_at_5": round(sum(r["recall_at_5"] for r in results) / len(results), 4),
            "mrr": round(sum(r["mrr"] for r in results) / len(results), 4),
            "avg_time_s": round(sum(times) / len(times), 4),
        }

    v2_sum = summarize(v2_results, v2_times)
    v3_sum = summarize(v3_results, v3_times)

    # Q18/Q19
    def q_report(results, keyword):
        r = next((x for x in results if keyword in x["question"]), None)
        return {"hit": r["hit"], "pages": r["pages"], "types": r["types"]} if r else None

    # Print
    print(f"\n{'='*70}")
    print(f"{'Metric':<20} {'V2 Hybrid':<15} {'V3 Reranked':<15} {'Delta':<10}")
    print(f"{'='*70}")
    for metric in ["hit_rate", "top1_hit_rate", "recall_at_5", "mrr", "avg_time_s"]:
        v2v, v3v = v2_sum[metric], v3_sum[metric]
        delta = f"{v3v - v2v:+.4f}"
        print(f"{metric:<20} {v2v:<15.4f} {v3v:<15.4f} {delta:<10}")

    print(f"\nRanking changes: {len(ranking_changes)}/{len(questions)} questions affected")
    for rc in ranking_changes:
        print(f"  '{rc['question'][:40]}': v2={rc['v2_top3_pages']} → v3={rc['v3_top3_pages']}")

    print(f"\nQ18 (image):")
    for label, res in [("V2", v2_results), ("V3", v3_results)]:
        r = q_report(res, "楼梯")
        print(f"  {label}: {'HIT' if r['hit'] else 'MISS'} pages={r['pages']} types={r['types']}")

    print(f"\nQ19 (text):")
    for label, res in [("V2", v2_results), ("V3", v3_results)]:
        r = q_report(res, "不动")
        print(f"  {label}: {'HIT' if r['hit'] else 'MISS'} pages={r['pages']} types={r['types']}")

    # ── Save ──
    out_dir = PROJECT_ROOT / "storage" / "runs" / "v3_rerank"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": "v3_rerank", "version": "V3",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "candidate_top_k": 20, "final_top_k": 5,
        "v1_collection": v1_col, "bm25_index": bm25_path,
        "ranked_questions": len(ranking_changes),
        "total_time_s": round(total_time, 1),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    comparison = {"v2_summary": v2_sum, "v3_summary": v3_sum,
                  "v2_results": v2_results, "v3_results": v3_results}
    with open(out_dir / "v2_v3_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    with open(out_dir / "v3_results.json", "w", encoding="utf-8") as f:
        json.dump(v3_results, f, ensure_ascii=False, indent=2)

    with open(out_dir / "ranking_changes.json", "w", encoding="utf-8") as f:
        json.dump(ranking_changes, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
