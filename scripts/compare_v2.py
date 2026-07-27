#!/usr/bin/env python
"""Phase 4 — V1 Dense vs V2 BM25 vs V2 Hybrid comparison on 20 fixed questions."""

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


def _get_latest_v1_col(client) -> str:
    kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    all_cols = ts + kw
    return all_cols[-1] if all_cols else "v1_multimodal_kw"


def _compute_mrr(gold_pages, retrieved_pages) -> float:
    for rank, p in enumerate(retrieved_pages, 1):
        if p in gold_pages:
            return 1.0 / rank
    return 0.0


def _compute_recall_at_k(gold_pages, retrieved_pages, k=5) -> float:
    found = len(set(gold_pages) & set(retrieved_pages[:k]))
    return found / len(gold_pages) if gold_pages else 1.0


def main() -> None:
    from pymilvus import MilvusClient
    from src.retrieval.hybrid_retriever import HybridRetriever

    milvus_path = str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(milvus_path)
    v1_col = _get_latest_v1_col(client)
    client.close()

    bm25_path = str(PROJECT_ROOT / "storage" / "bm25")
    retriever = HybridRetriever(collection_name=v1_col, bm25_index_path=bm25_path)

    with open(PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"V2 Comparison: {len(questions)} questions")
    print(f"Collection: {v1_col}")

    modes = ["dense", "bm25", "hybrid"]
    all_results: dict[str, list] = {m: [] for m in modes}

    for q in questions:
        q_text = q["question"]
        gold = q.get("gold_pages", [])
        modality = q.get("modality_required", "text")

        for mode in modes:
            retrieved = retriever.search(q_text, top_k=5, mode=mode)
            pages = [r["page_number"] for r in retrieved]
            types = [r.get("content_type", "text") for r in retrieved]
            channel = retrieved[0].get("retrieval_channel", mode) if retrieved else mode
            all_results[mode].append({
                "question": q_text,
                "modality": modality,
                "gold_pages": gold,
                "mode": mode,
                "retrieved_pages": pages,
                "content_types": types,
                "channel": channel,
                "recall_at_5": round(_compute_recall_at_k(gold, pages), 4),
                "mrr": round(_compute_mrr(gold, pages), 4),
                "hit": bool(set(gold) & set(pages[:5])),
                "top_result": {
                    "page": retrieved[0]["page_number"] if retrieved else 0,
                    "type": retrieved[0].get("content_type", "") if retrieved else "",
                    "dense_rank": retrieved[0].get("dense_rank") if retrieved else None,
                    "bm25_rank": retrieved[0].get("bm25_rank") if retrieved else None,
                    "rrf_score": retrieved[0].get("rrf_score") if retrieved else None,
                } if retrieved else None,
            })

    retriever.close()

    # ── Summary ──
    summary: dict[str, dict] = {}
    for mode in modes:
        entries = all_results[mode]
        hits = sum(1 for e in entries if e["hit"])
        recall5 = sum(e["recall_at_5"] for e in entries) / len(entries) if entries else 0
        mrr = sum(e["mrr"] for e in entries) / len(entries) if entries else 0
        summary[mode] = {
            "hit_rate": round(hits / len(entries), 4) if entries else 0,
            "hits": hits,
            "total": len(entries),
            "recall_at_5": round(recall5, 4),
            "mrr": round(mrr, 4),
        }
        # Q18/Q19 special
        q18 = next((e for e in entries if "楼梯" in e["question"]), None)
        q19 = next((e for e in entries if "不动" in e["question"]), None)
        summary[mode]["q18"] = {"hit": q18["hit"], "pages": q18["retrieved_pages"], "types": q18["content_types"]} if q18 else None
        summary[mode]["q19"] = {"hit": q19["hit"], "pages": q19["retrieved_pages"], "types": q19["content_types"]} if q19 else None

    # ── Print ──
    print(f"\n{'='*60}")
    print(f"{'Mode':<10} {'Hits':<8} {'Hit Rate':<10} {'Recall@5':<10} {'MRR':<10}")
    print(f"{'='*60}")
    for mode in modes:
        s = summary[mode]
        print(f"{mode:<10} {s['hits']}/{s['total']:<5} {s['hit_rate']:<10.4f} {s['recall_at_5']:<10.4f} {s['mrr']:<10.4f}")
    print(f"\nQ18 (image):")
    for mode in modes:
        s = summary[mode]["q18"]
        print(f"  {mode}: {'HIT' if s['hit'] else 'MISS'} pages={s['pages']} types={s['types']}")
    print(f"\nQ19 (text):")
    for mode in modes:
        s = summary[mode]["q19"]
        print(f"  {mode}: {'HIT' if s['hit'] else 'MISS'} pages={s['pages']} types={s['types']}")

    # ── Save ──
    output = {
        "experiment": "v2_hybrid",
        "version": "V2",
        "v1_collection": v1_col,
        "bm25_index": bm25_path,
        "summary": summary,
        "per_question": {m: all_results[m] for m in modes},
    }
    out_path = PROJECT_ROOT / "storage" / "runs" / "v2_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
