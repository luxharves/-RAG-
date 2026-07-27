#!/usr/bin/env python
"""Phase 3 — V0 vs V1 comparison on fixed 20 questions."""

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


def main() -> None:
    from src.retrieval.retriever import DenseRetriever
    from pymilvus import MilvusClient

    # ── Load 20 fixed questions ──
    with open(PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    # ── Find latest V1 collection ──
    milvus_path = str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(milvus_path)
    # Prefer keyword-enhanced V1, then timestamped, then legacy
    v1_kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    v1_ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    v1_all = v1_ts + v1_kw  # kw last, so [-1] picks kw
    v1_col = v1_all[-1] if v1_all else "v1_multimodal_kw"
    print(f"V1 collection: {v1_col}")
    client.close()

    # ── Run retrieval ──
    v0_retriever = DenseRetriever(collection_name="v0_naive_rag")
    v1_retriever = DenseRetriever(collection_name=v1_col)

    results = []
    print(f"V0 vs V1 comparison: {len(questions)} questions")
    print(f"V0: v0_naive_rag  V1: {v1_col}\n")

    for i, q in enumerate(questions, 1):
        q_text = q["question"]
        modality = q.get("modality_required", "text")
        gold_pages = q.get("gold_pages", [])

        v0_retrieved = v0_retriever.search(q_text, top_k=5)
        v1_retrieved = v1_retriever.search(q_text, top_k=5)

        v0_pages = [c["page_number"] for c in v0_retrieved]
        v1_pages = [c["page_number"] for c in v1_retrieved]
        v1_types = [c.get("content_type", "text") for c in v1_retrieved]

        v0_hit = bool(set(gold_pages) & set(v0_pages[:5]))
        v1_hit = bool(set(gold_pages) & set(v1_pages[:5]))

        results.append({
            "question_id": i,
            "question": q_text,
            "modality": modality,
            "gold_pages": gold_pages,
            "v0_pages": v0_pages,
            "v1_pages": v1_pages,
            "v1_content_types": v1_types,
            "v0_hit": v0_hit,
            "v1_hit": v1_hit,
        })

        status = "V0✓/V1✓" if (v0_hit and v1_hit) else ("V0✗/V1✓" if v1_hit else ("V0✓/V1✗" if v0_hit else "V0✗/V1✗"))
        print(f"  Q{i:2d} [{modality:5s}] {status}: v0={v0_pages} v1={v1_pages}")

    v0_retriever.close()
    v1_retriever.close()

    # ── Summary ──
    v0_hits = sum(1 for r in results if r["v0_hit"])
    v1_hits = sum(1 for r in results if r["v1_hit"])

    # Image-specific: Q18
    image_qs = [r for r in results if r["modality"] == "image"]
    text_qs = [r for r in results if r["modality"] == "text"]

    summary = {
        "experiment": "v0_v1_comparison",
        "v0_collection": "v0_naive_rag",
        "v1_collection": v1_col,
        "total_questions": len(questions),
        "v0_hit_rate": round(v0_hits / len(text_qs), 4) if text_qs else 0,
        "v1_hit_rate": round(v1_hits / len(questions), 4),
        "image_questions": {
            "q18_v0_hit": any(r["v0_hit"] for r in image_qs),
            "q18_v1_hit": any(r["v1_hit"] for r in image_qs),
            "q18_v1_content_types": [r["v1_content_types"] for r in image_qs],
        },
        "per_question": results,
    }

    out_path = PROJECT_ROOT / "storage" / "runs" / "v0_v1_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print(f"V0: {v0_hits}/{len(questions)} hit")
    print(f"V1: {v1_hits}/{len(questions)} hit")
    print(f"Q18 (image): V0={'HIT' if summary['image_questions']['q18_v0_hit'] else 'MISS'}, "
          f"V1={'HIT' if summary['image_questions']['q18_v1_hit'] else 'MISS'}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
