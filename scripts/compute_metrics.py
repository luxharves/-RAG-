#!/usr/bin/env python
"""Phase 1 — Official retrieval metrics calculator.

Reads questions JSON and V0 results, auto-excludes non-text questions
based on modality_required, outputs retrieval_metrics.json.

This is the SINGLE authoritative metrics script for Phase 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    # ── 1. Load questions (authoritative gold data) ──
    questions_path = PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json"
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    # Build lookup: {question_id: {...}}
    q_lookup: dict[int, dict] = {}
    for i, q in enumerate(questions, 1):
        q_lookup[i] = q

    # ── 2. Load V0 retrieval results ──
    baseline_dir = PROJECT_ROOT / "storage" / "runs" / "v0_baseline"
    result_files = sorted(baseline_dir.glob("v0_results_*.json"))
    if not result_files:
        print("ERROR: No V0 results found")
        sys.exit(1)

    with open(result_files[-1], encoding="utf-8") as f:
        v0_results = json.load(f)
    print(f"Loaded {len(v0_results)} retrieval results")

    # ── 3. Filter: text-only questions ──
    TEXT_ONLY_MODALITIES = {"text"}
    excluded_ids: list[int] = []
    text_results = []

    for result in v0_results:
        qid = result["question_id"]
        q_info = q_lookup.get(qid, {})
        modality = q_info.get("modality_required", "text")
        if modality in TEXT_ONLY_MODALITIES:
            text_results.append(result)
        else:
            excluded_ids.append(qid)

    # ── 4. Compute metrics ──
    per_question = []
    hits = 0
    total_gold = 0
    recalled_gold = 0

    for result in text_results:
        qid = result["question_id"]
        q_info = q_lookup.get(qid, {})
        gold_pages = q_info.get("gold_pages", [])

        retrieved_pages = set(
            c["page_number"] for c in result["retrieved_chunks"]
        )
        found = [p for p in gold_pages if p in retrieved_pages]
        is_hit = len(found) > 0
        recall = len(found) / len(gold_pages) if gold_pages else 1.0

        if is_hit:
            hits += 1
        total_gold += len(gold_pages)
        recalled_gold += len(found)

        per_question.append({
            "question_id": qid,
            "question": result["question"][:80],
            "modality": q_info.get("modality_required", "?"),
            "gold_pages": gold_pages,
            "retrieved_pages": sorted(retrieved_pages),
            "found_pages": found,
            "hit": is_hit,
            "recall_at_5": round(recall, 4),
        })

    avg_recall = recalled_gold / total_gold if total_gold else 0.0
    hit_rate = hits / len(text_results) if text_results else 0.0

    # ── 5. Generate retrieval_metrics.json ──
    metrics = {
        "experiment": "v0_naive",
        "version": "V0",
        "retrieval_channel": "dense",
        "embedding_model": "BAAI/bge-m3",
        "num_questions_total": len(v0_results),
        "num_questions_text_only": len(text_results),
        "excluded_questions": sorted(excluded_ids),
        "exclusion_rule": "modality_required != 'text'",
        "top_k": 5,
        "metrics": {
            "recall_at_5": round(avg_recall, 4),
            "top5_hit_count": hits,
            "top5_hit_rate": round(hit_rate, 4),
            "total_gold_pages": total_gold,
            "total_recalled_pages": recalled_gold,
        },
        "per_question": per_question,
    }

    metrics_path = baseline_dir / "retrieval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── 6. Print summary ──
    print(f"\n{'=' * 60}")
    print(f"V0 Text-Only Retrieval Metrics")
    print(f"{'=' * 60}")
    print(f"  Total questions:  {len(v0_results)}")
    print(f"  Text-only:        {len(text_results)}")
    print(f"  Excluded:         {sorted(excluded_ids)} "
          f"(modality_required != 'text')")
    print(f"  Recall@5:         {avg_recall:.4f} "
          f"({recalled_gold}/{total_gold})")
    print(f"  Top-5 Hits:       {hits}/{len(text_results)}")
    print(f"  Hit Rate:         {hit_rate:.4f}")
    print(f"\nPer-question:")
    for pq in per_question:
        status = "HIT " if pq["hit"] else "MISS"
        print(f"  Q{pq['question_id']:2d}: {status}  "
              f"modality={pq['modality']:5s}  "
              f"gold={str(pq['gold_pages']):10s}  "
              f"found={str(pq['found_pages']):10s}  "
              f"recall={pq['recall_at_5']:.2f}")
    print(f"\n  Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
