#!/usr/bin/env python
"""Phase 2 — RAGAS evaluation on Golden Dataset (100 questions).

Uses ragas 0.4.3 for: faithfulness, answer_relevancy, context_precision, context_recall.
Also computes: Recall@5, MRR.

Saves V0 Baseline to storage/runs/v0_baseline/phase2_metrics.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ⚠️ pymilvus + ragas workarounds
import os as _os
_os.environ["MILVUS_URI"] = "http://localhost:19530"
import src.eval.ragas_patch  # noqa: E402 — must be before ragas imports


def _compute_recall_at_k(gold_pages, retrieved_pages, k=5) -> float:
    found = len(set(gold_pages) & set(retrieved_pages[:k]))
    return found / len(gold_pages) if gold_pages else 1.0


def _compute_mrr(gold_pages, retrieved_pages) -> float:
    for rank, p in enumerate(retrieved_pages, 1):
        if p in gold_pages:
            return 1.0 / rank
    return 0.0


def main() -> None:
    # ── 1. Load Golden Dataset ──
    dataset_path = PROJECT_ROOT / "data" / "eval_dataset" / "golden_100.json"
    with open(dataset_path, encoding="utf-8") as f:
        questions = json.load(f)
    text_qs = [q for q in questions if q.get("modality_required", "text") == "text"]
    print(f"Golden Dataset: {len(questions)} total, {len(text_qs)} text-only")

    # ── 2. Run retrieval + generation ──
    from src.retrieval.retriever import DenseRetriever
    from src.generation.generator import generate_answer

    retriever = DenseRetriever(collection_name="v0_naive_rag")
    results = []

    # Load caches: Phase 1 results + Phase 2 generation cache
    baseline_dir = PROJECT_ROOT / "storage" / "runs" / "v0_baseline"
    phase1_cache: dict[str, dict] = {}
    phase1_files = sorted(baseline_dir.glob("v0_results_*.json"))
    if phase1_files:
        with open(phase1_files[-1], encoding="utf-8") as f:
            for r in json.load(f):
                phase1_cache[r["question"]] = r

    gen_cache_path = baseline_dir / "phase2_gen_cache.json"
    if gen_cache_path.exists():
        with open(gen_cache_path, encoding="utf-8") as f:
            p2_gen_cache = json.load(f)
    else:
        p2_gen_cache = {}
    gen_cache_dirty = False

    n_phase1 = n_p2cached = n_new = 0
    total_start = time.perf_counter()

    for i, q in enumerate(text_qs, 1):
        q_text = q["question"]
        gold_pages = q.get("gold_pages", [])

        if q_text in phase1_cache:
            entry = phase1_cache[q_text]
            retrieved = entry["retrieved_chunks"]
            answer = entry["answer"]
            source = "phase1_cached"
            n_phase1 += 1
        elif q_text in p2_gen_cache:
            entry = p2_gen_cache[q_text]
            retrieved = entry["retrieved_chunks"]
            answer = entry["answer"]
            source = "phase2_cached"
            n_p2cached += 1
        else:
            retrieved = retriever.search(q_text, top_k=5)
            gen = generate_answer(q_text, retrieved)
            answer = gen["answer"]
            source = "phase2_generated"
            p2_gen_cache[q_text] = {"retrieved_chunks": retrieved, "answer": answer}
            gen_cache_dirty = True
            n_new += 1

        retrieved_pages = [c["page_number"] for c in retrieved]
        results.append({
            "question_id": i,
            "question": q_text,
            "gold_pages": gold_pages,
            "answer": answer,
            "retrieved_pages": retrieved_pages,
            "result_source": source,
            "recall_at_5": round(_compute_recall_at_k(gold_pages, retrieved_pages, 5), 4),
            "mrr": round(_compute_mrr(gold_pages, retrieved_pages), 4),
            "hit_at_5": 1 if set(gold_pages) & set(retrieved_pages[:5]) else 0,
        })

        if i % 20 == 0:
            print(f"  [{i:3d}/{len(text_qs)}] "
                  f"p1={n_phase1} p2c={n_p2cached} new={n_new}")

    retriever.close()
    total_time = time.perf_counter() - total_start

    # Save generation cache
    if gen_cache_dirty:
        with open(gen_cache_path, "w", encoding="utf-8") as f:
            json.dump(p2_gen_cache, f, ensure_ascii=False)

    # ── 3. Compute retrieval metrics ──
    recall5_vals = [r["recall_at_5"] for r in results]
    mrr_vals = [r["mrr"] for r in results]
    hits = sum(r["hit_at_5"] for r in results)

    retrieval_metrics = {
        "recall_at_5": round(sum(recall5_vals) / len(recall5_vals), 4),
        "mrr": round(sum(mrr_vals) / len(mrr_vals), 4),
        "top5_hit_count": hits,
        "top5_hit_rate": round(hits / len(results), 4),
    }

    # ── 4. RAGAS evaluation ──
    from ragas import evaluate as ragas_evaluate
    # ragas 0.4.3: use deprecated import path (collections require llm param, buggy)
    from ragas.metrics import (  # noqa
        faithfulness, context_precision, context_recall,
    )
    from openai import OpenAI
    from ragas.llms import llm_factory
    from datasets import Dataset as HFDataset
    from src.config.settings import settings

    # Configure ragas LLM
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    ragas_llm = llm_factory(settings.llm_model, client=client)

    # Sample for ragas (30 stratified, to manage API cost)
    import random
    random.seed(42)
    by_diff = {}
    for r in results:
        q_info = questions[r["question_id"] - 1]
        by_diff.setdefault(q_info.get("difficulty", "easy"), []).append(r)
    sample = []
    for diff in ["easy", "medium", "hard"]:
        pool = by_diff.get(diff, [])
        sample.extend(random.sample(pool, min(10, len(pool))))

    # Build ragas dataset
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    # Build lookup from full results for contexts
    results_by_q = {r["question"]: r for r in results}

    for r in sample:
        q_info = questions[r["question_id"] - 1]
        # Get retrieved chunks from caches
        q_text = r["question"]
        retrieved_for_contexts = []
        if q_text in phase1_cache:
            retrieved_for_contexts = phase1_cache[q_text].get("retrieved_chunks", [])
        elif q_text in p2_gen_cache:
            retrieved_for_contexts = p2_gen_cache[q_text].get("retrieved_chunks", [])
        contexts = [c["content"] for c in retrieved_for_contexts] if retrieved_for_contexts else ["（无检索结果）"]

        ragas_data["question"].append(r["question"])
        ragas_data["answer"].append(r["answer"])
        ragas_data["contexts"].append(contexts if contexts else ["（无检索结果）"])
        ragas_data["ground_truth"].append(q_info.get("reference_answer", ""))

    ds = HFDataset.from_dict(ragas_data)
    print(f"\nRAGAS evaluation on {len(sample)} questions...")

    # AnswerRelevancy via custom LLM judge (ragas needs embeddings API)
    from src.eval.metrics import answer_relevancy as custom_relevancy

    try:
        ragas_result = ragas_evaluate(
            ds,
            metrics=[faithfulness, context_precision, context_recall],
            llm=ragas_llm,
        )
        ragas_df = ragas_result.to_pandas()
        ragas_metrics = {
            "faithfulness": round(float(ragas_df["faithfulness"].mean()), 4),
            "context_precision": round(float(ragas_df["context_precision"].mean()), 4),
            "context_recall": round(float(ragas_df["context_recall"].mean()), 4),
            "answer_relevancy": round(
                sum(custom_relevancy(r["question"], r["answer"]) for r in sample) / len(sample), 4
            ),
            "sample_size": len(sample),
            "engine": {
                "faithfulness": "ragas==0.4.3",
                "context_precision": "ragas==0.4.3",
                "context_recall": "ragas==0.4.3",
                "answer_relevancy": "custom_llm_judge (no embeddings API available)",
            },
        }
    except Exception as e:
        print(f"  RAGAS eval error: {e}")
        ragas_metrics = {
            "error": str(e),
            "sample_size": len(sample),
            "engine": "ragas==0.4.3 (failed)",
        }

    # ── Tag ragas_sampled questions ──
    sampled_questions = {r["question"] for r in sample}
    for r in results:
        if r["question"] in sampled_questions:
            r["result_source"] += "+ragas_sampled"

    # ── 5. Save V0 Baseline ──
    output = {
        "experiment": "v0_naive",
        "version": "V0",
        "phase": 2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": str(dataset_path),
        "total_questions": len(questions),
        "text_only_questions": len(text_qs),
        "result_sources": {
            "phase1_cached": n_phase1,
            "phase2_cached": n_p2cached,
            "phase2_generated": n_new,
            "ragas_sampled": len(sample),
        },
        "total_time_seconds": round(total_time, 1),
        "retrieval_metrics": retrieval_metrics,
        "ragas_metrics": ragas_metrics,
        "per_question": results,
    }

    output_path = baseline_dir / "phase2_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"V0 Baseline — Phase 2 RAGAS Metrics")
    print(f"{'=' * 60}")
    print(f"  Questions:     {len(text_qs)} text-only")
    print(f"  Recall@5:      {retrieval_metrics['recall_at_5']:.4f}")
    print(f"  MRR:           {retrieval_metrics['mrr']:.4f}")
    print(f"  Top-5 Hit:     {retrieval_metrics['top5_hit_rate']:.4f}")
    for k, v in ragas_metrics.items():
        if isinstance(v, float):
            print(f"  {k}:  {v:.4f}")
    print(f"  Saved to:      {output_path}")


if __name__ == "__main__":
    main()
