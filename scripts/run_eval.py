#!/usr/bin/env python
"""Phase 1 — Batch evaluation script. Runs all 20 questions and saves results."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ⚠️  pymilvus ORM requires http:// URI — override env after saving real path.
import os as _os
from dotenv import dotenv_values as _dotenv_values
_env_vals = _dotenv_values(str(PROJECT_ROOT / ".env"))
_REAL_MILVUS_URI = _env_vals.get("MILVUS_URI", "milvus.db")
_os.environ["MILVUS_URI"] = "http://localhost:19530"


def main() -> None:
    from src.retrieval.retriever import DenseRetriever
    from src.generation.generator import generate_answer

    questions_path = PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json"
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running V0 baseline evaluation: {len(questions)} questions")
    print(f"Collection: v0_naive_rag")

    retriever = DenseRetriever(collection_name="v0_naive_rag")
    results = []
    total_start = time.perf_counter()

    for i, q in enumerate(questions, 1):
        q_text = q["question"]
        t0 = time.perf_counter()

        # Retrieve
        retrieved = retriever.search(q_text, top_k=5)
        retrieval_time = time.perf_counter() - t0

        # Generate
        gen_result = generate_answer(q_text, retrieved)
        total_time = time.perf_counter() - t0

        result = {
            "question_id": i,
            "question": q_text,
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "answer": gen_result["answer"],
            "retrieved_chunks": retrieved,
            "citations": gen_result.get("citations", []),
            "model": gen_result.get("model"),
            "usage": gen_result.get("usage"),
            "timing": {
                "retrieval_seconds": round(retrieval_time, 2),
                "generation_seconds": round(total_time - retrieval_time, 2),
                "total_seconds": round(total_time, 2),
            },
        }
        results.append(result)
        print(f"  [{i:2d}/{len(questions)}] {q_text[:40]}... "
              f"({total_time:.1f}s, {len(retrieved)} chunks)")

    total_elapsed = time.perf_counter() - total_start
    retriever.close()

    # ── Save V0 Baseline ──
    run_dir = PROJECT_ROOT / "storage" / "runs" / "v0_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = run_dir / f"v0_results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save summary
    summary = {
        "experiment": "v0_naive",
        "version": "V0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": len(questions),
        "total_time_seconds": round(total_elapsed, 1),
        "avg_time_per_question": round(total_elapsed / len(questions), 1),
        "results_file": str(output_path),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"V0 Baseline saved:")
    print(f"  Results: {output_path}")
    print(f"  Questions: {len(questions)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Avg time/q: {total_elapsed / len(questions):.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
