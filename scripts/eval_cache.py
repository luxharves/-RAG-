#!/usr/bin/env python
"""V9 — Semantic cache evaluation.

Phase 1 (warm): run N golden questions through the full pipeline, storing each
    response into the semantic cache.
Phase 2 (measure): re-run the same questions (exact hits) + hand-written
    paraphrases (semantic hits), timing each cached lookup.

Reports: exact/semantic/overall hit rate, cached vs uncached latency, and the
LLM calls saved.

Usage: python scripts/eval_cache.py [--warm 12] [--paraphrases 10]
"""

from __future__ import annotations

import argparse
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

# Hand-written paraphrases of common golden questions (semantic-hit targets)
PARAPHRASES = [
    "设备开不了机是什么原因",
    "扫地机器人怎样才能开机",
    "怎么清洁集尘盒",
    "集尘盒应该如何清洗",
    "机器人会不会从楼梯上掉下去",
    "如何避免机器人摔下楼梯",
    "怎样连接手机App",
    "机器人和手机怎么配对",
    "如何更换拖布",
    "拖布怎么换",
    "机器人一直回充是为什么",
    "为什么机器人老在自动回充",
]


def _latest_col() -> str:
    from pymilvus import MilvusClient
    c = MilvusClient(str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db")))
    kw = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_kw_")])
    ts = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_2")])
    c.close()
    return (ts + kw)[-1]


def _warm_question(vqa, cache, question: str, doc_filter: str) -> float:
    t0 = time.perf_counter()
    state = vqa.run(question, doc_filter=doc_filter)
    resp = {
        "question": question,
        "answer": state.get("answer", ""),
        "final_status": state.get("final_status", "refused"),
        "sources": [{"chunk_id": c.get("chunk_id"), "page_number": c.get("page_number"),
                     "source_file": c.get("source_file")}
                    for c in state.get("retrieved_chunks", [])],
        "verification": {"supported": state.get("verification_result", {}).get("supported", False)},
    }
    cache.put(question, resp)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", type=int, default=12,
                        help="number of golden questions to warm the cache")
    parser.add_argument("--paraphrases", type=int, default=len(PARAPHRASES))
    args = parser.parse_args()

    from src.infra.embedder import Embedder
    from src.infra.reranker import Reranker
    from src.infra.semantic_cache import SemanticCache
    from src.retrieval.reranked_retriever import RerankedRetriever
    from src.generation.generator import generate_answer
    from src.workflow.verified_qa import VerifiedQA
    from src.workflow.grounding import GroundingVerifier, CrossEncoderScorer
    from src.eval.doc_registry import resolve_doc_filter

    robo = resolve_doc_filter("Roborock G10S")

    questions = json.loads(
        (PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json").read_text(encoding="utf-8")
    )[: args.warm]

    embedder = Embedder(); embedder.load()
    rr = Reranker(); rr.load()
    retriever = RerankedRetriever(collection_name=_latest_col(),
                                  bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
                                  reranker=rr)
    verifier = GroundingVerifier(scorer=CrossEncoderScorer(rr), scorer_floor=0.1)
    vqa = VerifiedQA(retriever, generate_answer, verifier, max_retries=1)
    # Use a SEPARATE cache db so the eval never pollutes the production cache
    # (the eval stores simplified responses, not full QueryResponse payloads).
    eval_db = PROJECT_ROOT / "storage" / "runs" / "v9_cache" / "eval_cache.db"
    cache = SemanticCache(embedder, eval_db, threshold=0.9)
    cache.clear()

    # ── Phase 1: warm ──
    print(f"Warming {args.warm} questions...", flush=True)
    warm_times = []
    try:
        for i, q in enumerate(questions, 1):
            t = _warm_question(vqa, cache, q["question"], robo)
            warm_times.append(t)
            print(f"  [{i}/{args.warm}] warmed ({t:.1f}s) {q['question'][:24]}", flush=True)
    finally:
        retriever.close()

    # ── Phase 2: measure ──
    exact_hits = semantic_hits = miss = 0
    cached_lat: list[float] = []

    for q in questions:
        t0 = time.perf_counter()
        hit = cache.get(q["question"])
        cached_lat.append(time.perf_counter() - t0)
        if hit is None:
            miss += 1
        elif hit[1] == "exact":
            exact_hits += 1

    for p in PARAPHRASES[: args.paraphrases]:
        t0 = time.perf_counter()
        hit = cache.get(p)
        cached_lat.append(time.perf_counter() - t0)
        if hit is None:
            miss += 1
        elif hit[1] == "semantic":
            semantic_hits += 1
        elif hit[1] == "exact":
            exact_hits += 1

    total_lookups = len(questions) + min(args.paraphrases, len(PARAPHRASES))
    total_hits = exact_hits + semantic_hits
    avg_warm = sum(warm_times) / len(warm_times)
    avg_cached = sum(cached_lat) / len(cached_lat) if cached_lat else 0.0
    llm_saved = exact_hits  # exact re-runs would each have cost one LLM call

    report = {
        "warmed": len(questions),
        "lookups": total_lookups,
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "miss": miss,
        "overall_hit_rate": round(total_hits / total_lookups, 4),
        "avg_uncached_s": round(avg_warm, 2),
        "avg_cached_s": round(avg_cached, 3),
        "llm_calls_saved": llm_saved,
        "cache_stats": cache.stats(),
    }
    out = PROJECT_ROOT / "storage" / "runs" / "v9_cache"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*52}")
    print(f"warmed {report['warmed']} | lookups {total_lookups}")
    print(f"exact_hits={exact_hits} semantic_hits={semantic_hits} miss={miss}")
    print(f"overall_hit_rate={report['overall_hit_rate']}")
    print(f"avg_uncached={avg_warm:.1f}s  avg_cached={avg_cached*1000:.0f}ms")
    print(f"LLM calls saved: {llm_saved} (exact re-runs)")
    print(f"cache entries: {cache.stats()['entries']}")
    print(f"Saved: {out / 'cache_eval.json'}")


if __name__ == "__main__":
    main()
