#!/usr/bin/env python
"""Run RERANKED retrieval (V3/V4, identical in retrieval-only mode) on a
question range, appending to a partial results file.

Each foreground run covers a batch small enough to finish before the 10-min
tool timeout, working around the environment that kills long background tasks.

Usage: python scripts/run_retrieval_batch.py --start 0 --end 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os as _os
from dotenv import dotenv_values as _dv
_os.environ["MILVUS_URI"] = "http://localhost:19530"
_ENV = _dv(str(PROJECT_ROOT / ".env"))

OUT = PROJECT_ROOT / "storage" / "runs" / "final_eval_extended_full"


def _latest_col() -> str:
    from pymilvus import MilvusClient
    c = MilvusClient(str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db")))
    kw = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_kw_")])
    ts = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_2")])
    c.close()
    return (ts + kw)[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--dataset", default="golden_extended.json")
    args = parser.parse_args()

    from src.eval.doc_registry import resolve_doc_filter
    from src.retrieval.reranked_retriever import RerankedRetriever

    questions = json.loads(
        (PROJECT_ROOT / "data" / "eval_dataset" / args.dataset).read_text(encoding="utf-8")
    )[args.start:args.end]

    r = RerankedRetriever(collection_name=_latest_col(),
                          bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"))

    out_path = OUT / "v3_partial.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []

    try:
        for i, q in enumerate(questions, args.start):
            doc_filter = resolve_doc_filter(q.get("source_document"))
            hits = r.search(q["question"], top_k=5, mode="reranked", doc_filter=doc_filter)
            results.append({
                "question_id": i,
                "question": q["question"],
                "source_document": q.get("source_document", ""),
                "gold_pages": q.get("gold_pages", []),
                "retrieved_pages": [int(h.get("page_number", 0)) for h in hits],
                "retrieved_sources": [h.get("source_file", "") for h in hits],
            })
            if (i - args.start + 1) % 5 == 0:
                print(f"  [{i - args.start + 1}/{len(questions)}] {q['question'][:20]}", flush=True)
    finally:
        r.close()
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"batch [{args.start}:{args.end}] done — total saved {len(results)}", flush=True)


if __name__ == "__main__":
    main()
