#!/usr/bin/env python
"""Phase 1 — Query Pipeline (V0 Naive RAG).

Retrieves top-K chunks from Milvus and generates a cited answer via LLM.

Usage:
    python scripts/query.py --config configs/experiments/v0_naive.yaml --question "设备无法开机怎么办？"
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="V0 Naive RAG — Query")
    parser.add_argument(
        "--config",
        default="configs/experiments/v0_naive.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    # Load config
    import yaml

    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    collection_name = config.get("name", "v0_naive_rag")
    top_k = config.get("retrieval", {}).get("dense_top_k", args.top_k)

    # ── Retrieve ──
    from src.retrieval.retriever import DenseRetriever

    t0 = time.perf_counter()
    retriever = DenseRetriever(collection_name=collection_name)
    retrieved = retriever.search(args.question, top_k=top_k)
    retrieval_time = time.perf_counter() - t0

    # ── Generate ──
    from src.generation.generator import generate_answer

    result = generate_answer(args.question, retrieved)
    gen_time = time.perf_counter() - t0 - retrieval_time

    # ── Output ──
    if args.json:
        output = {
            "question": args.question,
            "answer": result["answer"],
            "retrieved_chunks": retrieved,
            "citations": result["citations"],
            "model": result["model"],
            "usage": result["usage"],
            "timing": {
                "retrieval_seconds": round(retrieval_time, 2),
                "generation_seconds": round(gen_time, 2),
                "total_seconds": round(time.perf_counter() - t0, 2),
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print(f"Q: {args.question}")
        print("=" * 60)
        print(f"\nA: {result['answer']}")
        print(f"\n{'─' * 60}")
        print(f"Sources ({len(retrieved)} chunks):")
        for i, c in enumerate(retrieved, 1):
            fn = Path(c["source_file"]).name if c["source_file"] else "?"
            print(f"  [{i}] {fn}, p{c['page_number']} "
                  f"(score={c['retrieval_score']:.4f})")
            snippet = c['content'][:100].encode('ascii', errors='replace').decode('ascii')
            print(f"      \"{snippet}...\"")
        print(f"\nModel: {result['model']}")
        print(f"Timing: retrieval={retrieval_time:.1f}s, "
              f"gen={gen_time:.1f}s, total={time.perf_counter()-t0:.1f}s")

    retriever.close()


if __name__ == "__main__":
    main()
