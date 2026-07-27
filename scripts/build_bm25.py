#!/usr/bin/env python
"""Phase 4 — Build BM25 persistent index from V1 keyword-enhanced collection."""

from __future__ import annotations

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
    from pymilvus import MilvusClient
    from src.retrieval.bm25 import BM25Retriever

    milvus_path = str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(milvus_path)

    # Find latest V1 kw collection
    v1_kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    v1_ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    v1_all = v1_ts + v1_kw
    if not v1_all:
        print("ERROR: No V1 collection found")
        sys.exit(1)
    col = v1_all[-1]
    print(f"Source: {col}")

    client.load_collection(col)
    chunks = client.query(
        collection_name=col,
        filter="chunk_id != \"\"",
        output_fields=["chunk_id", "page_number", "content_type",
                       "source_file", "content"],
        limit=200,
    )
    client.close()
    print(f"Loaded {len(chunks)} chunks")

    t0 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.build(chunks)
    idx_path = PROJECT_ROOT / "storage" / "bm25"
    bm25.save(idx_path)
    elapsed = time.perf_counter() - t0

    print(f"BM25 index saved: {idx_path}")
    print(f"  Documents: {bm25.num_docs}")
    print(f"  Time: {elapsed:.1f}s")

    # Quick test
    results = bm25.search("机器人会不会从楼梯摔下去", top_k=3)
    print(f"\nTest query: 机器人会不会从楼梯摔下去")
    for r in results:
        print(f"  page={r['page_number']} type={r['content_type']} score={r['bm25_score']:.4f}")


if __name__ == "__main__":
    main()
