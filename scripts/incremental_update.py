#!/usr/bin/env python
"""Phase 7 — Incremental document update using IncrementalIndexer.

Usage:
    python scripts/incremental_update.py --input data/raw_docs
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


def _latest_kw_collection(client) -> str:
    kw = sorted([c for c in client.list_collections()
                 if c.startswith("v1_multimodal_kw_")])
    ts = sorted([c for c in client.list_collections()
                 if c.startswith("v1_multimodal_2")])
    return (ts + kw)[-1] if (ts or kw) else "v1_multimodal_kw_latest"


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental document update")
    parser.add_argument("--input", default="data/raw_docs")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--bm25-dir", default="storage/bm25")
    parser.add_argument("--manifests-dir", default="storage/manifests")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()

    from pymilvus import MilvusClient
    from src.infra.embedder import Embedder
    from src.retrieval.bm25 import BM25Retriever
    from src.ingestion.manifest import ManifestStore
    from src.ingestion.incremental import IncrementalIndexer

    input_dir = PROJECT_ROOT / args.input
    milvus_path = str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db"))

    # ── Setup ──
    client = MilvusClient(milvus_path)
    collection = args.collection or _latest_kw_collection(client)
    print(f"Collection: {collection}")

    embedder = Embedder()
    embedder.load()
    print(f"Embedder: {embedder.device}, dim={embedder.dim}")

    bm25 = BM25Retriever()
    bm25_path = PROJECT_ROOT / args.bm25_dir
    if (bm25_path / "bm25_index.pkl").exists():
        bm25.load(bm25_path)
        print(f"BM25 loaded: {bm25.num_docs} docs")
    else:
        bm25.build([])
        print("BM25: empty init")

    store = ManifestStore(PROJECT_ROOT / args.manifests_dir)
    print(f"Manifest: {len(store.all_files())} tracked files")

    indexer = IncrementalIndexer(client, collection, bm25, store, embedder)

    # ── Process ──
    print(f"\nProcessing: {input_dir}")
    counts = indexer.process(str(input_dir))
    elapsed = time.perf_counter() - t0

    # ── Persist BM25 ──
    bm25.save(bm25_path)
    store.save()

    # ── Report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dir": str(input_dir),
        "collection": collection,
        "bm25_dir": str(bm25_path),
        "manifests_dir": str(store.store_dir),
        "counts": {
            "added_count": counts["added"],
            "unchanged_count": counts["unchanged"],
            "modified_count": counts["modified"],
            "deleted_count": counts["deleted"],
            "reprocessed_pages": counts["reprocessed_pages"],
            "reused_chunks": counts["reused_chunks"],
            "embedded_chunks": counts["embedded_chunks"],
            "removed_chunks": counts["removed_chunks"],
        },
        "embed_calls": indexer.embed_call_count,
        "bm25_docs": bm25.num_docs,
        "elapsed_seconds": round(elapsed, 2),
    }

    out_dir = PROJECT_ROOT / "storage" / "runs" / "v5_incremental"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output or str(out_dir / "update_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "v5_incremental", "version": "V5",
            "collection": collection, "report": report_path,
        }, f, ensure_ascii=False, indent=2)

    # ── Print ──
    print(f"\n{'='*55}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  embed_calls: {indexer.embed_call_count}")
    print(f"  bm25_docs: {bm25.num_docs}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"\nReport: {report_path}")

    client.close()


if __name__ == "__main__":
    main()
