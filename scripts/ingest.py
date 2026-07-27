#!/usr/bin/env python
"""Phase 1 — Ingestion Pipeline (V0 Naive RAG).

Parses a PDF, chunks it, embeds chunks with BGE-M3, and stores them in Milvus.

Usage:
    python scripts/ingest.py --config configs/experiments/v0_naive.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ⚠️  pymilvus ORM module reads MILVUS_URI at import time and only
# accepts http:// URIs. Save the real path from .env before overriding.
import os as _os
from dotenv import dotenv_values as _dotenv_values
_env_vals = _dotenv_values(str(PROJECT_ROOT / ".env"))
_REAL_MILVUS_URI = _env_vals.get("MILVUS_URI", "milvus.db")
_os.environ["MILVUS_URI"] = "http://localhost:19530"


def main() -> None:
    parser = argparse.ArgumentParser(description="V0 Naive RAG — Ingestion")
    parser.add_argument(
        "--config",
        default="configs/experiments/v0_naive.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="PDF file to ingest (overrides config)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in characters",
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

    print(f"Config: {config_path}")
    print(f"Experiment: {config.get('name', 'v0_naive')}")

    # ── Find PDF ──
    pdf_path = args.pdf
    if not pdf_path:
        raw_docs = PROJECT_ROOT / "data" / "raw_docs"
        pdfs = list(raw_docs.glob("*.pdf"))
        if not pdfs:
            print("ERROR: No PDF found in data/raw_docs/")
            sys.exit(1)
        pdf_path = str(pdfs[0])
    print(f"PDF: {pdf_path}")

    # ── Parse PDF ──
    from src.ingestion.pdf_parser import parse_pdf

    t0 = time.perf_counter()
    print("Parsing PDF...")
    doc = parse_pdf(pdf_path)
    print(f"  Pages with text: {len(doc.pages)} / {doc.metadata['total_pages']}")
    print(f"  Document ID: {doc.document_id}")
    print(f"  Version: {doc.version}")

    # ── Chunk ──
    from src.ingestion.chunker import chunk_document

    print(f"Chunking (size={args.chunk_size}, overlap={args.overlap})...")
    chunks = chunk_document(doc, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"  Total chunks: {len(chunks)}")

    # ── Embed ──
    from src.infra.embedder import Embedder

    print("Loading BGE-M3 embedder...")
    embedder = Embedder()
    embedder.load()
    print(f"  Device: {embedder.device}, Dim: {embedder.dim}")

    print("Embedding chunks...")
    texts = [c.content for c in chunks]
    vectors = embedder.encode_batch(texts)
    print(f"  Vectors: {len(vectors)}")

    # ── Store in Milvus ──
    from src.config.settings import settings

    collection_name = "v0_naive_rag"
    print(f"Storing in Milvus collection: {collection_name}")

    from pymilvus import MilvusClient

    _milvus_path = _REAL_MILVUS_URI
    if not _milvus_path.startswith("http"):
        _milvus_path = str(PROJECT_ROOT / _milvus_path)
    client = MilvusClient(_milvus_path)

    # Drop if exists; on Windows Milvus Lite this may fail
    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
    except Exception:
        pass

    # Create fresh collection (handle case where drop silently failed)
    if client.has_collection(collection_name):
        print("  Warning: collection already exists, reusing")
    else:
        client.create_collection(
            collection_name=collection_name,
            dimension=embedder.dim,
            metric_type="COSINE",
            auto_id=True,
        )

    # Insert data
    data = []
    for chunk, vector in zip(chunks, vectors):
        data.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_version": chunk.document_version,
            "page_number": chunk.page_number,
            "content": chunk.content,
            "content_type": chunk.content_type,
            "source_file": chunk.source_file,
            "vector": vector,
        })

    res = client.insert(collection_name=collection_name, data=data)
    print(f"  Inserted: {res['insert_count']} rows")

    # Milvus Lite auto-creates a default index; ensure collection is loaded
    client.load_collection(collection_name)
    print("  Collection loaded and ready.")

    elapsed = time.perf_counter() - t0
    print(f"\nIngestion complete in {elapsed:.1f}s")
    print(f"  Document: {doc.source_file}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Collection: {collection_name}")

    # ── Save run metadata ──
    run_dir = PROJECT_ROOT / "storage" / "runs" / "v0_ingest"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "experiment": config.get("name", "v0_naive"),
        "version": config.get("version", "V0"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pdf": pdf_path,
        "document_id": doc.document_id,
        "document_version": doc.version,
        "total_pages_in_pdf": doc.metadata["total_pages"],
        "pages_with_text": len(doc.pages),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "num_chunks": len(chunks),
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.dim,
        "collection_name": collection_name,
        "duration_seconds": round(elapsed, 1),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print(f"Run metadata saved to: {run_dir / 'metadata.json'}")

    client.close()


if __name__ == "__main__":
    main()
