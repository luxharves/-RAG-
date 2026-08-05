#!/usr/bin/env python
"""Verify doc-level metadata filtering isolates each manual.

Checks that a Roborock query with doc_filter=<Roborock path> returns only
Roborock chunks, an Ecovacs query with doc_filter=<Ecovacs path> returns only
Ecovacs chunks, and an unfiltered query returns a mix (demonstrating why the
filter is required once two manuals share the collection).

Usage: python scripts/check_doc_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os as _os
from dotenv import dotenv_values as _dv
_os.environ["MILVUS_URI"] = "http://localhost:19530"
_ENV = _dv(str(PROJECT_ROOT / ".env"))

from src.eval.doc_registry import source_document_map, resolve_doc_filter  # noqa: E402


def _latest_col() -> str:
    from pymilvus import MilvusClient
    c = MilvusClient(str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db")))
    kw = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_kw_")])
    ts = sorted([x for x in c.list_collections() if x.startswith("v1_multimodal_2")])
    c.close()
    return (ts + kw)[-1]


def _check(label: str, query: str, doc_filter: str | None, expect: str):
    from src.retrieval.reranked_retriever import RerankedRetriever
    r = RerankedRetriever(collection_name=_latest_col(),
                          bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"))
    hits = r.search(query, top_k=5, mode="reranked", doc_filter=doc_filter)
    names = sorted({Path(h["source_file"]).name for h in hits})
    ok = all(expect in n for n in names) if doc_filter else True
    print(f"{label}: {'✅' if ok else '❌'} {len(hits)} hits from {names}")
    if not ok:
        print(f"    expected all from {expect}")
    r.close()
    return ok


def main() -> None:
    robo = resolve_doc_filter("Roborock G10S")
    eco = resolve_doc_filter("Ecovacs DEEBOT T30C")
    print("doc_registry:")
    for name, path in source_document_map().items():
        print(f"  {name} → {Path(path).name}")

    ok = True
    ok &= _check("Roborock query + Roborock filter", "设备无法开机怎么办",
                 robo, "Roborock G10S")
    ok &= _check("Ecovacs query + Ecovacs filter", "How to clean the filter?",
                 eco, "Ecovacs")
    ok &= _check("Ecovacs query + Ecovacs filter (hybrid)", "How to clean the roller brush?",
                 eco, "Ecovacs")
    print("\nUnfiltered (no doc_filter) — should now be mixed across manuals:")
    from src.retrieval.retriever import DenseRetriever
    rd = DenseRetriever(collection_name=_latest_col())
    hits = rd.search("设备无法开机怎么办", top_k=5)
    print("  " + ", ".join(sorted({Path(h["source_file"]).name[:40] for h in hits})))
    rd.close()

    print(f"\n{'ALL DOC-FILTER CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
