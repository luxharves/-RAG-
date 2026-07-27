"""Phase 7 integration — true incremental index (same collection, in-place)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MILVUS_URI"] = "http://localhost:19530"


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    path.write_text("\f".join(page_texts), encoding="utf-8")


@pytest.fixture
def ws():
    """Temp workspace: docs/, milvus.db, bm25/, manifests/."""
    root = Path(tempfile.mkdtemp(prefix="p7incr_"))
    (root / "docs").mkdir()
    yield {
        "root": root, "docs": root / "docs",
        "milvus": str(root / "milvus.db"),
        "bm25": str(root / "bm25"),
        "manifests": root / "manifests",
    }
    shutil.rmtree(root, ignore_errors=True)


def _setup(ws):
    """Create collection, load embedder, return (indexer, collection_name)."""
    from pymilvus import MilvusClient
    from src.infra.embedder import Embedder
    from src.retrieval.bm25 import BM25Retriever
    from src.ingestion.manifest import ManifestStore
    from src.ingestion.incremental import IncrementalIndexer

    # Mock PDF parsing (temp files are plain text, not real PDFs)
    import src.ingestion.incremental as incr_mod
    orig_parse = incr_mod.parse_pdf
    orig_chunk = incr_mod.chunk_document

    def _fake_parse(path):
        from src.ingestion.document import Document, _make_document_id
        import hashlib
        text = Path(path).read_text(encoding="utf-8")
        pages = [p.strip() for p in text.split("\f") if p.strip()]
        if not pages:
            raise ValueError("empty document")
        doc_id = _make_document_id(str(path))
        return Document(
            document_id=doc_id,
            source_file=str(path),
            version=hashlib.sha256(text.encode()).hexdigest()[:16],
            pages=pages,
            page_numbers=list(range(1, len(pages) + 1)),
            metadata={"total_pages": len(pages)},
        )

    def _fake_chunk(doc, chunk_size=500, overlap=50):
        from src.ingestion.document import Chunk
        chunks = []
        for seq, (pn, text) in enumerate(zip(doc.page_numbers, doc.pages)):
            cid = hashlib.sha256(f"{doc.document_id}|p{pn}|s{seq}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                document_id=doc.document_id,
                document_version=doc.version,
                page_number=pn,
                content=text,
                content_type="text",
                source_file=doc.source_file,
                seq=seq,
                chunk_id=cid,
            ))
        return chunks

    incr_mod.parse_pdf = _fake_parse
    incr_mod.chunk_document = _fake_chunk

    embedder = Embedder()
    embedder.load()

    client = MilvusClient(ws["milvus"])
    col = f"incr_test_{int(time.time() * 1000)}"
    client.create_collection(col, dimension=embedder.dim, metric_type="COSINE", auto_id=True)

    bm25 = BM25Retriever()
    bm25.build([])  # empty init
    bm25.save(ws["bm25"])

    store = ManifestStore(ws["manifests"])

    indexer = IncrementalIndexer(client, col, bm25, store, embedder)
    return indexer, col, client, embedder


class TestIncrementalSameCollection:
    """All operations on the SAME collection — true incremental."""

    def test_add_then_retrieve(self, ws):
        _make_pdf(ws["docs"] / "a.pdf", ["安全信息 温度范围0-40度", "产品介绍 按键说明"])
        indexer, col, client, _ = _setup(ws)

        counts = indexer.process(str(ws["docs"]))
        assert counts["added"] == 1
        assert counts["embedded_chunks"] == 2
        assert indexer.embed_call_count == 1

        client.load_collection(col)
        results = client.query(col, filter="source_file != \"\"",
                               output_fields=["source_file"], limit=10)
        assert len(results) == 2
        client.close()

    def test_unchanged_zero_embed(self, ws):
        _make_pdf(ws["docs"] / "a.pdf", ["第一页", "第二页"])
        indexer, col, client, embedder = _setup(ws)

        # First pass
        counts1 = indexer.process(str(ws["docs"]))
        assert counts1["embedded_chunks"] == 2
        calls_before = indexer.embed_call_count

        # Second pass — unchanged
        counts2 = indexer.process(str(ws["docs"]))
        assert counts2["unchanged"] == 1
        assert counts2["reused_chunks"] == 2
        assert counts2["embedded_chunks"] == 0
        assert counts2["added"] == 0
        assert indexer.embed_call_count == calls_before, \
            "Unchanged doc must not call embedder"

        client.load_collection(col)
        results = client.query(col, filter="source_file != \"\"",
                               output_fields=["source_file"], limit=10)
        assert len(results) == 2
        client.close()

    def test_modify_replaces_in_place(self, ws):
        _make_pdf(ws["docs"] / "doc.pdf", ["旧版本内容 页面1", "旧版本内容 页面2"])
        indexer, col, client, _ = _setup(ws)

        # Initial
        indexer.process(str(ws["docs"]))
        client.load_collection(col)
        old_results = client.query(col, filter="content like \"%旧版本%\"",
                                   output_fields=["content"], limit=10)
        assert len(old_results) == 2

        # Modify
        _make_pdf(ws["docs"] / "doc.pdf", ["新版本内容 页面1", "新版本内容 页面2"])
        counts = indexer.process(str(ws["docs"]))
        assert counts["modified"] == 1
        assert counts["embedded_chunks"] == 2
        assert counts["removed_chunks"] == 2  # old chunks deleted

        # Same collection — old content GONE, new content PRESENT
        old_after = client.query(col, filter="content like \"%旧版本%\"",
                                 output_fields=["content"], limit=10)
        assert len(old_after) == 0, "Old chunks must be deleted from same collection"

        new_after = client.query(col, filter="content like \"%新版本%\"",
                                 output_fields=["content"], limit=10)
        assert len(new_after) == 2, "New chunks must be present"
        client.close()

    def test_delete_removes_from_index(self, ws):
        _make_pdf(ws["docs"] / "keep.pdf", ["保留文档"])
        _make_pdf(ws["docs"] / "del.pdf", ["删除文档"])
        indexer, col, client, _ = _setup(ws)

        indexer.process(str(ws["docs"]))
        client.load_collection(col)

        # Both present
        all_sources = {r["source_file"] for r in client.query(
            col, filter="source_file != \"\"", output_fields=["source_file"], limit=20)}
        assert any("del.pdf" in s for s in all_sources)
        assert any("keep.pdf" in s for s in all_sources)

        # Delete
        (ws["docs"] / "del.pdf").unlink()
        counts = indexer.process(str(ws["docs"]))
        assert counts["deleted"] == 1
        assert counts["removed_chunks"] >= 1

        # Same collection — deleted content GONE, kept content PRESENT
        after_sources = {r["source_file"] for r in client.query(
            col, filter="source_file != \"\"", output_fields=["source_file"], limit=20)}
        assert not any("del.pdf" in s for s in after_sources), \
            "Deleted doc must not be retrievable"
        assert any("keep.pdf" in s for s in after_sources), \
            "Kept doc must still be retrievable"
        client.close()

    def test_bm25_incremental_update(self, ws):
        """BM25 must reflect adds and deletes in-place."""
        _make_pdf(ws["docs"] / "bm.pdf", ["BM25测试文档 包含关键词温度"])
        indexer, col, client, _ = _setup(ws)

        indexer.process(str(ws["docs"]))

        # Check in-memory BM25 directly
        bm = indexer.bm25
        assert bm.num_docs == 1

        hits = bm.search("温度", top_k=3)
        assert len(hits) >= 1
        assert any("bm.pdf" in h.get("source_file", "") for h in hits)

        # Add another doc
        _make_pdf(ws["docs"] / "bm2.pdf", ["另一文档 包含关键词电压"])
        indexer.process(str(ws["docs"]))
        assert bm.num_docs == 2
        hits2 = bm.search("电压", top_k=3)
        assert len(hits2) >= 1

        # Delete first doc
        (ws["docs"] / "bm.pdf").unlink()
        indexer.process(str(ws["docs"]))
        assert bm.num_docs == 1
        hits3 = bm.search("温度", top_k=3)
        assert not any("bm.pdf" in h.get("source_file", "") for h in hits3), \
            "Deleted doc must not appear in BM25 results"

    def test_failure_preserves_state(self, ws):
        """Embedding failure must not corrupt index or manifest."""
        _make_pdf(ws["docs"] / "s1.pdf", ["稳定文档 第一页"])
        _make_pdf(ws["docs"] / "s2.pdf", ["另一个稳定文档"])
        indexer, col, client, embedder = _setup(ws)

        indexer.process(str(ws["docs"]))
        client.load_collection(col)
        before = client.query(col, filter="source_file != \"\"",
                              output_fields=["source_file"], limit=20)
        before_sources = {r["source_file"] for r in before}
        assert len(before_sources) >= 2

        # Corrupt embedder
        orig_encode = embedder.encode_batch
        def failing_encode(*args, **kwargs):
            raise RuntimeError("simulated embed failure")
        embedder.encode_batch = failing_encode

        # Add a new doc — should fail
        _make_pdf(ws["docs"] / "s3.pdf", ["新文档 会失败"])
        try:
            indexer.process(str(ws["docs"]))
        except Exception:
            pass

        # Restore embedder
        embedder.encode_batch = orig_encode

        # Old data must still be retrievable
        after = client.query(col, filter="source_file != \"\"",
                             output_fields=["source_file"], limit=20)
        after_sources = {r["source_file"] for r in after}
        for s in before_sources:
            assert s in after_sources, f"Pre-existing source {s} lost after failure"

        # Manifest must still have old entries
        assert len(indexer.store.all_files()) >= 2
        client.close()
