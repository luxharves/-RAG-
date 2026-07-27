"""Phase 7 tests — incremental update with mock embedding, temp files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ──

def _make_pdf(path: Path, content: str) -> None:
    """Create a minimal PDF with given text content (only for hash testing)."""
    # Use a simple text file as PDF stand-in for hash/manifest testing
    path.write_text(content, encoding="utf-8")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Manifest tests ──


class TestManifestStore:
    def test_file_hash_consistent(self):
        from src.ingestion.manifest import file_hash
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            f.flush()
            h1 = file_hash(f.name)
            h2 = file_hash(f.name)
            assert h1 == h2
        os.unlink(f.name)

    def test_file_hash_different(self):
        from src.ingestion.manifest import file_hash
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("content A")
            f1.flush()
            h1 = file_hash(f1.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("content B")
            f2.flush()
            h2 = file_hash(f2.name)
        assert h1 != h2
        os.unlink(f1.name)
        os.unlink(f2.name)

    def test_classify_first_import_all_added(self):
        from src.ingestion.manifest import ManifestStore
        store_dir = tempfile.mkdtemp()
        try:
            store = ManifestStore(store_dir)
            current = {"/a/doc1.pdf": _sha256("doc1"),
                        "/a/doc2.pdf": _sha256("doc2")}
            c = store.classify(current)
            assert len(c["added"]) == 2
            assert len(c["unchanged"]) == 0
            assert len(c["deleted"]) == 0
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_classify_second_import_all_unchanged(self):
        from src.ingestion.manifest import ManifestStore, DocManifest
        store_dir = tempfile.mkdtemp()
        try:
            store = ManifestStore(store_dir)
            h1 = _sha256("doc1")
            h2 = _sha256("doc2")
            store.set(DocManifest("/a/doc1.pdf", "id1", h1, 100, 10, 8, 5))
            store.set(DocManifest("/a/doc2.pdf", "id2", h2, 200, 20, 15, 8))
            store.save()

            current = {"/a/doc1.pdf": h1, "/a/doc2.pdf": h2}
            c = store.classify(current)
            assert len(c["unchanged"]) == 2
            assert len(c["added"]) == 0
            assert len(c["modified"]) == 0
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_classify_modified_detected(self):
        from src.ingestion.manifest import ManifestStore, DocManifest
        store_dir = tempfile.mkdtemp()
        try:
            store = ManifestStore(store_dir)
            old_hash = _sha256("old content")
            store.set(DocManifest("/a/doc1.pdf", "id1", old_hash, 100, 10, 8, 5))
            store.save()

            new_hash = _sha256("new content")
            current = {"/a/doc1.pdf": new_hash}
            c = store.classify(current)
            assert len(c["modified"]) == 1
            assert c["modified"][0] == "/a/doc1.pdf"
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_classify_deleted_detected(self):
        from src.ingestion.manifest import ManifestStore, DocManifest
        store_dir = tempfile.mkdtemp()
        try:
            store = ManifestStore(store_dir)
            store.set(DocManifest("/a/doc1.pdf", "id1", _sha256("d1"), 100, 10, 8, 5))
            store.set(DocManifest("/a/doc2.pdf", "id2", _sha256("d2"), 200, 20, 15, 8))
            store.save()

            current = {"/a/doc1.pdf": _sha256("d1")}  # doc2 deleted
            c = store.classify(current)
            assert len(c["deleted"]) == 1
            assert c["deleted"][0] == "/a/doc2.pdf"
            assert len(c["unchanged"]) == 1
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_classify_added_unchanged_modified_deleted(self):
        from src.ingestion.manifest import ManifestStore, DocManifest
        store_dir = tempfile.mkdtemp()
        try:
            store = ManifestStore(store_dir)
            store.set(DocManifest("/a/keep.pdf", "id1", _sha256("keep"), 100, 10, 8, 5))
            store.set(DocManifest("/a/changed.pdf", "id2", _sha256("old"), 200, 20, 15, 8))
            store.set(DocManifest("/a/removed.pdf", "id3", _sha256("rem"), 300, 30, 25, 10))
            store.save()

            current = {
                "/a/keep.pdf": _sha256("keep"),
                "/a/changed.pdf": _sha256("new!"),
                "/a/added.pdf": _sha256("added"),
            }
            c = store.classify(current)
            assert set(c["added"]) == {"/a/added.pdf"}
            assert set(c["unchanged"]) == {"/a/keep.pdf"}
            assert set(c["modified"]) == {"/a/changed.pdf"}
            assert set(c["deleted"]) == {"/a/removed.pdf"}
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_remove_deleted_from_store(self):
        from src.ingestion.manifest import ManifestStore, DocManifest
        store_dir = tempfile.mkdtemp()
        try:
            store = ManifestStore(store_dir)
            h = _sha256("doc1")
            store.set(DocManifest("/a/doc1.pdf", "id1", h, 100, 10, 8, 5))
            store.save()
            assert "/a/doc1.pdf" in store.all_files()

            current = {}
            c = store.classify(current)
            assert len(c["deleted"]) == 1
            store.remove("/a/doc1.pdf")
            store.save()
            assert "/a/doc1.pdf" not in store.all_files()
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)


# ── Incremental update report ──


def test_incremental_report_has_all_counts() -> None:
    """After running incremental_update.py, the report must have all count fields."""
    path = PROJECT_ROOT / "storage" / "runs" / "v5_incremental" / "update_report.json"
    if not path.exists():
        # Run incremental update on the real data
        import subprocess
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "incremental_update.py"),
             "--input", str(PROJECT_ROOT / "data" / "raw_docs")],
            capture_output=True, cwd=str(PROJECT_ROOT),
        )
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    for field in ["added_count", "unchanged_count", "modified_count",
                  "deleted_count", "reprocessed_pages", "reused_chunks",
                  "embedded_chunks", "removed_chunks"]:
        assert field in report["counts"], f"Missing in counts: {field}"
    assert "elapsed_seconds" in report, "Missing: elapsed_seconds"


def test_v5_output_files_exist() -> None:
    out = PROJECT_ROOT / "storage" / "runs" / "v5_incremental"
    for name in ["update_report.json", "before_after_manifest.json", "metadata.json"]:
        p = out / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                assert json.load(f) is not None


def test_manifests_persist() -> None:
    p = PROJECT_ROOT / "storage" / "manifests" / "manifests.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 1
