"""Phase 9C tests — /documents, /documents/ingest with incremental stats."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    """Mocked test client."""
    from unittest.mock import patch, MagicMock
    with patch("src.api.deps.get_milvus_client"), \
         patch("src.api.deps.get_retriever"), \
         patch("src.api.deps.get_vqa"), \
         patch("src.api.deps.get_bm25"), \
         patch("src.api.deps.get_latest_v1_collection") as mock_col, \
         patch("src.api.deps.get_incremental_indexer") as mock_idx, \
         patch("src.api.deps.get_settings") as mock_set, \
         patch("src.api.deps.get_embedder"):
        mock_col.return_value = "test_collection"

        # Mock settings
        ms = MagicMock()
        ms.data_dir = "data"
        mock_set.return_value = ms

        # Mock IncrementalIndexer
        idx = MagicMock()
        idx.process.return_value = {
            "added": 1, "unchanged": 0, "modified": 0, "deleted": 0,
            "reprocessed_pages": 2, "reused_chunks": 0,
            "embedded_chunks": 2, "removed_chunks": 0,
        }
        mock_idx.return_value = idx

        # Mock manifest store for /documents
        with patch("src.ingestion.manifest.ManifestStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.all_files.return_value = {"D:/data/manual.pdf"}
            from src.ingestion.manifest import DocManifest
            mock_store.get.return_value = DocManifest(
                source_file="D:/data/manual.pdf", document_id="abc123",
                file_hash="a" * 64, file_size=1000, total_pages=10,
                text_pages=8, num_chunks=5,
            )
            mock_store_cls.return_value = mock_store

            from main import app
            return TestClient(app)


class TestIngest:
    def test_ingest_invalid_type(self, client):
        r = client.post("/documents/ingest", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")})
        assert r.status_code == 422

    def test_ingest_no_filename(self, client):
        r = client.post("/documents/ingest", files={"file": ("", b"", "application/pdf")})
        assert r.status_code in (400, 422)

    def test_ingest_incremental_fields(self, client):
        """Response must include added/unchanged/modified/deleted/reused/embedded/removed."""
        # Mock fitz.open to return a valid doc with 1 page
        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 1
            mock_doc.close = MagicMock()
            mock_fitz.return_value = mock_doc
            content = b"fake pdf content"
            r = client.post("/documents/ingest", files={"file": ("test.pdf", io.BytesIO(content), "application/pdf")})
            assert r.status_code == 200
            data = r.json()
            for field in ["added", "unchanged", "modified", "deleted",
                          "reprocessed_pages", "reused_chunks",
                          "embedded_chunks", "removed_chunks"]:
                assert field in data, f"Missing incremental field: {field}"


class TestDocumentsList:
    def test_documents_returns_list(self, client):
        r = client.get("/documents")
        assert r.status_code == 200
        data = r.json()
        assert "documents" in data
        assert len(data["documents"]) >= 1
        doc = data["documents"][0]
        for field in ["document_id", "source_file", "version", "num_chunks", "status"]:
            assert field in doc, f"Missing: {field}"

    def test_documents_no_absolute_path(self, client):
        r = client.get("/documents")
        assert r.status_code == 200
        for doc in r.json()["documents"]:
            assert "D:" not in doc.get("source_file", ""), "Must not leak absolute path"
            assert "\\" not in doc.get("source_file", ""), "Must not leak absolute path"


class TestHealthUnaffected:
    def test_health_still_works(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
