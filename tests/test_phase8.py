"""Phase 8 tests — FastAPI routes with mocked retriever/VQA (no Milvus locks)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    """Test client with mocked retriever and VQA to avoid Milvus Lite locks."""
    with patch("src.api.deps.get_milvus_client") as mock_mc, \
         patch("src.api.deps.get_retriever") as mock_ret, \
         patch("src.api.deps.get_vqa") as mock_vqa, \
         patch("src.api.deps.get_bm25") as mock_bm, \
         patch("src.api.deps.get_latest_v1_collection") as mock_col, \
         patch("src.api.routes.get_semantic_cache") as mock_cache:

        mock_col.return_value = "test_collection"
        mock_cache.return_value = MagicMock()
        mock_cache.return_value.get.return_value = None  # always a cache miss in tests
        mock_mc.return_value = MagicMock()
        mock_bm.return_value = MagicMock()
        mock_bm.return_value.num_docs = 48

        # Mock retriever
        mock_ret.return_value = MagicMock()
        mock_ret.return_value.collection_name = "test_collection"

        # Mock VQA
        def fake_run(question):
            refused = "核聚变" in question or "火星" in question
            return {
                "question": question,
                "retrieved_chunks": [
                    {"chunk_id": "c1", "source_file": "/a/manual.pdf",
                     "page_number": 24, "content_type": "text",
                     "rerank_score": 0.5},
                ],
                "answer": "根据现有说明书内容无法回答此问题。" if refused
                          else "请检查电源线，确认电池有电后重启。[来源: manual.pdf, p24]",
                "final_status": "refused" if refused else "answered",
                "citations": [],
                "verification_result": {
                    "supported": not refused,
                    "confidence": 0.9,
                    "unsupported_claims": [],
                    "evidence_chunk_ids": ["c1"],
                    "reason": "matches",
                },
                "retry_count": 0,
                "trace": ["retrieve", "generate", "verify", "decide"],
            }

        mock_vqa.return_value.run = fake_run

        from main import app
        return TestClient(app)


class TestHealth:
    def test_health_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "V5"

    def test_health_has_models(self, client):
        r = client.get("/health")
        assert "models" in r.json()


class TestQuery:
    def test_query_schema(self, client):
        r = client.post("/query", json={"question": "设备无法开机怎么办？"})
        assert r.status_code == 200
        data = r.json()
        for field in ["question", "answer", "final_status", "sources",
                       "evidence_chunk_ids", "verification", "timing_s"]:
            assert field in data, f"Missing: {field}"

    def test_query_sources_have_fields(self, client):
        r = client.post("/query", json={"question": "如何清洁集尘盒？"})
        assert r.status_code == 200
        for s in r.json()["sources"]:
            for field in ["chunk_id", "source_file", "page_number", "content_type"]:
                assert field in s

    def test_refusal_for_nonsense(self, client):
        r = client.post("/query", json={"question": "如何更换核聚变反应堆？"})
        assert r.status_code == 200
        assert r.json()["final_status"] == "refused"

    def test_experiment_param(self, client):
        r = client.post("/query", json={"question": "test?", "experiment": "v4"})
        assert r.status_code == 200
        assert r.json()["experiment"] == "v4"


class TestIngest:
    def test_ingest_no_filename(self, client):
        r = client.post("/documents/ingest", files={"file": ("", b"", "text/plain")})
        assert r.status_code in (400, 422)

    def test_ingest_invalid_type(self, client):
        import io
        r = client.post("/documents/ingest",
                        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")})
        assert r.status_code == 422
        assert "Unsupported" in r.text or "unsupported" in r.text.lower()


class TestExperiments:
    def test_not_found(self, client):
        r = client.get("/experiments/nonexistent_xxx")
        assert r.status_code == 404

    def test_found(self, client):
        r = client.get("/experiments/v0_baseline")
        assert r.status_code == 200
        assert "id" in r.json()
        assert "files" in r.json()


class TestEvaluate:
    def test_evaluate_route(self, client):
        r = client.post("/evaluate", json={
            "experiment": "v0_baseline", "dataset": "v0_questions", "max_questions": 3,
        })
        assert r.status_code == 200
        data = r.json()
        assert "run_id" in data
        assert "metrics" in data
        assert "files" in data

    def test_evaluate_not_found(self, client):
        r = client.post("/evaluate", json={"experiment": "nonexistent_xxx"})
        assert r.status_code == 404


class TestSwagger:
    def test_docs(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = list(r.json()["paths"].keys())
        assert "/health" in paths
        assert "/query" in paths
        assert "/documents/ingest" in paths
        assert "/evaluate" in paths
        assert "/experiments/{exp_id}" in paths
