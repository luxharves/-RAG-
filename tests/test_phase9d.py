"""Phase 9D tests — /experiments, /experiments/{id}, no data fabrication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    from unittest.mock import patch, MagicMock
    with patch("src.api.deps.get_milvus_client"), \
         patch("src.api.deps.get_retriever"), \
         patch("src.api.deps.get_vqa"), \
         patch("src.api.deps.get_bm25"), \
         patch("src.api.deps.get_latest_v1_collection"), \
         patch("src.api.deps.get_settings"), \
         patch("src.api.deps.get_embedder"), \
         patch("src.api.deps.get_incremental_indexer"):
        from main import app
        return TestClient(app)


class TestExperimentsList:
    def test_list_available(self, client):
        r = client.get("/experiments")
        assert r.status_code == 200
        data = r.json()
        assert "experiments" in data
        exps = data["experiments"]
        assert len(exps) >= 1
        for e in exps:
            for field in ["id", "name", "description", "available", "files"]:
                assert field in e

    def test_has_v0_thru_v5(self, client):
        r = client.get("/experiments")
        ids = {e["id"] for e in r.json()["experiments"]}
        for vid in ["v0_baseline", "v1_multimodal", "v2_comparison",
                     "v3_rerank", "v4_verified", "v5_incremental"]:
            assert vid in ids, f"Missing: {vid}"


class TestExperimentDetail:
    def test_detail_returns_metadata(self, client):
        r = client.get("/experiments/v0_baseline")
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert "files" in data

    def test_not_found(self, client):
        r = client.get("/experiments/nonexistent_xxx")
        assert r.status_code == 404

    def test_detail_no_api_key_leak(self, client):
        """Response must not contain API keys or secrets."""
        r = client.get("/experiments/v0_baseline")
        text = r.text.lower()
        assert "sk-" not in text, "API key leaked"
        assert "api_key" not in text

    def test_detail_no_absolute_path(self, client):
        for vid in ["v0_baseline", "v3_rerank", "v4_verified"]:
            r = client.get(f"/experiments/{vid}")
            if r.status_code != 200:
                continue
            text = r.text
            assert "D:\\\\" not in text, f"{vid} leak absolute path"
            assert "C:\\\\" not in text, f"{vid} leak absolute path"


class TestEvaluateUnaffected:
    def test_evaluate_still_works(self, client):
        r = client.post("/evaluate", json={"experiment": "v0_baseline"})
        assert r.status_code == 200
        assert "run_id" in r.json()


class TestNoRagasTrigger:
    def test_experiments_list_readonly(self, client):
        """GET /experiments must be read-only — no file modifications."""
        import os, time
        p = PROJECT_ROOT / "storage" / "runs" / "v0_baseline" / "retrieval_metrics.json"
        mtime_before = os.path.getmtime(p) if p.exists() else 0
        client.get("/experiments")
        client.get("/experiments/v0_baseline")
        if p.exists():
            assert os.path.getmtime(p) == mtime_before, "Experiments endpoints must not modify files"
