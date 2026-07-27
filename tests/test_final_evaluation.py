from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "storage" / "runs" / "final_eval"


def test_final_metrics_has_uniform_versions() -> None:
    path = FINAL / "final_metrics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["total_questions"] == 100
    assert data["dataset_sha256"]
    assert data["source_document_sha256"]
    assert set(data["versions"]) == {"V0", "V1", "V2", "V3", "V4"}
    for version in data["versions"].values():
        assert version["retrieval_metrics"]["evaluated_question_count"] == 100
        assert version["ragas_metrics"]["evaluated_question_count"] == 100


def test_final_summary_has_two_formal_tables() -> None:
    path = FINAL / "evaluation_summary.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    assert "## Retrieval Metrics" in content
    assert "## RAGAS / Answer Metrics" in content
    assert "## V5 Incremental Metrics" in content


def test_final_result_files_are_not_old_20_question_runs() -> None:
    for version in ["v0", "v1", "v2", "v3", "v4"]:
        path = FINAL / f"{version}_results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 100
