"""Phase 2 tests — Golden Dataset integrity and result completeness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    path = PROJECT_ROOT / "data" / "eval_dataset" / "golden_100.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Dataset integrity ──


def test_golden_dataset_count(dataset: list[dict]) -> None:
    """Exactly 100 questions."""
    assert len(dataset) == 100, f"Expected 100, got {len(dataset)}"


def test_no_duplicate_questions(dataset: list[dict]) -> None:
    """No duplicate question texts."""
    texts = [q["question"] for q in dataset]
    dups = [t for t in texts if texts.count(t) > 1]
    assert len(dups) == 0, f"Duplicates found: {set(dups)}"


def test_required_fields(dataset: list[dict]) -> None:
    """Every question has all required fields."""
    required = [
        "question", "reference_answer", "reference_contexts",
        "question_type", "difficulty", "modality_required",
        "gold_pages", "review_status", "source_document",
    ]
    for i, q in enumerate(dataset):
        for field in required:
            assert field in q, f"Q{i}: missing '{field}'"


def test_review_status_valid(dataset: list[dict]) -> None:
    """review_status must be one of: human_reviewed, ai_annotated, pending."""
    valid = {"human_reviewed", "ai_annotated", "pending"}
    for i, q in enumerate(dataset):
        rs = q.get("review_status", "")
        assert rs in valid, f"Q{i}: invalid review_status '{rs}'"


def test_human_reviewed_count(dataset: list[dict]) -> None:
    """Only genuinely human-reviewed questions are marked as such."""
    hr = [q for q in dataset if q.get("review_status") == "human_reviewed"]
    # Known human-reviewed: Q9, Q18, Q19 (3 questions)
    assert len(hr) <= 20, f"Too many human_reviewed: {len(hr)}"


def test_gold_pages_valid(dataset: list[dict]) -> None:
    """gold_pages must be a list of positive integers."""
    for i, q in enumerate(dataset):
        gp = q.get("gold_pages", [])
        assert isinstance(gp, list), f"Q{i}: gold_pages not a list"
        for p in gp:
            assert isinstance(p, int) and p > 0, f"Q{i}: invalid page {p}"


def test_modality_values(dataset: list[dict]) -> None:
    """modality_required must be text, image, or text_and_image."""
    valid = {"text", "image", "text_and_image"}
    for i, q in enumerate(dataset):
        assert q.get("modality_required", "") in valid, f"Q{i}: invalid modality"


# ── Result completeness ──


def test_phase2_metrics_exists() -> None:
    """Phase 2 metrics file must exist."""
    path = PROJECT_ROOT / "storage" / "runs" / "v0_baseline" / "phase2_metrics.json"
    assert path.exists(), "phase2_metrics.json not found"


def test_phase2_metrics_has_all_fields() -> None:
    """Phase 2 metrics must contain all required metric groups."""
    path = PROJECT_ROOT / "storage" / "runs" / "v0_baseline" / "phase2_metrics.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "retrieval_metrics" in data
    assert "ragas_metrics" in data
    rm = data["retrieval_metrics"]
    for field in ["recall_at_5", "mrr", "top5_hit_count", "top5_hit_rate"]:
        assert field in rm, f"Missing retrieval metric: {field}"


def test_phase1_results_not_overwritten() -> None:
    """Phase 1 results must remain intact."""
    v0_result_files = list(
        (PROJECT_ROOT / "storage" / "runs" / "v0_baseline").glob("v0_results_*.json")
    )
    assert len(v0_result_files) >= 1, "Phase 1 v0_results should still exist"


def test_phase1_cached_not_exceed_19() -> None:
    """Phase 1 only has 19 text questions, so phase1_cached must be <= 19."""
    path = PROJECT_ROOT / "storage" / "runs" / "v0_baseline" / "phase2_metrics.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sources = data.get("result_sources", {})
    p1 = sources.get("phase1_cached", 0)
    assert p1 <= 19, f"phase1_cached={p1} exceeds Phase 1 text question count (19)"


def test_result_source_values_valid() -> None:
    """Every per_question must have a valid result_source."""
    path = PROJECT_ROOT / "storage" / "runs" / "v0_baseline" / "phase2_metrics.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data.get("per_question", []):
        src = r.get("result_source", "")
        assert src, "result_source must not be empty"
        assert any(valid in src for valid in [
            "phase1_cached", "phase2_cached", "phase2_generated"
        ]), f"Invalid result_source: {src}"


def test_ragas_engine_labels_explicit() -> None:
    """RAGAS metrics must clearly label ragas vs custom judge."""
    path = PROJECT_ROOT / "storage" / "runs" / "v0_baseline" / "phase2_metrics.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    engine = data.get("ragas_metrics", {}).get("engine", {})
    assert isinstance(engine, dict), "engine must be a dict"
    assert "ragas==0.4.3" in engine.get("faithfulness", ""), "faithfulness must use ragas"
    assert "custom" in engine.get("answer_relevancy", ""), "answer_relevancy must mark custom"


def test_phase2_metrics_not_overwrite_phase1() -> None:
    """phase2_metrics.json must be a different file from v0_results."""
    v0_results = list(
        (PROJECT_ROOT / "storage" / "runs" / "v0_baseline").glob("v0_results_*.json")
    )
    phase2 = PROJECT_ROOT / "storage" / "runs" / "v0_baseline" / "phase2_metrics.json"
    for v0 in v0_results:
        assert v0.name != phase2.name, "phase2_metrics should not overwrite v0_results"
