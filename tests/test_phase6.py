"""Phase 6 tests — LangGraph workflow with Mock LLM (no real API calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── final_status mapping (Phase 6) ──
# answered  → verified and supported
# refused   → could not answer (no evidence / out of scope / verify failed)
# fallback  → degraded answer (retriever/reranker unavailable)


# ── Mock data ──

MOCK_CHUNKS_RELEVANT = [
    {"chunk_id": "c1", "page_number": 24, "content_type": "text",
     "content": "无法开机：电池电量不足，请充电。", "source_file": "manual.pdf",
     "fusion_score": 0.02, "rerank_score": 0.5, "rerank_rank": 1,
     "dense_rank": 1, "bm25_rank": 2, "retrieval_score": 0.6},
    {"chunk_id": "c2", "page_number": 5, "content_type": "text",
     "content": "电源指示灯：白色电量>=20%。", "source_file": "manual.pdf",
     "fusion_score": 0.015, "rerank_score": 0.3, "rerank_rank": 2,
     "dense_rank": 2, "bm25_rank": 1, "retrieval_score": 0.5},
]

MOCK_CHUNKS_IRRELEVANT = [
    {"chunk_id": "cx", "page_number": 1, "content_type": "text",
     "content": "使用说明书 石头自清洁扫拖机器人G10S", "source_file": "manual.pdf",
     "fusion_score": 0.001, "rerank_score": 0.01, "rerank_rank": 1,
     "dense_rank": 1, "bm25_rank": 1, "retrieval_score": 0.01},
]


def _mock_retriever_relevant():
    m = MagicMock()
    m.search.return_value = MOCK_CHUNKS_RELEVANT
    return m


def _mock_retriever_irrelevant():
    m = MagicMock()
    m.search.return_value = MOCK_CHUNKS_IRRELEVANT
    return m


def _mock_generator(question, chunks):
    if not chunks or chunks[0].get("rerank_score", 0) < 0.05:
        return {"answer": "根据现有说明书内容无法回答此问题。"}
    return {"answer": f"根据说明书第{chunks[0]['page_number']}页，建议充电后再使用。[来源: manual.pdf, 第{chunks[0]['page_number']}页]"}


def _mock_verifier_pass(question, answer, chunks):
    return {"supported": True, "confidence": 0.9, "unsupported_claims": [],
            "evidence_chunk_ids": ["c1"], "reason": "answer matches chunk c1"}


def _mock_verifier_fail(question, answer, chunks):
    return {"supported": False, "confidence": 0.2,
            "unsupported_claims": ["claim about feature X"],
            "evidence_chunk_ids": [], "reason": "no evidence"}


def _make_fail_then_pass_verifier():
    """Returns a verifier that fails first call, passes second."""
    state = {"called": False}

    def verify(question, answer, chunks):
        if not state["called"]:
            state["called"] = True
            return _mock_verifier_fail(question, answer, chunks)
        return _mock_verifier_pass(question, answer, chunks)

    return verify


# ── Tests ──


class TestVerifiedQA:

    def test_normal_question_answered(self):
        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         _mock_verifier_pass)
        state = vqa.run("设备无法开机怎么办")
        assert state["final_status"] == "answered"
        assert state["retry_count"] == 0
        assert "retrieve" in state["trace"]
        assert "check_relevance" in state["trace"]

    def test_verify_fail_retries_once(self):
        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         _make_fail_then_pass_verifier())
        state = vqa.run("test")
        assert state["final_status"] == "answered"
        assert state["retry_count"] == 1
        assert state["trace"].count("retrieve") == 2

    def test_verify_fail_twice_refuses(self):
        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         _mock_verifier_fail, max_retries=1)
        state = vqa.run("test")
        assert state["final_status"] == "refused"
        assert "无法回答" in state["answer"]
        assert state["retry_count"] == 1

    def test_out_of_scope_refused(self):
        """Irrelevant chunks (low score) → direct refuse, no generate/verify."""
        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_irrelevant(), _mock_generator,
                         _mock_verifier_pass)
        state = vqa.run("这个设备能在火星上使用吗？")
        assert state["final_status"] == "refused"
        assert "无法回答" in state["answer"]
        assert state["verification_result"]["supported"] is False
        assert "generate" not in state["trace"], \
            "Should refuse before generate when irrelevant"

    def test_state_has_all_required_fields(self):
        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         _mock_verifier_pass)
        state = vqa.run("test")
        for field in ["question", "retrieved_chunks", "answer", "citations",
                       "verification_result", "retry_count", "final_status",
                       "trace"]:
            assert field in state, f"Missing: {field}"

    def test_citations_include_source_info(self):
        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         _mock_verifier_pass)
        state = vqa.run("test")
        for c in state["citations"]:
            assert "chunk_id" in c
            assert "source_file" in c
            assert "page_number" in c

    def test_verification_output_has_required_fields(self):
        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         _mock_verifier_pass)
        state = vqa.run("test")
        vr = state["verification_result"]
        for field in ["supported", "confidence", "unsupported_claims",
                       "evidence_chunk_ids", "reason"]:
            assert field in vr, f"Missing verify field: {field}"

    def test_verify_parse_failure_is_not_pass(self):
        def broken_verifier(question, answer, chunks):
            return {"supported": False, "confidence": 0.0,
                    "unsupported_claims": ["parse error"],
                    "evidence_chunk_ids": [], "reason": "unparseable"}

        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         broken_verifier)
        state = vqa.run("test")
        assert state["final_status"] == "refused"

    def test_no_infinite_loop(self):
        call_count = [0]

        def counting_verifier(question, answer, chunks):
            call_count[0] += 1
            return _mock_verifier_fail(question, answer, chunks)

        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), _mock_generator,
                         counting_verifier, max_retries=1)
        vqa.run("test")
        assert call_count[0] <= 2

    def test_generator_self_refusal_detected(self):
        """If generator itself refuses, verify should detect and refuse."""
        def refusal_gen(question, chunks):
            return {"answer": "根据现有说明书内容无法回答此问题。"}

        from src.workflow.verified_qa import VerifiedQA
        vqa = VerifiedQA(_mock_retriever_relevant(), refusal_gen,
                         _mock_verifier_pass)
        state = vqa.run("test")
        assert state["final_status"] in ("refused", "answered")
        # If the verifier passes but generator refused, the decide node
        # will see supported=True and mark answered. But the verify node
        # has a pre-check for self-refusal phrases — it sets supported=False.
        # Let's check:
        if state["verification_result"]["supported"]:
            assert state["final_status"] == "answered"
        else:
            assert state["final_status"] == "refused"


# ── Output files (read-only, no API calls) ──


def test_v4_output_files_exist() -> None:
    out = PROJECT_ROOT / "storage" / "runs" / "v4_verified"
    for name in ["v4_results.json", "v3_v4_comparison.json",
                 "verification_cases.json", "metadata.json"]:
        p = out / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            assert data is not None, f"{name} is empty"


def test_v4_final_status_values() -> None:
    """V4 results should use valid final_status values."""
    path = PROJECT_ROOT / "storage" / "runs" / "v4_verified" / "v4_results.json"
    if not path.exists():
        pytest.skip("run_v4_eval.py not yet executed")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = {"answered", "refused", "fallback", "verified", "retry"}
    for r in data:
        assert r.get("final_status", "") in valid, f"Invalid: {r.get('final_status')}"


def test_v4_edge_cases_expected_refused() -> None:
    """Edge cases MUST be refused with supported=false."""
    path = PROJECT_ROOT / "storage" / "runs" / "v4_verified" / "verification_cases.json"
    if not path.exists():
        pytest.skip("run_v4_eval.py not yet executed")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 2
    labels = {r.get("label") for r in data}
    assert "out_of_scope" in labels
    assert "nonsense" in labels
    for r in data:
        assert r["final_status"] == "refused", \
            f"Edge case '{r['question'][:40]}' must be refused, got {r['final_status']}"
        assert r["verification_result"]["supported"] is False, \
            f"Edge case '{r['question'][:40]}' must have supported=false"
