"""V6 grounding tests — deterministic, NO API calls, NO GPU.

FakeEmbedder maps each distinct token to a standard basis vector, so cosine
between two texts is exactly |shared| / (sqrt(|A|) * sqrt(|B|)).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fake embedder ──


class FakeEmbedder:
    """Token→basis-vector embedding with exact, controllable cosines."""

    _TOK_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

    def __init__(self):
        self._index: dict[str, int] = {}
        self._dim = 0

    def _tokenize(self, text: str) -> list[str]:
        toks = self._TOK_RE.findall(text)
        return list(dict.fromkeys(t.lower() for t in toks))

    def _vec(self, token: str) -> list[float]:
        if token not in self._index:
            self._index[token] = self._dim
            self._dim += 1
        v = [0.0] * self._dim
        v[self._index[token]] = 1.0
        return v

    def encode(self, text: str) -> list[float]:
        toks = self._tokenize(text)
        if not toks:
            return [0.0] * self._dim
        v: list[float] = []
        for t in toks:
            tv = self._vec(t)
            if len(tv) > len(v):
                v.extend([0.0] * (len(tv) - len(v)))
            for i, x in enumerate(tv):
                v[i] += x
        n = len(toks)
        v = [x / n for x in v]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


class FakeScorer:
    """Cross-encoder scorer stub: score = token-overlap ratio in [0,1]."""

    _tok = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(m.lower() for m in FakeScorer._tok.findall(text))

    def __call__(self, sentence: str, chunk_texts: list[str]) -> list[float]:
        st = self._tokens(sentence)
        if not st:
            return [1.0] * len(chunk_texts)
        return [len(st & self._tokens(c)) / len(st) for c in chunk_texts]


def _chunk(content: str, chunk_id: str = "c1", page: int = 24,
           source: str = "manual.pdf", rerank_score: float | None = None) -> dict:
    return {
        "chunk_id": chunk_id, "content": content, "source_file": source,
        "page_number": page, "content_type": "text",
        **({"rerank_score": rerank_score} if rerank_score is not None else {}),
    }


from src.workflow.grounding import (  # noqa: E402
    GroundingVerifier, _cos, _threshold_loop,
    parse_citation_markers, split_sentences, strip_citation_markers,
)


# ── Sentence splitting ──


class TestSplitSentences:
    def test_split_sentences_basic(self):
        assert split_sentences("甲。乙！丙？") == ["甲。", "乙！", "丙？"]

    def test_split_sentences_respects_brackets(self):
        out = split_sentences("第一句。括号（内部有句号。依然不分。）第二句。")
        assert len(out) == 2
        assert out[0] == "第一句。"
        assert "第二句。" in out[1]

    def test_split_sentences_citation_marker_not_split(self):
        out = split_sentences("该功能位于[来源: manual.pdf, 第24页]。B。")
        assert len(out) == 2
        assert "[来源: manual.pdf, 第24页]" in out[0]

    def test_split_sentences_consecutive_terminators(self):
        out = split_sentences("啊！！真的？")
        assert len(out) == 2
        assert all(s and not all(c in "。！？!?；;" for c in s) for s in out)

    def test_split_sentences_empty(self):
        assert split_sentences("") == []
        assert split_sentences("  ") == []

    def test_split_sentences_code_block_atomic(self):
        out = split_sentences("先说明。\n```\n代码示例。\n```\n结束语。")
        assert out == ["先说明。", "代码示例。", "结束语。"]


# ── Citation marker helpers ──


class TestCitationMarkers:
    def test_strip_markers(self):
        assert strip_citation_markers("需要充电[来源: manual.pdf, 第24页]。") == "需要充电。"

    def test_strip_markers_fullwidth(self):
        assert strip_citation_markers("需要充电[来源：说明书.pdf，第5页]。") == "需要充电。"

    def test_parse_citation_markers(self):
        out = parse_citation_markers("该功能[来源: 说明书.pdf, 第5页]。")
        assert len(out) == 1
        assert out[0]["claimed_source"] == "说明书.pdf"
        assert out[0]["claimed_page"] == 5

    def test_parse_citation_markers_none(self):
        assert parse_citation_markers("没有任何引用标记。") == []


# ── Threshold ladder ──


class TestThresholdLoop:
    def test_high_similarity_passes_at_top_rung(self):
        supported, eff = _threshold_loop([0.8], 0.55, 0.35, 0.9)
        assert supported == [True]
        assert eff[0] == 0.55

    def test_mid_similarity_rescued_by_descent(self):
        supported, eff = _threshold_loop([0.5], 0.55, 0.35, 0.9)
        assert supported == [True]
        assert eff[0] == pytest.approx(0.495)

    def test_below_floor_never_supported(self):
        supported, _ = _threshold_loop([0.1], 0.55, 0.35, 0.9)
        assert supported == [False]


# ── GroundingVerifier ──


class TestGroundingVerifier:
    def _verifier(self, **kw):
        return GroundingVerifier(FakeEmbedder(), **kw)

    def test_verifier_dict_shape(self):
        v = self._verifier()
        r = v.verify("q", "电源充电完成。", [_chunk("电源充电完成")])
        for key in ("supported", "confidence", "unsupported_claims",
                    "evidence_chunk_ids", "reason"):
            assert key in r

    def test_supported_sentence_high_similarity(self):
        v = self._verifier()
        r = v.verify("q", "电源充电完成。", [_chunk("电源充电完成", chunk_id="c9")])
        assert r["supported"] is True
        assert r["evidence_chunk_ids"] == ["c9"]

    def test_unsupported_sentence_disjoint(self):
        v = self._verifier()
        r = v.verify("q", "甲乙丙丁戊。", [_chunk("电源充电完成")])
        assert r["supported"] is False
        assert r["unsupported_claims"] == ["甲乙丙丁戊。"]

    def test_descending_threshold_rescues_paraphrase(self):
        v = self._verifier()
        # 16-token sentence vs 4-token chunk → cos = sqrt(4/16) = 0.5
        r = v.verify("q", "甲乙丙丁戊己庚辛壬癸子丑寅卯辰。",
                     [_chunk("甲乙丙丁")])
        assert r["supported"] is True
        se = r["sentence_evidence"][0]
        assert se["supported"] is True
        assert se["effective_threshold"] < 0.55

    def test_low_support_ratio_refused(self):
        v = self._verifier()
        r = v.verify("q", "电源充电完成。子丑寅卯辰。巳午未申酉。",
                     [_chunk("电源充电完成")])
        assert r["supported"] is False
        assert len(r["unsupported_claims"]) == 2

    def test_short_sentences_skipped(self):
        v = self._verifier()
        r = v.verify("q", "电源充电完成。是的。", [_chunk("电源充电完成")])
        assert r["supported"] is True
        assert r["grounding_meta"]["n_skipped_short"] == 1

    def test_empty_chunks_refused(self):
        v = self._verifier()
        r = v.verify("q", "电源充电完成。", [])
        assert r["supported"] is False
        assert "no chunks" in r["reason"]

    def test_empty_answer_refused(self):
        v = self._verifier()
        r = v.verify("q", "", [_chunk("电源充电完成")])
        assert r["supported"] is False

    def test_refusal_phrase_refused(self):
        v = self._verifier()
        r = v.verify("q", "根据现有说明书内容无法回答此问题。",
                     [_chunk("电源充电完成")])
        assert r["supported"] is False
        assert r["unsupported_claims"] == ["generator self-refused"]

    def test_evidence_dedup(self):
        v = self._verifier()
        r = v.verify("q", "电源充电完成。电源充电完成。",
                     [_chunk("电源充电完成", chunk_id="c1")])
        assert r["evidence_chunk_ids"] == ["c1"]

    def test_deterministic(self):
        v = self._verifier()
        chunks = [_chunk("电源充电完成")]
        assert v.verify("q", "电源充电完成。", chunks) == \
            v.verify("q", "电源充电完成。", chunks)

    def test_citation_audit(self):
        v = self._verifier(audit_citations=True)
        chunk = _chunk("电源充电完成", chunk_id="c1", page=24,
                       source="D:\\docs\\manual.pdf")
        r = v.verify(
            "q",
            "电源充电完成[来源: manual.pdf, 第24页]。"
            "子丑寅卯辰[来源: manual.pdf, 第99页]。"
            "戊己庚辛壬[来源: manual.pdf, 第24页]。",
            [chunk],
        )
        audit = r["citation_audit"]
        statuses = [a["status"] for a in audit]
        assert statuses == ["confirmed", "unmatched", "unconfirmed"]


# ── Cross-encoder scorer path ──


class TestCrossEncoderScorer:
    def test_requires_scorer_or_embedder(self):
        with pytest.raises(ValueError):
            GroundingVerifier()

    def test_scorer_supported(self):
        v = GroundingVerifier(scorer=FakeScorer(), scorer_floor=0.1)
        r = v.verify("q", "电源充电完成。", [_chunk("电源充电完成", chunk_id="c1")])
        assert r["supported"] is True
        assert r["grounding_meta"]["scorer"] == "cross_encoder"
        assert r["evidence_chunk_ids"] == ["c1"]

    def test_scorer_rejects_fabrication(self):
        # topical-but-fabricated: disjoint tokens → score 0 → below floor
        v = GroundingVerifier(scorer=FakeScorer(), scorer_floor=0.1)
        r = v.verify("q", "本产品由核聚变反应堆提供动力。",
                     [_chunk("电源充电完成")])
        assert r["supported"] is False
        assert len(r["unsupported_claims"]) == 1
        assert r["sentence_evidence"][0]["supported"] is False

    def test_scorer_floor_boundary(self):
        v = GroundingVerifier(scorer=FakeScorer(), scorer_floor=0.1)
        hi = v.verify("q", "电源充电完成。", [_chunk("电源充电完成")])
        assert hi["sentence_evidence"][0]["best_similarity"] == 1.0
        lo = v.verify("q", "甲乙丙丁戊。", [_chunk("电源充电完成")])
        assert lo["sentence_evidence"][0]["best_similarity"] == 0.0
        assert lo["supported"] is False

    def test_scorer_low_support_ratio_refused(self):
        v = GroundingVerifier(scorer=FakeScorer(), scorer_floor=0.1)
        r = v.verify("q", "电源充电完成。甲乙丙丁戊。",
                     [_chunk("电源充电完成")])
        # 1/2 supported → ratio 0.5 < 0.7 → refused
        assert r["supported"] is False


# ── Integration with VerifiedQA (mock retriever/generator, no API) ──


class TestGroundingInWorkflow:
    def _make_retriever(self, chunks):
        m = MagicMock()
        m.search.return_value = chunks
        return m

    def test_supported_answer_answered(self):
        from src.workflow.verified_qa import VerifiedQA
        chunks = [_chunk("电源充电完成", page=24, rerank_score=0.5)]
        chunks[0]["rerank_score"] = 0.5
        retriever = self._make_retriever(chunks)

        def gen(question, cs):
            return {"answer": "电源充电完成。"}

        vqa = VerifiedQA(retriever, gen, GroundingVerifier(FakeEmbedder()))
        state = vqa.run("无法开机怎么办")
        assert state["final_status"] == "answered"
        assert state["retry_count"] == 0

    def test_unsupported_answer_refused_after_retry(self):
        from src.workflow.verified_qa import VerifiedQA
        chunks = [_chunk("电源充电完成", page=24)]
        chunks[0]["rerank_score"] = 0.5
        retriever = self._make_retriever(chunks)

        def gen(question, cs):
            return {"answer": "电源充电完成。子丑寅卯辰。巳午未申酉。"}

        vqa = VerifiedQA(retriever, gen, GroundingVerifier(FakeEmbedder()),
                         max_retries=1)
        state = vqa.run("无法开机怎么办")
        assert state["final_status"] == "refused"
        assert state["retry_count"] == 1
