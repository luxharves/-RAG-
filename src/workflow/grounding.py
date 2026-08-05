"""V6 — Deterministic sentence-level grounding verification.

Replaces the LLM-as-judge verifier with a deterministic, reproducible check:

    answer → split sentences → embed (BGE-M3) → cosine vs retrieved chunks
    → per-sentence support via RAGFlow-style descending threshold
    → support_ratio → supported / refuse

The returned dict keeps the exact 5-key contract expected by
``src.workflow.verified_qa.VerifiedQA`` so the LangGraph flow is unchanged.
Extra keys (``sentence_evidence``, ``grounding_meta``, ``citation_audit``) are
ignored by VerifiedQA and consumed by the eval script / API.

Mechanism mirrors RAGFlow ``internal/service/citation.go``: citations are
COMPUTED (embedding similarity), not claimed by the LLM.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ── Sentence splitting ──

_SENT_ENDINGS = set("。！？!?；;")
_BRACKET_OPEN = set("（(「『[【{“\"'`")
_BRACKET_CLOSE = set("）)」』]】}\"'`")
_BRACKET_MAP = {
    "（": "）", "(": ")", "「": "」", "『": "』", "[": "]", "【": "】",
    "{": "}", "“": "”", '"': '"', "'": "'", "`": "`",
}

_MARKER_RE = re.compile(r"\[来源[:：]\s*([^\[\]]*?)\]")
_CIT_RE = re.compile(r"\[来源[:：]\s*(.+?)\s*[,，]\s*第\s*(\d+)\s*页\]")


def _fences(text: str) -> int:
    """Number of fenced code-block markers at position 0 (0, 3, or 6)."""
    n = 0
    for ch in text[:6]:
        if ch != "`":
            break
        n += 1
    return n if n == 3 else 0


def split_sentences(text: str) -> list[str]:
    """Split Chinese/English text into sentences.

    A boundary fires only on ``。！？!?；;`` at bracket depth 0, so citation
    markers ``[来源: …, 第N页]``, quoted/paired punctuation and parentheses are
    never split. Fenced code blocks are hoisted as one atomic sentence.
    Consecutive terminators collapse into one boundary.
    """
    if not text:
        return []

    sentences: list[str] = []
    buf: list[str] = []
    stack: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Fenced code block → one atomic sentence
        if ch == "`" and not stack:
            markers = _fences(text[i:])
            if markers:
                if buf:
                    sentences.append("".join(buf).strip())
                    buf = []
                end = text.find("```", i + 3)
                if end == -1:
                    sentences.append(text[i + 3:].strip())
                    break
                sentences.append(text[i + 3:end].strip())
                i = end + 3
                continue

        if ch in _BRACKET_OPEN:
            stack.append(_BRACKET_MAP.get(ch, ch))
            buf.append(ch)
        elif ch in _BRACKET_CLOSE:
            if stack and stack[-1] == ch:
                stack.pop()
            buf.append(ch)
        elif ch in _SENT_ENDINGS and not stack:
            buf.append(ch)
            sentences.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1

    if buf and "".join(buf).strip():
        sentences.append("".join(buf).strip())
    # Drop empty fragments and pure-punctuation fragments produced by
    # consecutive terminators (e.g. the second "！" in "啊！！").
    return [
        s for s in sentences
        if s and not all(ch in _SENT_ENDINGS for ch in s)
    ]


# ── Citation marker helpers ──


def strip_citation_markers(sentence: str) -> str:
    """Remove ``[来源: …]`` markers before encoding (they dilute cosine)."""
    return _MARKER_RE.sub("", sentence).strip()


def parse_citation_markers(sentence: str) -> list[dict]:
    """Extract ``[来源: <source>, 第<page>页]`` markers as dicts."""
    out: list[dict] = []
    for m in _CIT_RE.finditer(sentence):
        out.append({
            "marker": m.group(0),
            "claimed_source": m.group(1).strip(),
            "claimed_page": int(m.group(2)),
        })
    return out


def _normalize_source(source: str) -> str:
    """Lenient filename matching: basename + lowercase."""
    import os
    return os.path.basename(source).strip().lower()


# ── Similarity / threshold ladder ──


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine between two vectors (embeddings are normalized → dot product).

    Pads the shorter vector with zeros so it tolerates embeddings built by
    fake embedders whose latent dimension grows on demand.
    """
    n = max(len(a), len(b))
    total = 0.0
    for i in range(n):
        x = a[i] if i < len(a) else 0.0
        y = b[i] if i < len(b) else 0.0
        total += x * y
    return total


def _threshold_loop(
    maxsims: list[float],
    initial_threshold: float,
    floor: float,
    decay: float,
) -> tuple[list[bool], list[Optional[float]]]:
    """RAGFlow-style descending threshold ladder.

    A sentence is supported if its best cosine reaches the current rung.
    The rung starts at ``initial_threshold`` and decays by ``decay`` until
    ``floor``. Returns ``(supported, effective_threshold)``; the effective
    threshold records the highest rung a sentence cleared (paraphrase knob).
    """
    n = len(maxsims)
    supported = [False] * n
    eff: list[Optional[float]] = [None] * n
    remaining = set(range(n))
    threshold = initial_threshold

    while threshold >= floor and remaining:
        for i in list(remaining):
            if maxsims[i] >= threshold:
                supported[i] = True
                eff[i] = round(threshold, 4)
                remaining.discard(i)
        threshold *= decay

    return supported, eff


# ── Cross-encoder scorer (BGE-Reranker) ──


class CrossEncoderScorer:
    """Cross-encoder grounding scorer.

    Scores ``(sentence, chunk)`` jointly with a cross-encoder (BGE-Reranker),
    which is far more discriminative than bi-encoder cosine for detecting
    topical-but-fabricated claims ("Frankenstein hallucination").

    Interface: ``scorer(sentence, chunk_texts) -> list[float]`` aligned with
    ``chunk_texts``. Wraps any object with ``score(query, documents)``.
    """

    def __init__(self, reranker):
        self._reranker = reranker

    def __call__(self, sentence: str, chunk_texts: list[str]) -> list[float]:
        if not chunk_texts:
            return []
        scores = self._reranker.score(sentence, chunk_texts)
        # BGE-Reranker-v2-m3 already emits [0,1] sigmoid scores; clamp defensively
        return [max(0.0, min(1.0, float(s))) for s in scores]


# ── Verifier ──


class GroundingVerifier:
    """Deterministic sentence-level grounding verifier.

    The embedder is injected (constructor) so tests can pass a fake; it only
    needs ``encode(text) -> list[float]`` (normalized vector).
    """

    def __init__(
        self,
        embedder=None,
        *,
        scorer=None,
        initial_threshold: float = 0.55,
        threshold_floor: float = 0.35,
        decay: float = 0.9,
        min_support_ratio: float = 0.7,
        min_sentence_len: int = 5,
        refusal_phrases: tuple[str, ...] = (
            "根据现有说明书内容无法回答",
            "无法回答此问题",
        ),
        audit_citations: bool = False,
        scorer_floor: float = 0.1,
    ):
        if scorer is None and embedder is None:
            raise ValueError("GroundingVerifier requires 'scorer' or 'embedder'")
        self.embedder = embedder
        self.scorer = scorer
        self.scorer_floor = scorer_floor
        self.initial_threshold = initial_threshold
        self.threshold_floor = threshold_floor
        self.decay = decay
        self.min_support_ratio = min_support_ratio
        self.min_sentence_len = min_sentence_len
        self.refusal_phrases = refusal_phrases
        self.audit_citations = audit_citations

    @property
    def _mode(self) -> str:
        return "cross_encoder" if self.scorer is not None else "cosine"

    def __call__(self, question: str, answer: str, chunks: list[dict]) -> dict:
        return self.verify(question, answer, chunks)

    # ── Public ──

    def verify(self, question: str, answer: str, chunks: list[dict]) -> dict:
        result = self._verify_inner(question, answer, chunks)
        return result

    # ── Internal ──

    def _refuse(self, reason: str, unsupported: list[str] | None = None) -> dict:
        return {
            "supported": False,
            "confidence": 0.0,
            "unsupported_claims": unsupported or [],
            "evidence_chunk_ids": [],
            "reason": reason,
        }

    def _verify_inner(self, question: str, answer: str, chunks: list[dict]) -> dict:
        # ── Guards ──
        if not chunks:
            return self._refuse("no chunks retrieved")
        answer = (answer or "").strip()
        if not answer:
            return self._refuse("empty answer")
        if any(p in answer for p in self.refusal_phrases):
            return self._refuse(
                "refusal phrase detected", ["generator self-refused"]
            )

        valid_chunks = [c for c in chunks if str(c.get("content", "")).strip()]
        if not valid_chunks:
            return self._refuse("all chunks empty")
        chunk_texts = [c["content"] for c in valid_chunks]

        # ── Encode chunks (cosine path only) ──
        chunk_vecs = None
        if self.scorer is None:
            try:
                chunk_vecs = [self.embedder.encode(c) for c in chunk_texts]
            except Exception:
                return self._refuse("embedding failed")

        # ── Per-sentence similarity ──
        raw_sentences = split_sentences(answer)
        sentence_evidence: list[dict] = []
        for sent in raw_sentences:
            clean = strip_citation_markers(sent)
            short = len(clean) < self.min_sentence_len
            entry: dict[str, Any] = {
                "sentence": sent,
                "clean": clean,
                "supported": False,
                "best_similarity": None,
                "effective_threshold": None,
                "best_chunk_id": None,
                "status": "skipped_short" if short else "unsupported",
            }
            if not short:
                try:
                    if self.scorer is not None:
                        sims = self.scorer(clean, chunk_texts)
                        if not sims:
                            raise ValueError("scorer returned empty")
                    else:
                        sent_vec = self.embedder.encode(clean)
                        sims = [_cos(sent_vec, cv) for cv in chunk_vecs]
                except Exception:
                    sentence_evidence.append(entry)
                    continue
                best_i = int(max(range(len(sims)), key=lambda j: sims[j]))
                entry["best_similarity"] = round(float(sims[best_i]), 4)
                entry["best_chunk_id"] = valid_chunks[best_i].get("chunk_id", "")
            sentence_evidence.append(entry)

        # Map claim indices → evidence entries
        claim_entries = [e for e in sentence_evidence if e["status"] == "unsupported"]

        if not claim_entries:
            # Nothing substantive to ground (all short) → nothing to refuse
            return {
                "supported": True,
                "confidence": 1.0,
                "unsupported_claims": [],
                "evidence_chunk_ids": [],
                "reason": "no substantive claims to ground",
                "sentence_evidence": sentence_evidence,
                "grounding_meta": self._meta(0, 0, 0, 0),
                "citation_audit": self._audit(sentence_evidence, valid_chunks)
                if self.audit_citations else None,
            }

        # ── Support decision ──
        if self.scorer is not None:
            # Cross-encoder path: simple absolute floor (scores already [0,1])
            floor = self.scorer_floor
            for entry in claim_entries:
                sup = (entry["best_similarity"] or 0.0) >= floor
                entry["supported"] = sup
                entry["effective_threshold"] = floor
                entry["status"] = "supported" if sup else "unsupported"
        else:
            maxsims = [e["best_similarity"] or 0.0 for e in claim_entries]
            supported_flags, eff = _threshold_loop(
                maxsims, self.initial_threshold, self.threshold_floor, self.decay
            )
            for entry, flag, eff_t in zip(claim_entries, supported_flags, eff):
                entry["supported"] = flag
                entry["effective_threshold"] = eff_t
                entry["status"] = "supported" if flag else "unsupported"

        n_claims = len(claim_entries)
        n_supported = sum(1 for e in claim_entries if e["supported"])
        support_ratio = n_supported / n_claims if n_claims else 0.0

        # ── Evidence chunk ids ──
        evidence: list[str] = []
        seen: set[str] = set()
        for e in claim_entries:
            if e["supported"] and e["best_chunk_id"]:
                cid = e["best_chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    evidence.append(cid)

        # ── Aggregate ──
        supported = support_ratio >= self.min_support_ratio
        unsupported_claims = [
            e["clean"][:60] for e in claim_entries if not e["supported"]
        ]
        reason = (
            f"grounding: {n_supported}/{n_claims} sentences supported "
            f"(ratio {support_ratio:.2f} >= {self.min_support_ratio})"
            if supported else
            f"grounding: {n_supported}/{n_claims} sentences supported "
            f"(ratio {support_ratio:.2f} < {self.min_support_ratio})"
        )

        return {
            "supported": supported,
            "confidence": round(support_ratio, 2),
            "unsupported_claims": unsupported_claims,
            "evidence_chunk_ids": evidence,
            "reason": reason,
            "sentence_evidence": sentence_evidence,
            "grounding_meta": self._meta(
                n_claims, n_supported, support_ratio,
                sum(1 for e in sentence_evidence if e["status"] == "skipped_short"),
            ),
            "citation_audit": self._audit(sentence_evidence, valid_chunks)
            if self.audit_citations else None,
        }

    def _meta(self, n_claims: int, n_supported: int, ratio: float,
              n_skipped: int) -> dict:
        return {
            "scorer": self._mode,
            "scorer_floor": self.scorer_floor,
            "initial_threshold": self.initial_threshold,
            "threshold_floor": self.threshold_floor,
            "decay": self.decay,
            "min_support_ratio": self.min_support_ratio,
            "support_ratio": round(ratio, 4),
            "n_sentences": n_claims,
            "n_supported": n_supported,
            "n_skipped_short": n_skipped,
        }

    # ── Citation audit ──

    def _audit(self, sentence_evidence: list[dict],
               chunks: list[dict]) -> list[dict]:
        """Check every LLM-claimed [来源: X, 第Y页] against computed grounding."""
        audit: list[dict] = []
        for entry in sentence_evidence:
            for marker in parse_citation_markers(entry["sentence"]):
                # Resolve claimed source+page to a retrieved chunk
                matched = None
                for c in chunks:
                    if (_normalize_source(marker["claimed_source"])
                            == _normalize_source(c.get("source_file", ""))
                            and marker["claimed_page"] == int(c.get("page_number", -1))):
                        matched = c
                        break
                if matched is None:
                    status = "unmatched"  # hallucinated citation (no such chunk)
                elif entry["supported"] and entry["best_chunk_id"] == matched.get("chunk_id"):
                    status = "confirmed"
                else:
                    status = "unconfirmed"  # chunk exists but didn't ground this sentence
                audit.append({
                    "marker": marker["marker"],
                    "claimed_source": marker["claimed_source"],
                    "claimed_page": marker["claimed_page"],
                    "matched_chunk_id": matched.get("chunk_id") if matched else None,
                    "matched": matched is not None,
                    "grounded": status == "confirmed",
                    "status": status,
                })
        return audit
