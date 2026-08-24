"""V3 Reranked Retriever: Hybrid (V2) candidate recall + BGE-Reranker-v2-m3 fine ranking.

Flow: Query → Hybrid top-20 → Reranker score → top-5 final
"""

from __future__ import annotations

from src.retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalError
from src.infra.reranker import Reranker


class RerankerUnavailableError(Exception):
    """Raised when Reranker is unavailable."""


class RerankedRetriever:
    """Two-stage retriever: Hybrid (stage 1) + Reranker (stage 2)."""

    def __init__(
        self,
        collection_name: str = "v1_multimodal_kw",
        bm25_index_path: str | None = None,
        candidate_top_k: int = 20,
        final_top_k: int = 5,
        reranker: Reranker | None = None,
        embedder=None,
    ):
        self.collection_name = collection_name
        self._hybrid = HybridRetriever(
            collection_name=collection_name,
            bm25_index_path=bm25_index_path,
            embedder=embedder,
        )
        # Injectable shared instance (e.g. deps.get_reranker) so the grounding
        # verifier can reuse the SAME cross-encoder instead of loading a 2nd.
        self._reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k

    def _ensure_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        if not self._reranker.is_loaded:
            self._reranker.load()
        return self._reranker

    def search(
        self, query: str, top_k: int | None = None, mode: str = "reranked",
        doc_filter: str | None = None,
    ) -> list[dict]:
        final_k = top_k or self.final_top_k

        if mode == "v2_hybrid":
            return self._hybrid.search(query, top_k=final_k, mode="hybrid",
                                       doc_filter=doc_filter)

        # ── Stage 1: Hybrid candidate recall ──
        try:
            candidates = self._hybrid.search(
                query, top_k=self.candidate_top_k, mode="hybrid",
                doc_filter=doc_filter,
            )
        except HybridRetrievalError:
            raise HybridRetrievalError(
                "Stage-1 hybrid recall failed — cannot proceed to rerank"
            )

        if len(candidates) == 0:
            return []

        # ── Stage 2: Reranker scoring ──
        reranker_ok = False
        try:
            reranker = self._ensure_reranker()
            # Build (query, doc) pairs
            docs = [c.get("content", "") for c in candidates]
            scores = reranker.score(query, docs)
            reranker_ok = True
        except Exception:
            scores = []

        if not reranker_ok or len(scores) != len(candidates):
            # Degrade to V2 Hybrid
            for i, c in enumerate(candidates[:final_k]):
                c["rerank_score"] = None
                c["rerank_rank"] = None
                c["ranking_changed"] = False
                c["degrade_reason"] = "reranker_unavailable"
            return candidates[:final_k]

        # Annotate with rerank scores and preserve original ranks
        for i, c in enumerate(candidates):
            c["rerank_score"] = round(float(scores[i]), 4)
            c["original_hybrid_rank"] = i + 1

        # Sort by rerank_score descending
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Annotate final ranks and detect changes
        for new_rank, c in enumerate(reranked, 1):
            old_rank = c.get("original_hybrid_rank", new_rank)
            c["rerank_rank"] = new_rank
            c["ranking_changed"] = (old_rank != new_rank)

        return reranked[:final_k]

    def close(self) -> None:
        self._hybrid.close()
        self._reranker = None
