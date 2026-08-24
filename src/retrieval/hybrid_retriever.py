"""Hybrid retriever: Dense + BM25 with RRF fusion.

Supports three modes:
- dense: BGE-M3 only
- bm25: BM25 only
- hybrid: RRF fusion of both

Fallback behaviour:
- hybrid with BM25 unavailable → degrades to dense (with degrade_reason)
- hybrid with Dense unavailable → degrades to bm25 (with degrade_reason)
- both unavailable → raises HybridRetrievalError
"""

from __future__ import annotations

from src.retrieval.retriever import DenseRetriever
from src.retrieval.bm25 import BM25Retriever


class HybridRetrievalError(Exception):
    """Raised when both Dense and BM25 are unavailable."""


def _rrf_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """Reciprocal Rank Fusion combining dense and BM25 results.

    RRF score = 1/(k + rank_dense) + 1/(k + rank_bm25)
    """
    combined: dict[str, dict] = {}

    for rank, r in enumerate(dense_results, 1):
        cid = r.get("chunk_id", r.get("content", ""))
        if cid not in combined:
            combined[cid] = {
                **r,
                "dense_rank": rank,
                "dense_score": r.get("retrieval_score", 0),
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
                "fusion_score": 0.0,
                "retrieval_channel": "hybrid",
            }
        combined[cid]["rrf_score"] += 1.0 / (k + rank)

    for rank, r in enumerate(bm25_results, 1):
        cid = r.get("chunk_id", r.get("content", ""))
        if cid not in combined:
            combined[cid] = {
                **r,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": rank,
                "bm25_score": r.get("bm25_score", 0),
                "rrf_score": 0.0,
                "fusion_score": 0.0,
                "retrieval_channel": "hybrid",
                "retrieval_score": None,
            }
        else:
            combined[cid]["bm25_rank"] = rank
            combined[cid]["bm25_score"] = r.get("bm25_score", 0)
        combined[cid]["rrf_score"] += 1.0 / (k + rank)

    sorted_results = sorted(
        combined.values(), key=lambda x: x["rrf_score"], reverse=True
    )
    for r in sorted_results:
        val = round(r["rrf_score"], 6)
        r["rrf_score"] = val
        r["fusion_score"] = val
    return sorted_results[:top_k]


def _mark_degrade(results: list[dict], channel: str, reason: str) -> list[dict]:
    """Annotate results with degrade metadata."""
    for r in results:
        r["retrieval_channel"] = channel
        r["degrade_reason"] = reason
        r["rrf_score"] = None
        r["fusion_score"] = None
        if channel == "dense":
            r["dense_rank"] = r.get("dense_rank") or (results.index(r) + 1 if "dense_rank" in r else None)
    return results


class HybridRetriever:
    """Retriever supporting dense, bm25, and hybrid (RRF) modes."""

    def __init__(
        self,
        collection_name: str = "v1_multimodal_kw",
        bm25_index_path: str | None = None,
        embedder=None,
    ):
        self.collection_name = collection_name
        self._dense: DenseRetriever | None = None
        self._bm25: BM25Retriever | None = None
        self._bm25_path = bm25_index_path
        self._embedder = embedder  # shared BGE-M3 (see DenseRetriever)

    def _ensure_dense(self) -> DenseRetriever:
        if self._dense is None:
            self._dense = DenseRetriever(
                collection_name=self.collection_name,
                embedder=self._embedder,
            )
        return self._dense

    def _ensure_bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever()
            if self._bm25_path:
                self._bm25.load(self._bm25_path)
        return self._bm25

    def search(
        self, query: str, top_k: int = 5, mode: str = "hybrid",
        doc_filter: str | None = None,
    ) -> list[dict]:
        """Search with specified mode: dense, bm25, hybrid.

        doc_filter: a source_file path to restrict results to one document
            (passed through to both dense and BM25 before fusion).

        Returns results with: retrieval_channel, retrieval_score,
        dense_rank, dense_score, bm25_rank, bm25_score,
        rrf_score, fusion_score, degrade_reason.
        """
        dense_ok = False
        bm25_ok = False

        # Dense
        try:
            dense_results = self._ensure_dense().search(query, top_k=20, doc_filter=doc_filter)
            dense_ok = True
        except Exception:
            dense_results = []
            dense_ok = False

        # BM25
        try:
            bm25_retriever = self._ensure_bm25()
            if bm25_retriever.is_loaded:
                bm25_results = bm25_retriever.search(query, top_k=20, doc_filter=doc_filter)
                bm25_ok = True
            else:
                bm25_results = []
                bm25_ok = False
        except Exception:
            bm25_results = []
            bm25_ok = False

        # ── mode=dense ──
        if mode == "dense":
            if not dense_ok:
                raise HybridRetrievalError(
                    "Dense retrieval unavailable — check Milvus connection"
                )
            for i, r in enumerate(dense_results[:top_k]):
                r["dense_rank"] = i + 1
                r["dense_score"] = r.get("retrieval_score", 0)
                r["bm25_rank"] = None
                r["bm25_score"] = None
                r["rrf_score"] = None
                r["fusion_score"] = None
                r["retrieval_channel"] = "dense"
            return dense_results[:top_k]

        # ── mode=bm25 ──
        if mode == "bm25":
            if not bm25_ok:
                raise HybridRetrievalError(
                    "BM25 unavailable — index not loaded"
                )
            return bm25_results[:top_k]

        # ── mode=hybrid ──
        if mode == "hybrid":
            # Both available
            if dense_ok and bm25_ok:
                return _rrf_fusion(dense_results, bm25_results, top_k=top_k)

            # Degrade: BM25 unavailable, fall back to Dense
            if dense_ok and not bm25_ok:
                for i, r in enumerate(dense_results[:top_k]):
                    r["dense_rank"] = i + 1
                    r["dense_score"] = r.get("retrieval_score", 0)
                return _mark_degrade(
                    dense_results[:top_k],
                    channel="dense",
                    reason="bm25_unavailable",
                )

            # Degrade: Dense unavailable, fall back to BM25
            if not dense_ok and bm25_ok:
                return _mark_degrade(
                    bm25_results[:top_k],
                    channel="bm25",
                    reason="dense_unavailable",
                )

            # Both unavailable
            raise HybridRetrievalError(
                "Hybrid retrieval failed: both Dense and BM25 are unavailable. "
                "Check Milvus connection and BM25 index."
            )

        return []

    def close(self) -> None:
        if self._dense is not None:
            self._dense.close()
            self._dense = None
