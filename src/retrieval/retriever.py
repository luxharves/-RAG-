"""Dense retriever using BGE-M3 embeddings and Milvus."""

from __future__ import annotations

import os as _os
_os.environ.setdefault("MILVUS_URI", "http://localhost:19530")

from pathlib import Path as _Path
from dotenv import dotenv_values as _dotenv_values

# Read the real Milvus path from .env before the env-var override
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_ENV_VALS = _dotenv_values(str(_PROJECT_ROOT / ".env"))
_DEFAULT_MILVUS_URI = _ENV_VALS.get("MILVUS_URI", "milvus.db")

from pymilvus import MilvusClient  # noqa: E402
from src.infra.embedder import Embedder  # noqa: E402


def make_source_file_filter(source_file: str) -> str:
    """Build a Milvus filter expression matching one source_file (escaped).

    Mirrors the backslash-escaping precedent in IncrementalIndexer._delete_document.
    """
    escaped = source_file.replace("\\", "\\\\").replace('"', '\\"')
    return f'source_file == "{escaped}"'


class DenseRetriever:
    """Retrieves top-K chunks via dense vector similarity."""

    def __init__(
        self,
        collection_name: str = "v0_naive_rag",
        uri: str | None = None,
        embedder: Embedder | None = None,
    ):
        self.collection_name = collection_name
        self.uri = uri or _DEFAULT_MILVUS_URI
        self._client: MilvusClient | None = None
        # Injectable shared instance (e.g. deps.get_embedder) so the semantic
        # cache and the retriever reuse ONE BGE-M3 — a 2nd local load on an 8GB
        # GPU trips the 8GB budget and stalls the first search (mirrors the
        # shared-reranker pattern in RerankedRetriever).
        self._embedder: Embedder | None = embedder

    def _ensure_client(self) -> MilvusClient:
        if self._client is None:
            uri = self.uri
            if uri and not uri.startswith("http"):
                uri = str(_PROJECT_ROOT / uri)
            self._client = MilvusClient(uri)
        return self._client

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
            self._embedder.load()
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str | None = None,
    ) -> list[dict]:
        """Search for top-K chunks matching the query.

        Args:
            query: the query text.
            top_k: number of results.
            doc_filter: a source_file path to restrict results to one document
                (metadata filtering); None = search the whole collection.

        Returns list of {chunk_id, content, source_file, page_number,
                          content_type, retrieval_score, retrieval_channel}.
        """
        embedder = self._ensure_embedder()
        client = self._ensure_client()

        query_vector = embedder.encode(query)

        # Ensure collection is loaded
        client.load_collection(self.collection_name)

        filter_expr = make_source_file_filter(doc_filter) if doc_filter else None
        search_kwargs = dict(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=[
                "chunk_id", "content", "source_file", "page_number",
                "content_type", "document_id",
            ],
        )
        if filter_expr:
            search_kwargs["filter"] = filter_expr
        results = client.search(**search_kwargs)

        hits = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            hits.append({
                "chunk_id": entity.get("chunk_id", ""),
                "content": entity.get("content", ""),
                "source_file": entity.get("source_file", ""),
                "page_number": entity.get("page_number", 0),
                "content_type": entity.get("content_type", "text"),
                "retrieval_channel": "dense",
                "retrieval_score": round(hit.get("distance", 0.0), 4),
                "rerank_score": None,
            })

        return hits

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
