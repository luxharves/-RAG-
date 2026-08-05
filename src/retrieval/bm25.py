"""BM25 sparse retrieval with persistent index for Chinese text."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 retriever with jieba tokenization and persistent index."""

    def __init__(self, index_path: str | Path | None = None):
        self._index_path = Path(index_path) if index_path else None
        self._corpus: list[list[str]] = []
        self._metadata: list[dict] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunk dicts."""
        self._corpus = [self.tokenize(c["content"]) for c in chunks]
        self._metadata = [
            {k: c[k] for k in ("chunk_id", "page_number", "content_type",
                                "source_file", "content") if k in c}
            for c in chunks
        ]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 10,
               doc_filter: str | None = None) -> list[dict]:
        """Search and return results with metadata.

        doc_filter: a source_file path to restrict results to one document
            (filtered post-scoring, before truncation).
        """
        if self._bm25 is None:
            return []
        tokens = self.tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Rank all, then filter by doc before truncation (cheap: get_scores is vectorized)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(ranked, 1):
            if doc_filter is not None and \
                    self._metadata[idx].get("source_file", "") != doc_filter:
                continue
            meta = dict(self._metadata[idx])
            meta["bm25_score"] = round(float(score), 4)
            meta["bm25_rank"] = len(results) + 1
            meta["retrieval_channel"] = "bm25"
            results.append(meta)
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._index_path
        if target is None:
            raise ValueError("No save path specified")
        target.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": self._corpus,
            "metadata": self._metadata,
        }
        with open(target / "bm25_index.pkl", "wb") as f:
            pickle.dump(data, f)
        # Save metadata as JSON for inspection
        with open(target / "bm25_meta.json", "w", encoding="utf-8") as f:
            json.dump([{k: m[k][:100] if k == "content" else m[k]
                         for k in m} for m in self._metadata],
                      f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._index_path
        if target is None:
            raise ValueError("No load path specified")
        with open(target / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        self._corpus = data["corpus"]
        self._metadata = data["metadata"]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None

    def add_chunks(self, chunks: list[dict]) -> None:
        """Incrementally add chunks to the BM25 index."""
        for c in chunks:
            tokens = self.tokenize(c.get("content", ""))
            self._corpus.append(tokens)
            self._metadata.append({
                "chunk_id": c.get("chunk_id", ""),
                "page_number": c.get("page_number", 0),
                "content_type": c.get("content_type", "text"),
                "source_file": c.get("source_file", ""),
                "content": c.get("content", ""),
            })
        # Rebuild BM25 Okapi index (Okapi doesn't support incremental update)
        self._bm25 = BM25Okapi(self._corpus)

    def remove_by_source(self, source_file: str) -> int:
        """Remove all chunks belonging to source_file. Returns count removed."""
        if self._bm25 is None or not self._corpus:
            return 0
        count = 0
        new_corpus, new_meta = [], []
        for tokens, meta in zip(self._corpus, self._metadata):
            if meta.get("source_file", "") == source_file:
                count += 1
            else:
                new_corpus.append(tokens)
                new_meta.append(meta)
        self._corpus = new_corpus
        self._metadata = new_meta
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
        return count

    @property
    def num_docs(self) -> int:
        return len(self._corpus)
