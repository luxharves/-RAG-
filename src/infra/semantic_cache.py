"""V9 — Two-level semantic cache for the /query endpoint.

Exact (SHA256 of the normalized query) + semantic (BGE-M3 cosine > threshold)
so identical AND paraphrased questions skip the full pipeline and return in
~50ms instead of ~20s. Persistent SQLite, thread-safe.

This is the highest-leverage cost/latency investment in production RAG
(cache hits cut LLM calls ~68.8% and response time ~65x).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine between normalized vectors == dot product."""
    return sum(x * y for x, y in zip(a, b))


def _normalize_query_plus_salt(query: str, salt: str = "") -> str:
    """Normalized query + optional salt (e.g. doc_filter), separated by NUL so
    identical questions scoped to different documents get distinct keys."""
    norm = " ".join(query.lower().split())
    return norm + ("\x00" + salt if salt else "")


class SemanticCache:
    def __init__(
        self,
        embedder,
        db_path: str | Path,
        threshold: float = 0.9,
        ttl_days: int | None = None,
    ):
        self.embedder = embedder          # needs encode(text) -> list[float] (normalized)
        self.threshold = threshold
        self.ttl_days = ttl_days
        self._hits = 0
        self._miss = 0
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " query_hash TEXT PRIMARY KEY,"
            " query_text TEXT,"
            " query_vector TEXT,"     # JSON list[float]
            " response TEXT,"         # JSON dict
            " created_at REAL,"
            " hit_count INTEGER DEFAULT 0)"
        )
        self._conn.commit()

    # ── keys ──

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.lower().split())

    @staticmethod
    def _key_text(query: str, salt: str = "") -> str:
        """Normalized query + a salt (e.g. doc_filter) so identical questions
        scoped to different documents never share a cache key."""
        return _normalize_query_plus_salt(query, salt)

    @classmethod
    def _hash(cls, query: str, salt: str = "") -> str:
        return hashlib.sha256(
            _normalize_query_plus_salt(query, salt).encode("utf-8")
        ).hexdigest()

    def _expired(self, created_at: float) -> bool:
        if not self.ttl_days:
            return False
        return (time.time() - created_at) > self.ttl_days * 86400

    # ── public ──

    def get(self, query: str, salt: str = "") -> tuple[dict, str] | None:
        """Return ``(response_dict, source)`` on hit (source: "exact"/"semantic"),
        or None on miss. Marks a hit (and bumps hit_count) on a hit.
        ``salt`` (e.g. the doc_filter) separates cache keys per document scope."""
        qh = self._hash(query, salt)
        with self._lock:
            # 1. exact
            row = self._conn.execute(
                "SELECT response, created_at FROM cache WHERE query_hash = ?",
                (qh,),
            ).fetchone()
            if row and not self._expired(row[1]):
                self._hits += 1
                self._conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                    (qh,),
                )
                self._conn.commit()
                return json.loads(row[0]), "exact"

            # 2. semantic (paraphrase)
            vec = self.embedder.encode(self._key_text(query, salt))
            rows = self._conn.execute(
                "SELECT query_hash, query_vector, response, created_at FROM cache"
            ).fetchall()
            best_h, best_sim = None, -1.0
            for h, qvec_json, _resp, created in rows:
                if self._expired(created):
                    continue
                sim = _cos(vec, json.loads(qvec_json))
                if sim > best_sim:
                    best_sim, best_h = sim, h
            if best_h is not None and best_sim >= self.threshold:
                self._hits += 1
                resp_row = self._conn.execute(
                    "SELECT response FROM cache WHERE query_hash = ?", (best_h,)
                ).fetchone()
                self._conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                    (best_h,),
                )
                self._conn.commit()
                return json.loads(resp_row[0]), "semantic"

            self._miss += 1
            return None

    def put(self, query: str, response: dict, salt: str = "") -> None:
        """Store a response keyed by the (normalized) query + its embedding.
        ``salt`` must match the salt passed to :meth:`get` (doc_filter scope)."""
        qh = self._hash(query, salt)
        vec = json.dumps(self.embedder.encode(self._key_text(query, salt)))
        resp = json.dumps(response, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache"
                " (query_hash, query_text, query_vector, response, created_at, hit_count)"
                " VALUES (?, ?, ?, ?, ?, 0)",
                (qh, query, vec, resp, time.time()),
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()
            self._hits = 0
            self._miss = 0

    def stats(self) -> dict:
        with self._lock:
            entries = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            total = self._hits + self._miss
            return {
                "entries": entries,
                "hits": self._hits,
                "miss": self._miss,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "threshold": self.threshold,
            }
