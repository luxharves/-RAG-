"""V9 semantic cache tests — deterministic, NO API/GPU.

FakeEmbedder maps tokens to basis vectors so cosine = |shared| / (sqrt|A|*sqrt|B|).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infra.semantic_cache import SemanticCache  # noqa: E402


class FakeEmbedder:
    _TOK = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")

    def __init__(self):
        self._idx: dict[str, int] = {}
        self._dim = 0

    def _tokens(self, text: str) -> list[str]:
        return list(dict.fromkeys(m.lower() for m in self._TOK.findall(text)))

    def encode(self, text: str) -> list[float]:
        toks = self._tokens(text)
        if not toks:
            return [0.0] * self._dim
        v: list[float] = []
        for t in toks:
            if t not in self._idx:
                self._idx[t] = self._dim
                self._dim += 1
            i = self._idx[t]
            if i >= len(v):
                v.extend([0.0] * (i + 1 - len(v)))
            v[i] += 1.0
        n = len(toks)
        v = [x / n for x in v]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v


def _cache(tmp_path, **kw):
    kw.setdefault("threshold", 0.9)
    return SemanticCache(FakeEmbedder(), tmp_path / "cache.db", **kw)


class TestSemanticCache:
    def test_exact_hit(self, tmp_path):
        c = _cache(tmp_path)
        c.put("设备无法开机怎么办", {"answer": "A"})
        assert c.get("设备无法开机怎么办") == ({"answer": "A"}, "exact")
        assert c.stats()["hits"] == 1

    def test_semantic_hit_paraphrase(self, tmp_path):
        c = _cache(tmp_path)
        c.put("设备无法开机怎么办", {"answer": "A"})
        # paraphrase shares nearly all tokens -> cosine ~0.95 > 0.9
        assert c.get("设备无法开机了怎么办") == ({"answer": "A"}, "semantic")
        assert c.stats()["hits"] == 1

    def test_disjoint_miss(self, tmp_path):
        c = _cache(tmp_path)
        c.put("设备无法开机怎么办", {"answer": "A"})
        assert c.get("今天天气怎么样") is None
        assert c.stats()["miss"] == 1

    def test_threshold_respects_limit(self, tmp_path):
        c = _cache(tmp_path, threshold=0.98)  # paraphrase cosine ~0.95 < 0.98
        c.put("设备无法开机怎么办", {"answer": "A"})
        assert c.get("设备无法开机了怎么办") is None
        assert c.stats()["miss"] == 1

    def test_roundtrip(self, tmp_path):
        c = _cache(tmp_path)
        resp = {"answer": "x", "sources": [], "cache_hit": False}
        c.put("q", resp)
        assert c.get("q") == (resp, "exact")

    def test_persistence(self, tmp_path):
        db = tmp_path / "c.db"
        c = SemanticCache(FakeEmbedder(), db)
        c.put("q", {"answer": "A"})
        c._conn.close()
        c2 = SemanticCache(FakeEmbedder(), db)
        assert c2.get("q") == ({"answer": "A"}, "exact")

    def test_stats(self, tmp_path):
        c = _cache(tmp_path)
        c.put("q1", {"answer": "A"})
        c.get("q1")            # hit
        c.get("q2")            # miss
        s = c.stats()
        assert s["entries"] == 1
        assert s["hits"] == 1
        assert s["miss"] == 1
        assert s["hit_rate"] == 0.5

    def test_clear(self, tmp_path):
        c = _cache(tmp_path)
        c.put("q", {"answer": "A"})
        c.clear()
        assert c.get("q") is None
        assert c.stats()["entries"] == 0
