#!/usr/bin/env python
"""Phase 0 Smoke Test — verifies all infrastructure components are operational.

Run: python scripts/smoke_test.py

Each test produces a structured result; no results are faked.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SmokeResult:
    component: str
    status: str  # PASS, FAIL, NOT_RUN
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        d = {"component": self.component, "status": self.status, "detail": self.detail}
        if self.metrics:
            d["metrics"] = self.metrics
        if self.error:
            d["error"] = self.error
        return d


def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"


# ──────────────────────────────────────────────
# Test implementations
# ──────────────────────────────────────────────


def test_config() -> SmokeResult:
    """Verify that configuration loads correctly."""
    t0 = time.perf_counter()
    try:
        from src.config.settings import settings

        checks = {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "model_device": settings.model_device,
            "milvus_uri": settings.milvus_uri,
            "project_root_exists": settings.project_root.exists(),
        }
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="Config",
            status="PASS",
            detail=f"Loaded settings from .env; device={settings.model_device}",
            metrics={"checks": checks, "duration": _format_duration(elapsed)},
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="Config",
            status="FAIL",
            detail="Failed to load configuration",
            metrics={"duration": _format_duration(elapsed)},
            error=traceback.format_exc(),
        )


def test_bge_m3() -> SmokeResult:
    """Load BGE-M3 and run a single-text embedding smoke test."""
    t0 = time.perf_counter()
    try:
        from src.infra.embedder import Embedder

        embedder = Embedder()
        embedder.load()

        text = "设备无法开机，请检查电源线是否连接正确。"
        vector = embedder.encode(text)

        if not isinstance(vector, list) or len(vector) == 0:
            raise ValueError(f"Invalid embedding vector: type={type(vector)}")

        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="BGE-M3",
            status="PASS",
            detail=f"Encoded text to {embedder.dim}-dim vector on {embedder.device}",
            metrics={
                "model": embedder.model_name,
                "device": embedder.device,
                "dim": embedder.dim,
                "vector_preview": vector[:5],
                "duration": _format_duration(elapsed),
            },
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="BGE-M3",
            status="FAIL",
            detail="Failed to load BGE-M3 or encode text",
            metrics={"duration": _format_duration(elapsed)},
            error=traceback.format_exc(),
        )


def test_bge_reranker() -> SmokeResult:
    """Load BGE-Reranker and score a query-document pair."""
    t0 = time.perf_counter()
    try:
        from src.infra.reranker import Reranker

        reranker = Reranker()
        reranker.load()

        query = "设备无法开机怎么办？"
        documents = [
            "请检查电源线是否连接正确，确保电源开关已打开。",
            "设备屏幕亮度可以在设置中调节。",
            "定期清洁设备表面可以延长使用寿命。",
        ]
        scores = reranker.score(query, documents)

        if len(scores) != len(documents):
            raise ValueError(f"Score count mismatch: {len(scores)} vs {len(documents)}")

        # Highest score should be the relevant doc (index 0)
        max_idx = scores.index(max(scores))
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="BGE-Reranker",
            status="PASS",
            detail=f"Scored {len(documents)} documents; best match at index {max_idx} (score={scores[max_idx]:.4f})",
            metrics={
                "model": reranker.model_name,
                "device": reranker.device,
                "scores": [round(s, 4) for s in scores],
                "top_index": max_idx,
                "duration": _format_duration(elapsed),
            },
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="BGE-Reranker",
            status="FAIL",
            detail="Failed to load BGE-Reranker or score documents",
            metrics={"duration": _format_duration(elapsed)},
            error=traceback.format_exc(),
        )


def test_milvus() -> SmokeResult:
    """Create a temporary Milvus collection, insert a vector, and search it back."""
    t0 = time.perf_counter()
    collection_name = "smoke_test_phase0"
    try:
        from src.infra.milvus_client import MilvusAdapter

        milvus = MilvusAdapter()
        milvus.connect()

        # Create a collection with a known dimension
        dim = 128
        milvus.create_collection(collection_name, dim)

        # Insert a test vector
        test_vector = [0.1] * dim
        test_text = "smoke test entry"
        milvus.insert(collection_name, test_text, test_vector)

        # Search it back
        results = milvus.search(collection_name, test_vector, top_k=1)

        if len(results) == 0:
            raise ValueError("Search returned no results")

        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="Milvus",
            status="PASS",
            detail=f"Created collection, inserted 1 vector, search returned {len(results)} hit(s)",
            metrics={
                "uri": milvus.uri,
                "collection": collection_name,
                "dim": dim,
                "hit_distance": round(results[0]["distance"], 4),
                "hit_text": results[0]["text"],
                "duration": _format_duration(elapsed),
            },
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="Milvus",
            status="FAIL",
            detail="Failed to complete Milvus smoke test",
            metrics={"duration": _format_duration(elapsed)},
            error=traceback.format_exc(),
        )
    finally:
        # Clean up the temporary collection
        try:
            from src.infra.milvus_client import MilvusAdapter
            m = MilvusAdapter()
            m.connect()
            m.drop_collection(collection_name)
            m.disconnect()
        except Exception:
            pass


def test_llm() -> SmokeResult:
    """Test OpenAI-compatible LLM chat completion."""
    t0 = time.perf_counter()
    try:
        from src.infra.llm_client import LLMClient

        client = LLMClient()

        if not client.is_configured:
            elapsed = time.perf_counter() - t0
            return SmokeResult(
                component="LLM",
                status="NOT_RUN",
                detail="LLM endpoint not configured (placeholder values in .env). Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL.",
                metrics={"duration": _format_duration(elapsed)},
            )

        messages = [
            {"role": "user", "content": "请用一句话回答：向量检索的核心原理是什么？"},
        ]
        response_text, raw = client.chat(messages)

        if not response_text:
            raise ValueError("LLM returned empty response")

        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="LLM",
            status="PASS",
            detail=f"LLM responded: {response_text[:80]}...",
            metrics={
                "model": raw.get("model"),
                "usage": raw.get("usage"),
                "response_preview": response_text[:120],
                "duration": _format_duration(elapsed),
            },
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="LLM",
            status="FAIL",
            detail="LLM call failed — check API endpoint, key, and model",
            metrics={"duration": _format_duration(elapsed)},
            error=traceback.format_exc(),
        )


def test_vlm() -> SmokeResult:
    """Test OpenAI-compatible VLM image understanding.

    If no test image is available, returns NOT_RUN.
    """
    t0 = time.perf_counter()
    try:
        from src.infra.vlm_client import VLMClient

        client = VLMClient()

        if not client.is_configured:
            elapsed = time.perf_counter() - t0
            return SmokeResult(
                component="Qwen3-VL-32B",
                status="NOT_RUN",
                detail="VLM endpoint not configured (placeholder values in .env). Set VLM_BASE_URL, VLM_API_KEY, and VLM_MODEL.",
                metrics={"duration": _format_duration(elapsed)},
            )

        # Look for a test image in data/raw_docs or data/processed
        test_image = None
        candidates = [
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "data" / "raw_docs",
            PROJECT_ROOT / "data" / "processed",
        ]
        for folder in candidates:
            if folder.exists():
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
                    images = list(folder.glob(ext))
                    if images:
                        test_image = images[0]
                        break
            if test_image:
                break

        if test_image is None:
            elapsed = time.perf_counter() - t0
            return SmokeResult(
                component="Qwen3-VL-32B",
                status="NOT_RUN",
                detail="No test image found in data/ — VLM endpoint is configured but no image to test with. Place a test image in data/raw_docs/ or data/processed/ to enable VLM smoke test.",
                metrics={"duration": _format_duration(elapsed)},
            )

        prompt = "请用一句话描述这张图片的内容。"
        response_text, raw = client.chat_with_image(test_image, prompt)

        if not response_text:
            raise ValueError("VLM returned empty response")

        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="Qwen3-VL-32B",
            status="PASS",
            detail=f"VLM described image: {response_text[:80]}...",
            metrics={
                "model": raw.get("model"),
                "usage": raw.get("usage"),
                "image": str(test_image),
                "response_preview": response_text[:120],
                "duration": _format_duration(elapsed),
            },
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            component="Qwen3-VL-32B",
            status="FAIL",
            detail="VLM call failed — check API endpoint",
            metrics={"duration": _format_duration(elapsed)},
            error=traceback.format_exc(),
        )


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────


def main() -> None:
    """Run all Phase 0 smoke tests and print structured results."""
    print("=" * 72)
    print("Phase 0 Smoke Test Suite")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 72)

    tests = [
        ("Config", test_config),
        ("BGE-M3", test_bge_m3),
        ("BGE-Reranker", test_bge_reranker),
        ("Milvus", test_milvus),
        ("LLM", test_llm),
        ("Qwen3-VL-32B", test_vlm),
    ]

    results: list[SmokeResult] = []

    for name, test_fn in tests:
        print(f"\n{'─' * 60}")
        print(f"Running: {name}...")
        result = test_fn()
        results.append(result)
        print(f"  Status: {result.status}")
        print(f"  Detail: {result.detail}")
        if result.error:
            # Print only the last line of the traceback for brevity
            err_lines = result.error.strip().split("\n")
            print(f"  Error:  {err_lines[-1]}")
        if result.metrics:
            print(f"  Metrics: {json.dumps(result.metrics, ensure_ascii=False, default=str)}")

    # ── Summary ──
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)

    summary = {
        "phase": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [r.to_dict() for r in results],
    }

    # Overall status
    statuses = [r.status for r in results]
    if "FAIL" in statuses:
        overall = "CONDITIONAL_PASS" if "PASS" in statuses else "FAIL"
    elif all(s == "NOT_RUN" for s in statuses):
        overall = "FAIL"
    elif all(s in ("PASS", "NOT_RUN") for s in statuses):
        overall = "PASS" if any(s == "PASS" for s in statuses) else "FAIL"
    else:
        overall = "FAIL"
    summary["overall"] = overall

    print(f"\nOverall: {overall}")
    for r in results:
        icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "NOT_RUN": "[SKIP]"}[r.status]
        print(f"  {icon} {r.component}: {r.detail[:100]}")

    # Save to file
    output_path = PROJECT_ROOT / "storage" / "phase0_smoke_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 72}")
    print("Phase 0 smoke test complete.")
    print("=" * 72)

    # Exit with non-zero if any critical failure
    if overall == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
