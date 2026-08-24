"""API dependencies — lazy singleton init, serial-safe Milvus Lite."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()


def _read_env() -> dict:
    from dotenv import dotenv_values
    return dotenv_values(str(PROJECT_ROOT / ".env"))


@lru_cache
def get_settings():
    from src.config.settings import Settings
    env = _read_env()
    return Settings(**{k: v for k, v in env.items() if v})


def _milvus_uri() -> str:
    env = _read_env()
    uri = env.get("MILVUS_URI", "milvus.db")
    if not uri.startswith("http"):
        uri = str(PROJECT_ROOT / uri)
    return uri


@lru_cache
def get_embedder():
    from src.infra.embedder import Embedder
    e = Embedder()
    e.load()
    return e


@lru_cache
def get_milvus_client():
    from pymilvus import MilvusClient
    with _lock:
        return MilvusClient(_milvus_uri())


@lru_cache
def get_latest_v1_collection() -> str:
    client = get_milvus_client()
    kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    return (ts + kw)[-1] if (ts or kw) else "v1_multimodal_kw_latest"


@lru_cache
def get_bm25():
    from src.retrieval.bm25 import BM25Retriever
    bm = BM25Retriever()
    p = PROJECT_ROOT / "storage" / "bm25"
    if (p / "bm25_index.pkl").exists():
        bm.load(p)
    else:
        bm.build([])
    return bm


@lru_cache
def get_reranker():
    from src.infra.reranker import Reranker
    r = Reranker()
    r.load()
    return r


@lru_cache
def get_retriever():
    from src.retrieval.reranked_retriever import RerankedRetriever
    return RerankedRetriever(
        collection_name=get_latest_v1_collection(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
        reranker=get_reranker(),
        # Share the ONE BGE-M3 with the semantic cache (avoids a 2nd GPU load
        # that stalls the first search on an 8GB card).
        embedder=get_embedder(),
    )


def _make_llm_verifier():
    """V4 LLM-as-judge verifier — kept for VERIFIER_MODE=llm reproducibility."""
    from src.infra.llm_client import LLMClient
    import json

    llm = LLMClient()

    def verifier(question, answer, chunks):
        if not chunks:
            return {"supported": False, "confidence": 0.0,
                    "unsupported_claims": ["no chunks"],
                    "evidence_chunk_ids": [], "reason": "no chunks retrieved"}
        ctx = "\n\n".join(
            f"[{i+1}] (p{c['page_number']}) {c.get('content','')[:300]}"
            for i, c in enumerate(chunks)
        )
        prompt = f"""严格判断ANSWER是否基于CONTEXT。输出JSON:
{{"supported":true/false,"confidence":0.0-1.0,"unsupported_claims":[],"evidence_chunk_ids":[],"reason":""}}
CONTEXT:\n{ctx}\nANSWER:\n{answer}\n只输出JSON:"""
        try:
            resp, _ = llm.chat([{"role": "user", "content": prompt}])
            s, e = resp.find("{"), resp.rfind("}") + 1
            if s >= 0:
                return json.loads(resp[s:e])
        except Exception:
            pass
        return {"supported": False, "confidence": 0.0,
                "unsupported_claims": ["parse error"],
                "evidence_chunk_ids": [], "reason": "verify parse failed"}

    return verifier


@lru_cache
def get_vqa():
    from src.workflow.verified_qa import VerifiedQA
    from src.generation.generator import generate_answer
    from src.workflow.grounding import GroundingVerifier, CrossEncoderScorer

    s = get_settings()
    if s.verifier_mode == "llm":
        verifier = _make_llm_verifier()
    elif s.grounding_scorer == "cosine":
        # Bi-encoder BGE-M3 cosine path (V6 original; less discriminative)
        verifier = GroundingVerifier(
            embedder=get_embedder(),
            initial_threshold=s.grounding_initial_threshold,
            threshold_floor=s.grounding_threshold_floor,
            decay=s.grounding_threshold_decay,
            min_support_ratio=s.grounding_min_support_ratio,
        )
    else:
        # V6 — cross-encoder grounding (default): joint (sentence, chunk) score
        verifier = GroundingVerifier(
            scorer=CrossEncoderScorer(get_reranker()),
            scorer_floor=s.grounding_scorer_floor,
            min_support_ratio=s.grounding_min_support_ratio,
        )
    return VerifiedQA(get_retriever(), generate_answer, verifier)


@lru_cache
def get_semantic_cache():
    from src.infra.semantic_cache import SemanticCache
    s = get_settings()
    return SemanticCache(
        embedder=get_embedder(),
        db_path=PROJECT_ROOT / s.cache_db_path,
        threshold=s.cache_threshold,
        ttl_days=s.cache_ttl_days,
    )


@lru_cache
def get_incremental_indexer():
    from src.ingestion.manifest import ManifestStore
    from src.ingestion.incremental import IncrementalIndexer
    store = ManifestStore(PROJECT_ROOT / "storage" / "manifests")
    return IncrementalIndexer(
        milvus_client=get_milvus_client(),
        collection_name=get_latest_v1_collection(),
        bm25=get_bm25(),
        manifest_store=store,
        embedder=get_embedder(),
    )
