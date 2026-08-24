"""FastAPI routes for the RAG demo."""

from __future__ import annotations

import json
import time
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.api.deps import (
    get_settings, get_retriever, get_vqa, get_embedder,
    get_milvus_client, get_latest_v1_collection, get_bm25,
    get_incremental_indexer, get_semantic_cache, PROJECT_ROOT,
)

router = APIRouter()


# ═══════════════════ Models ═══════════════════

class QueryRequest(BaseModel):
    question: str
    experiment: str = "v4"
    source_document: str | None = None  # friendly name, e.g. "Roborock G10S" (V8)


class QueryResponse(BaseModel):
    question: str
    answer: str
    final_status: str
    experiment: str
    sources: list[dict]
    evidence_chunk_ids: list[str]
    verification: dict
    timing_s: float
    cache_hit: bool = False
    cache_source: str | None = None  # "exact" | "semantic" | None
    doc_filter: str | None = None    # resolved source_file scope (V8)


class IngestResponse(BaseModel):
    document_id: str
    version: str
    chunks: int
    pages: int
    status: str
    added: int = 0
    unchanged: int = 0
    modified: int = 0
    deleted: int = 0
    reprocessed_pages: int = 0
    reused_chunks: int = 0
    embedded_chunks: int = 0
    removed_chunks: int = 0


class DocumentItem(BaseModel):
    document_id: str
    source_file: str
    version: str
    num_chunks: int
    status: str = "indexed"


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]


class EvaluateRequest(BaseModel):
    experiment: str = "v0_naive"
    dataset: str = "v0_questions"
    max_questions: int = 10


# ═══════════════════ Routes ═══════════════════

@router.get("/health")
async def health():
    settings = get_settings()
    final_eval_exists = (PROJECT_ROOT / "storage" / "runs" / "final_eval" / "final_metrics.json").exists()
    return {
        "status": "ok",
        "version": "V9",
        "collection": get_latest_v1_collection(),
        "milvus": settings.milvus_uri,
        "bm25_docs": get_bm25().num_docs,
        "final_eval": final_eval_exists,
        "models": {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "llm": settings.llm_model,
        },
    }


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF, save, parse, embed via IncrementalIndexer."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf",):
        raise HTTPException(422, f"Unsupported file type: {suffix}. Only PDF accepted.")

    settings = get_settings()
    raw_dir = PROJECT_ROOT / settings.data_dir / "raw_docs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Validate PDF can be opened
    try:
        import fitz
        doc = fitz.open(dest)
        total_pages = len(doc)
        doc.close()
        if total_pages == 0:
            dest.unlink()
            raise HTTPException(422, "PDF has no pages")
    except HTTPException:
        raise
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise HTTPException(422, f"Invalid PDF file: {e}")

    # Use IncrementalIndexer
    indexer = get_incremental_indexer()
    counts = indexer.process(str(raw_dir))
    bm = get_bm25()
    bm.save(PROJECT_ROOT / "storage" / "bm25")

    # Find the manifest for this file
    from src.ingestion.manifest import ManifestStore
    store = ManifestStore(PROJECT_ROOT / "storage" / "manifests")
    m = store.get(str(dest))

    return IngestResponse(
        document_id=m.document_id if m else "unknown",
        version=m.version if m else "unknown",
        chunks=counts["embedded_chunks"] + counts["reused_chunks"],
        pages=counts["reprocessed_pages"],
        status="ingested",
        added=counts["added"],
        unchanged=counts["unchanged"],
        modified=counts["modified"],
        deleted=counts["deleted"],
        reprocessed_pages=counts["reprocessed_pages"],
        reused_chunks=counts["reused_chunks"],
        embedded_chunks=counts["embedded_chunks"],
        removed_chunks=counts["removed_chunks"],
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """Return all tracked documents from the manifest store."""
    from src.ingestion.manifest import ManifestStore
    store = ManifestStore(PROJECT_ROOT / "storage" / "manifests")
    docs = []
    for source_file in sorted(store.all_files()):
        m = store.get(source_file)
        if m:
            docs.append(DocumentItem(
                document_id=m.document_id,
                source_file=Path(m.source_file).name,
                version=m.version,
                num_chunks=m.num_chunks,
                status="indexed",
            ))
    return DocumentListResponse(documents=docs)


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Answer via the LangGraph pipeline (V4→V6), with optional doc_filter
    scope (V8) and the V9 semantic-cache fast-path. The cache is salted with
    the doc_filter so answers scoped to different manuals never collide."""
    t0 = time.perf_counter()
    settings = get_settings()

    from src.eval.doc_registry import resolve_doc_filter
    doc_filter = resolve_doc_filter(req.source_document) if req.source_document else None
    salt = doc_filter or ""

    # ── V9 semantic cache: exact (SHA256) then semantic (BGE-M3 cosine) ──
    cache = None
    if settings.cache_enabled:
        cache = get_semantic_cache()
        cached = cache.get(req.question, salt=salt)
        if cached is not None:
            resp, source = cached
            resp["cache_hit"] = True
            resp["cache_source"] = source
            resp["timing_s"] = round(time.perf_counter() - t0, 2)
            return QueryResponse(**resp)

    vqa = get_vqa()
    state = vqa.run(req.question, doc_filter=doc_filter)

    sources = []
    for c in state.get("retrieved_chunks", []):
        sources.append({
            "chunk_id": c.get("chunk_id", ""),
            "source_file": Path(c.get("source_file", "")).name if c.get("source_file") else "",
            "page_number": c.get("page_number", 0),
            "content_type": c.get("content_type", "text"),
            "rerank_score": c.get("rerank_score"),
        })

    vr = state.get("verification_result", {})
    resp = QueryResponse(
        question=req.question,
        answer=state.get("answer", ""),
        final_status=state.get("final_status", "refused"),
        experiment=req.experiment,
        sources=sources,
        evidence_chunk_ids=[str(x) for x in vr.get("evidence_chunk_ids", [])],
        verification={
            "supported": vr.get("supported", False),
            "confidence": vr.get("confidence", 0.0),
            "reason": vr.get("reason", ""),
            "unsupported_claims": vr.get("unsupported_claims", []),
            "grounding_meta": vr.get("grounding_meta"),
            "sentence_evidence": vr.get("sentence_evidence"),
        },
        timing_s=round(time.perf_counter() - t0, 2),
        doc_filter=doc_filter,
    )
    if cache is not None:
        cache.put(req.question, resp.model_dump(), salt=salt)
    return resp


@router.get("/system")
async def system_status():
    """Live system state: LLM gateway circuit breakers (V7), semantic cache
    stats (V9), and grounding verifier config (V6)."""
    settings = get_settings()
    from src.infra.gateway import get_gateway
    gateway = get_gateway()
    cache = get_semantic_cache() if settings.cache_enabled else None
    return {
        "version": "V9",
        "gateway": gateway.state_dump(),
        "cache": cache.stats() if cache else None,
        "grounding": {
            "verifier_mode": settings.verifier_mode,
            "scorer": settings.grounding_scorer,
            "scorer_floor": settings.grounding_scorer_floor,
            "min_support_ratio": settings.grounding_min_support_ratio,
        },
    }


@router.get("/versions")
async def version_highlights():
    """V6/V8/V9 evaluation highlights, read from storage JSONs (README values
    as fallback so the page renders even before a fresh eval run)."""
    settings = get_settings()
    from src.infra.gateway import get_gateway

    # ── V6 deterministic grounding ──
    v6 = {"total": 100, "answered": 95, "avg_support_ratio": 0.9935,
          "poison_flagged": 28, "poison_total": 34, "poison_rate": round(28 / 34, 4),
          "over_refused": 1, "retries_used": 5}
    p = PROJECT_ROOT / "storage" / "runs" / "v6_grounding" / "metadata.json"
    if p.exists():
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            v6 = {**v6,
                  "total": m.get("total_cases", 100),
                  "answered": m.get("fixed_answered", 95),
                  "avg_support_ratio": m.get("avg_support_ratio", 0.9935),
                  "retries_used": m.get("retries_used", 5)}
        except Exception:
            pass
    pp = PROJECT_ROOT / "storage" / "runs" / "v6_grounding" / "poison_test.json"
    if pp.exists():
        try:
            rows = json.loads(pp.read_text(encoding="utf-8"))
            flagged = sum(1 for r in rows if r.get("poison_flagged_unsupported"))
            total = len(rows)
            if total:
                v6 = {**v6, "poison_flagged": flagged, "poison_total": total,
                      "poison_rate": round(flagged / total, 4)}
        except Exception:
            pass

    # ── V7 LLM gateway (live config) ──
    gw = get_gateway()
    v7 = {"timeout_s": settings.llm_timeout,
          "max_retries": settings.llm_max_retries,
          "circuit_threshold": settings.llm_circuit_threshold,
          "cooldown_s": settings.llm_circuit_cooldown,
          "configured_providers": len(gw.state_dump().get("providers", []))}

    # ── V8 multi-doc (123-question retrieval-only) ──
    v8 = {"questions": 123, "text": 114, "image": 9,
          "v3_hit_at_5": 0.9756, "v3_mrr": 0.8916, "v3_top1": 0.8293,
          "ecovacs_mrr": 0.90}
    p = PROJECT_ROOT / "storage" / "runs" / "final_eval_extended_full" / "final_metrics.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            rm = d.get("versions", {}).get("V3", {}).get("retrieval_metrics", {})
            v8 = {**v8,
                  "questions": d.get("total_questions", 123),
                  "v3_hit_at_5": rm.get("hit_at_5", 0.9756),
                  "v3_mrr": rm.get("mrr", 0.8916),
                  "v3_top1": rm.get("top1_hit_rate", 0.8293)}
            split = d.get("versions", {}).get("V3", {}).get("split_by_document", {})
            eco = split.get("Ecovacs DEEBOT T30C") or split.get("Ecovacs")
            if eco:
                v8["ecovacs_mrr"] = eco.get("mrr", 0.90)
        except Exception:
            pass

    # ── V9 semantic cache ──
    v9 = {"warmed": 12, "exact_hits": 12, "semantic_hits": 4,
          "overall_hit_rate": 0.6667, "avg_cached_s": 0.031,
          "avg_uncached_s": 113.62, "llm_calls_saved": 12}
    p = PROJECT_ROOT / "storage" / "runs" / "v9_cache" / "cache_eval.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            v9 = {**v9,
                  "warmed": d.get("warmed", 12),
                  "exact_hits": d.get("exact_hits", 12),
                  "semantic_hits": d.get("semantic_hits", 4),
                  "overall_hit_rate": d.get("overall_hit_rate", 0.6667),
                  "avg_cached_s": d.get("avg_cached_s", 0.031),
                  "avg_uncached_s": d.get("avg_uncached_s", 113.62),
                  "llm_calls_saved": d.get("llm_calls_saved", 12)}
        except Exception:
            pass

    return {"version": "V9",
            "v6_grounding": v6, "v7_gateway": v7,
            "v8_multidoc": v8, "v9_cache": v9}


@router.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    """Return existing cached metrics for the given experiment."""
    runs_dir = PROJECT_ROOT / "storage" / "runs" / req.experiment
    if not runs_dir.exists():
        raise HTTPException(404, f"Experiment not found: {req.experiment}")

    # Gather all metrics from JSON files
    metrics: dict[str, Any] = {}
    for f in sorted(runs_dir.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for k in ("metrics", "summary", "retrieval_metrics",
                          "ragas_metrics", "hit_rate", "recall_at_5",
                          "mrr", "top5_hit_rate", "faithfulness",
                          "context_precision", "context_recall",
                          "answer_relevancy"):
                    if k in data and data[k] is not None:
                        metrics[k] = data[k]
        except Exception:
            pass

    return {
        "run_id": req.experiment,
        "dataset": req.dataset,
        "metrics": metrics,
        "files": [f.name for f in sorted(runs_dir.iterdir()) if f.suffix == ".json"],
    }


@router.get("/final_eval")
async def get_final_eval():
    """Return the final evaluation metrics (V0-V4 with RAGAS)."""
    final_path = PROJECT_ROOT / "storage" / "runs" / "final_eval" / "final_metrics.json"
    if not final_path.exists():
        raise HTTPException(404, "Final evaluation not found")
    with open(final_path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/experiments")
async def list_experiments():
    """Return all available experiment versions."""
    runs_dir = PROJECT_ROOT / "storage" / "runs"
    # Also check root-level files for V1/V2
    root_files = {f.name for f in runs_dir.iterdir() if f.is_file() and f.suffix == ".json"}
    exp_defs = [
        {"id": "v0_baseline", "name": "V0 Baseline", "description": "文本 + Dense Retrieval", "key_metric": "recall_at_5"},
        {"id": "v1_multimodal", "name": "V1 Multimodal", "description": "+ 图片 VLM 描述 + 表格", "key_metric": "image_chunks",
         "root_file": "v0_v1_comparison.json"},
        {"id": "v2_comparison", "name": "V2 Hybrid Retrieval", "description": "+ BM25 + Dense RRF 融合", "key_metric": "hybrid_hit_rate",
         "root_file": "v2_comparison.json"},
        {"id": "v3_rerank", "name": "V3 Reranker", "description": "+ BGE-Reranker-v2-m3 精排", "key_metric": "mrr"},
        {"id": "v4_verified", "name": "V4 LangGraph Verify", "description": "+ 验证节点 + 重试 + 拒答", "key_metric": "verified"},
        {"id": "v5_incremental", "name": "V5 Incremental Update", "description": "+ SHA256 Hash 增量更新", "key_metric": "reused_chunks"},
        {"id": "v6_grounding", "name": "V6 Deterministic Grounding", "description": "+ 句级相似度接地验证(确定性,非 LLM 自查)", "key_metric": "support_ratio"},
    ]
    result = []
    for exp in exp_defs:
        exp_dir = runs_dir / exp["id"]
        has_dir = exp_dir.exists()
        has_root_file = exp.get("root_file", "") in root_files
        exp["available"] = has_dir or has_root_file
        files = []
        if has_dir:
            files = [f.name for f in sorted(exp_dir.iterdir()) if f.suffix == ".json"]
        if has_root_file:
            files.append(exp["root_file"])
        exp["files"] = files
        result.append(exp)
    return {"experiments": result}


def _extract_metrics(data: dict, prefix: str = "") -> dict:
    """Recursively flatten nested metrics into a single dict."""
    result = {}
    known_metrics = {
        "hit_rate", "recall_at_5", "mrr", "faithfulness",
        "context_precision", "context_recall", "answer_relevancy",
        "top5_hit_rate", "top1_hit_rate", "top5_hit_count",
        "added_count", "unchanged_count", "modified_count",
        "deleted_count", "reprocessed_pages", "reused_chunks",
        "embedded_chunks", "removed_chunks",
        "v1_hit_rate", "v0_hit_rate",  # V1 comparison
    }
    # Map version-prefixed keys to generic metric names
    key_aliases = {
        "v1_hit_rate": "hit_rate",
        "v0_hit_rate": "hit_rate",
    }
    for key, val in data.items():
        if key in known_metrics and isinstance(val, (int, float)):
            result[key] = val
            # Also emit alias
            if key in key_aliases:
                result[key_aliases[key]] = val
        elif isinstance(val, dict):
            sub = _extract_metrics(val, f"{prefix}{key}.")
            result.update(sub)
    return result


@router.get("/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    """Read existing experiment metadata and metrics."""
    runs_dir = PROJECT_ROOT / "storage" / "runs" / exp_id
    root_runs = PROJECT_ROOT / "storage" / "runs"

    # Map V1/V2 to their root-level files
    root_file_map = {
        "v1_multimodal": ["v0_v1_comparison.json", "v1_ingest/metadata.json"],
        "v2_comparison": ["v2_comparison.json"],
    }

    if not runs_dir.exists() and exp_id not in root_file_map:
        raise HTTPException(404, f"Experiment not found: {exp_id}")

    info: dict[str, Any] = {"id": exp_id, "files": [], "metrics": {}}
    all_metrics: dict[str, float] = {}

    # Read from subdirectory if exists
    sources = []
    if runs_dir.exists():
        sources.extend(str(f) for f in sorted(runs_dir.iterdir()) if f.suffix == ".json")
    # Also read root-level files for V1/V2
    for rf in root_file_map.get(exp_id, []):
        p = root_runs / rf
        if p.exists():
            sources.append(str(p))

    for src in sources:
        f = Path(src)
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                flat = _extract_metrics(data)
                # Prefer first occurrence (phase2 > retrieval_metrics)
                for k, v in flat.items():
                    if k not in all_metrics:
                        all_metrics[k] = v
                for k in ("experiment", "version", "timestamp", "total_questions"):
                    if k in data:
                        info.setdefault("metadata", {})[k] = data[k]
        except Exception:
            pass
        info["files"].append(f.name)

    # Unified flat metrics output: null for missing, real value for present
    metric_keys = [
        "hit_rate", "recall_at_5", "mrr", "faithfulness",
        "context_precision", "context_recall", "answer_relevancy",
        "top5_hit_rate", "top1_hit_rate",
    ]
    for mk in metric_keys:
        v = all_metrics.get(mk)
        info["metrics"][mk] = round(float(v), 4) if v is not None else None

    # Incremental metrics (V5)
    inc_keys = ["added_count", "unchanged_count", "modified_count",
                "deleted_count", "reprocessed_pages", "reused_chunks",
                "embedded_chunks", "removed_chunks"]
    for ik in inc_keys:
        v = all_metrics.get(ik)
        if v is not None:
            info.setdefault("incremental_metrics", {})[ik] = int(v)

    return info
