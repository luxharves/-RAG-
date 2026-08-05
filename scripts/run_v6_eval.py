#!/usr/bin/env python
"""Phase V6 — Deterministic sentence-level grounding evaluation.

Phase 1 (default): 20 fixed questions + 2 edge cases through the full
    VerifiedQA pipeline with the grounding verifier (default thresholds).
    Caches answer + retrieved chunks per case for the sweep.

Phase 2 (default): threshold sweep over the CACHED answers — no retrieval,
    no LLM, only BGE-M3 encodes of each case once, then pure-Python decision
    per combo. Picks the recommended thresholds.

Phase 3 (default): status comparison vs V4 (LLM-as-judge) results.

--full: run all 100 golden questions (slow: reranker ~20s/question) + sweep.

Usage:
    python scripts/run_v6_eval.py
    python scripts/run_v6_eval.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os as _os
from dotenv import dotenv_values as _dv
_os.environ["MILVUS_URI"] = "http://localhost:19530"
_ENV = _dv(str(PROJECT_ROOT / ".env"))

# Low-level grounding pieces reused by the sweep (no re-embedding per combo)
from src.workflow.grounding import (  # noqa: E402
    split_sentences, strip_citation_markers,
)

MIN_SENT_LEN = 5
OUT_DIR = PROJECT_ROOT / "storage" / "runs" / "v6_grounding"

# Defaults (starting point; sweep calibrates them)
DEFAULT_SCORER_FLOOR = 0.1
DEFAULT_RATIO = 0.7

SWEEP_SCORER_FLOOR = [0.05, 0.1, 0.15, 0.2, 0.3]
SWEEP_RATIO = [0.6, 0.7, 0.8]

EDGE_CASES = [
    {"question": "这个设备能在火星上使用吗？", "label": "out_of_scope"},
    {"question": "如何更换核聚变反应堆？", "label": "nonsense"},
]


# ── Milvus ──


def _get_v1_col() -> str:
    """Latest multimodal collection name.

    Reuses the shared lru-cached Milvus client from deps (a single connection)
    instead of opening/closing a throwaway one, which can trip Milvus Lite's
    gRPC keepalive throttling and stall the first search.
    """
    from src.api.deps import get_latest_v1_collection
    return get_latest_v1_collection()


# ── Case loading ──


def _load_cases(full: bool) -> list[dict]:
    if full:
        with open(PROJECT_ROOT / "data" / "eval_dataset" / "golden_100.json", encoding="utf-8") as f:
            data = json.load(f)
        return [
            {"question": q["question"], "gold_pages": q.get("gold_pages", []),
             "modality": q.get("modality_required", "text"), "label": "fixed"}
            for q in data
        ]
    with open(PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    return [
        {"question": q["question"], "gold_pages": q.get("gold_pages", []),
         "modality": q.get("modality_required", "text"), "label": "fixed"}
        for q in questions
    ] + EDGE_CASES


# ── Phase 1: full pipeline ──


def _run_case(vqa, case: dict, doc_filter: str | None = None) -> dict:
    t0 = time.perf_counter()
    state = vqa.run(case["question"], doc_filter=doc_filter)
    elapsed = time.perf_counter() - t0

    gold = case.get("gold_pages", [])
    pages = [int(c.get("page_number", 0)) for c in state.get("retrieved_chunks", [])]
    hit = bool(set(gold) & set(pages[:5])) if gold else None

    return {
        "question": case["question"],
        "label": case["label"],
        "modality": case.get("modality", "text"),
        "gold_pages": gold,
        "answer": state.get("answer", ""),
        "final_status": state.get("final_status", "refused"),
        "verification_result": state.get("verification_result", {}),
        "retry_count": state.get("retry_count", 0),
        "trace": state.get("trace", []),
        "retrieved_pages": pages[:5],
        "retrieval_hit": hit,
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "time_s": round(elapsed, 2),
    }


# ── Phase 2: sweep over cached answers (encode once) ──


def _precompute(scorer, answer: str, chunks: list[dict]) -> dict | None:
    valid = [c for c in chunks if str(c.get("content", "")).strip()]
    if not valid:
        return None
    chunk_texts = [c["content"] for c in valid]
    rows: list[dict] = []
    for sent in split_sentences(answer or ""):
        clean = strip_citation_markers(sent)
        if len(clean) >= MIN_SENT_LEN:
            try:
                scores = scorer(clean, chunk_texts)
                best = max(scores) if scores else 0.0
            except Exception:
                best = 0.0
            rows.append({"clean": clean, "best": float(best)})
    return {"rows": rows}


def _decide(pre: dict | None, answer: str,
            floor: float, ratio: float) -> dict:
    if any(p in (answer or "") for p in ("无法回答", "无法回答此问题")):
        return {"status": "refused", "supported": False, "ratio": 0.0,
                "n_supported": 0, "n": 0}
    if pre is None:
        return {"status": "refused", "supported": False, "ratio": 0.0,
                "n_supported": 0, "n": 0}
    n = len(pre["rows"])
    if n == 0:
        return {"status": "answered", "supported": True, "ratio": 1.0,
                "n_supported": 0, "n": 0}
    n_sup = sum(1 for row in pre["rows"] if row["best"] >= floor)
    r = n_sup / n
    ok = r >= ratio
    return {"status": "answered" if ok else "refused", "supported": ok,
            "ratio": round(r, 4), "n_supported": n_sup, "n": n}


def _sweep(scorer, records: list[dict]) -> dict:
    # Precompute cross-encoder scores once per case (only GPU cost in the sweep)
    pre = []
    for rec in records:
        pre.append(_precompute(scorer, rec.get("answer", ""),
                               rec.get("retrieved_chunks", [])))

    combos = [
        (fl, ra)
        for fl in SWEEP_SCORER_FLOOR for ra in SWEEP_RATIO
    ]
    sweep_results = []
    for fl, ra in combos:
        answered_fixed = 0
        refused_edge = 0
        for i, rec in enumerate(records):
            dec = _decide(pre[i], rec.get("answer", ""), fl, ra)
            if rec["label"] == "fixed":
                if dec["status"] == "answered":
                    answered_fixed += 1
            elif dec["status"] == "refused":
                refused_edge += 1
        sweep_results.append({
            "scorer_floor": fl,
            "min_support_ratio": ra,
            "answered_fixed": answered_fixed,
            "refused_edge": refused_edge,
            "score": answered_fixed + refused_edge,
        })

    # Recommended: max score, tie-break toward stricter floor
    best = max(sweep_results, key=lambda r: (r["score"], r["scorer_floor"]))
    return {"combos": sweep_results, "recommended": {
        "scorer": "reranker",
        "scorer_floor": best["scorer_floor"],
        "min_support_ratio": best["min_support_ratio"],
        "answered_fixed": best["answered_fixed"],
        "refused_edge": best["refused_edge"],
    }}


# ── Phase 3: V4 comparison ──


def _compare_v4(records: list[dict]) -> list[dict]:
    v4_path = PROJECT_ROOT / "storage" / "runs" / "v4_verified" / "v4_results.json"
    if not v4_path.exists():
        return []
    with open(v4_path, encoding="utf-8") as f:
        v4 = json.load(f)
    v4_by_q = {r["question"]: r for r in v4}
    out = []
    for rec in records:
        if rec["label"] != "fixed":
            continue
        v4r = v4_by_q.get(rec["question"])
        if v4r is None:
            continue
        out.append({
            "question": rec["question"],
            "v4_status": v4r.get("final_status", ""),
            "v6_status": rec["final_status"],
            "v4_supported": v4r.get("verification_result", {}).get("supported", False),
            "v6_supported": rec["verification_result"].get("supported", False),
            "flipped": (v4r.get("final_status", "") != rec["final_status"]),
        })
    return out


# ── Summary ──


def _summarize(records: list[dict]) -> dict:
    fixed = [r for r in records if r["label"] == "fixed"]
    edge = [r for r in records if r["label"] != "fixed"]
    answered = sum(1 for r in fixed if r["final_status"] == "answered")
    refused_edge_ok = all(r["final_status"] == "refused" for r in edge)
    ratios = [
        r["verification_result"].get("grounding_meta", {}).get("support_ratio")
        for r in records if r["verification_result"].get("grounding_meta")
    ]
    return {
        "total_cases": len(records),
        "fixed_questions": len(fixed),
        "fixed_answered": answered,
        "edge_cases": len(edge),
        "edge_all_refused": refused_edge_ok,
        "avg_support_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "retries_used": sum(1 for r in records if r["retry_count"] > 0),
    }


# ── Main ──


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 grounding evaluation")
    parser.add_argument("--full", action="store_true",
                        help="Run all 100 golden questions (slow) + sweep")
    args = parser.parse_args()

    from src.infra.reranker import Reranker
    from src.retrieval.reranked_retriever import RerankedRetriever
    from src.generation.generator import generate_answer
    from src.workflow.verified_qa import VerifiedQA
    from src.workflow.grounding import GroundingVerifier, CrossEncoderScorer
    from src.eval.doc_registry import resolve_doc_filter

    ROBOROCK_FILTER = resolve_doc_filter("Roborock G10S")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Setup ──
    rr = Reranker()
    rr.load()
    print(f"Reranker loaded: {rr.model_name}")

    retriever = RerankedRetriever(
        collection_name=_get_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
        reranker=rr,  # share ONE cross-encoder for rerank + grounding
    )
    scorer = CrossEncoderScorer(rr)
    verifier = GroundingVerifier(
        scorer=scorer,
        scorer_floor=DEFAULT_SCORER_FLOOR,
        min_support_ratio=DEFAULT_RATIO,
        audit_citations=True,
    )
    vqa = VerifiedQA(retriever=retriever, generator_fn=generate_answer,
                     verifier_fn=verifier, max_retries=1)

    cases = _load_cases(args.full)
    print(f"V6 grounding eval: {len(cases)} cases (full={args.full})")

    # ── Phase 1 ──
    records = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"  [{i:3d}/{len(cases)}] START {case['question'][:40]}...",
                  flush=True)
            try:
                rec = _run_case(vqa, case, doc_filter=ROBOROCK_FILTER)
            except Exception as exc:  # one failing case must not kill the run
                rec = {
                    "question": case["question"], "label": case["label"],
                    "modality": case.get("modality", "text"),
                    "gold_pages": case.get("gold_pages", []),
                    "answer": "", "final_status": "error",
                    "verification_result": {}, "retry_count": 0,
                    "trace": [], "retrieved_pages": [], "retrieval_hit": None,
                    "retrieved_chunks": [], "time_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  [{i:3d}/{len(cases)}] ERROR {type(exc).__name__}: {exc}",
                      flush=True)
            records.append(rec)
            print(f"  [{i:3d}/{len(cases)}] {rec['final_status']:8s} "
                  f"retry={rec['retry_count']} {case['question'][:40]}...",
                  flush=True)
    finally:
        retriever.close()

    out_name = "v6_results.json" if not args.full else "v6_full_results.json"
    with open(OUT_DIR / out_name, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # ── Phase 2: sweep ──
    print("\nSweeping scorer floors over cached answers...")
    sweep = _sweep(scorer, records)
    with open(OUT_DIR / "sweep_report.json", "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)
    rec = sweep["recommended"]
    print(f"Recommended: scorer_floor={rec['scorer_floor']} "
          f"ratio={rec['min_support_ratio']} → answered_fixed={rec['answered_fixed']} "
          f"refused_edge={rec['refused_edge']}")

    # ── Phase 3: V4 comparison ──
    comparison = _compare_v4(records)
    if comparison:
        with open(OUT_DIR / "v6_vs_v4_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        flipped = [c for c in comparison if c["flipped"]]
        print(f"V4 comparison: {len(comparison)} matched, {len(flipped)} flipped status")

    # ── Summary + metadata ──
    summary = _summarize(records)
    summary["run"] = "full" if args.full else "20+2"
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*52}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
