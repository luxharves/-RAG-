#!/usr/bin/env python
"""Phase 6 — V4 Verified QA evaluation on 20 fixed questions + edge cases."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os as _os
from dotenv import dotenv_values as _dv
_os.environ["MILVUS_URI"] = "http://localhost:19530"
_ENV = _dv(str(PROJECT_ROOT / ".env"))


def _make_verifier():
    """LLM-as-judge verifier: checks if answer is supported by chunks."""
    from src.infra.llm_client import LLMClient
    client = LLMClient()

    def verify(question: str, answer: str, chunks: list[dict]) -> dict:
        if not chunks:
            return {"supported": False, "confidence": 0.0,
                    "unsupported_claims": ["无检索结果"],
                    "evidence_chunk_ids": [], "reason": "no chunks retrieved"}

        ctx = "\n\n".join(
            f"[{i+1}] (page {c['page_number']}, id={c.get('chunk_id','')[:8]})\n{c.get('content','')[:400]}"
            for i, c in enumerate(chunks)
        )
        prompt = f"""你是严格的事实核查员。判断ANSWER是否完全基于CONTEXT。

CONTEXT:
{ctx}

ANSWER:
{answer}

输出JSON:
{{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "unsupported_claims": ["claim1", ...],
  "evidence_chunk_ids": ["id1", ...],
  "reason": "判断理由"
}}

只输出JSON:"""

        try:
            resp, _ = client.chat([{"role": "user", "content": prompt}])
            start = resp.find("{")
            end = resp.rfind("}") + 1
            if start >= 0:
                result = json.loads(resp[start:end])
                return {
                    "supported": result.get("supported", False),
                    "confidence": result.get("confidence", 0.0),
                    "unsupported_claims": result.get("unsupported_claims", []),
                    "evidence_chunk_ids": result.get("evidence_chunk_ids", []),
                    "reason": result.get("reason", ""),
                }
        except Exception:
            pass
        # Parse failure → NOT supported
        return {"supported": False, "confidence": 0.0,
                "unsupported_claims": ["verify_parse_failed"],
                "evidence_chunk_ids": [], "reason": "verify output parse error"}

    return verify


def _get_v1_col():
    from pymilvus import MilvusClient
    client = MilvusClient(str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db")))
    kw = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_kw_")])
    ts = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    client.close()
    return (ts + kw)[-1]


def main() -> None:
    from src.retrieval.reranked_retriever import RerankedRetriever
    from src.generation.generator import generate_answer
    from src.workflow.verified_qa import VerifiedQA
    from src.eval.doc_registry import resolve_doc_filter

    ROBOROCK_FILTER = resolve_doc_filter("Roborock G10S")

    # ── Setup ──
    retriever = RerankedRetriever(
        collection_name=_get_v1_col(),
        bm25_index_path=str(PROJECT_ROOT / "storage" / "bm25"),
    )
    verifier_fn = _make_verifier()
    vqa = VerifiedQA(
        retriever=retriever,
        generator_fn=generate_answer,
        verifier_fn=verifier_fn,
        max_retries=1,
    )

    # ── 20 fixed questions ──
    with open(PROJECT_ROOT / "data" / "eval_dataset" / "v0_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    # ── Extra test cases ──
    extra_cases = [
        {"question": "这个设备能在火星上使用吗？", "label": "out_of_scope"},
        {"question": "如何更换核聚变反应堆？", "label": "nonsense"},
    ]

    all_cases = [
        {"question": q["question"], "gold_pages": q.get("gold_pages", []),
         "modality": q.get("modality_required", "text"), "label": "fixed"}
        for q in questions
    ] + extra_cases

    results = []
    print(f"V4 Verified QA: {len(all_cases)} cases")

    for i, case in enumerate(all_cases, 1):
        q_text = case["question"]
        t0 = time.perf_counter()
        state = vqa.run(q_text, doc_filter=ROBOROCK_FILTER)
        elapsed = time.perf_counter() - t0

        gold = case.get("gold_pages", [])
        pages = [c["page_number"] for c in state["retrieved_chunks"]]
        hit = bool(set(gold) & set(pages[:5])) if gold else None

        results.append({
            "question": q_text,
            "label": case["label"],
            "modality": case.get("modality", "text"),
            "gold_pages": gold,
            "answer": state["answer"],
            "final_status": state["final_status"],
            "verification_result": state["verification_result"],
            "retry_count": state["retry_count"],
            "trace": state["trace"],
            "retrieved_pages": pages[:5],
            "retrieval_hit": hit,
            "time_s": round(elapsed, 2),
        })
        print(f"  [{i:2d}/{len(all_cases)}] {state['final_status']:8s} "
              f"retry={state['retry_count']} {q_text[:40]}...")

    retriever.close()

    # ── Summary ──
    verified = sum(1 for r in results if r["final_status"] == "verified")
    retried = sum(1 for r in results if r["retry_count"] > 0)
    refused = sum(1 for r in results if r["final_status"] == "refused")

    # ── Save ──
    out_dir = PROJECT_ROOT / "storage" / "runs" / "v4_verified"
    out_dir.mkdir(parents=True, exist_ok=True)

    # v4_results.json
    with open(out_dir / "v4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # verification_cases.json — edge cases only
    edge = [r for r in results if r["label"] not in ("fixed",)]
    with open(out_dir / "verification_cases.json", "w", encoding="utf-8") as f:
        json.dump(edge, f, ensure_ascii=False, indent=2)

    # V3/V4 comparison (hit rate only — V4 doesn't change retrieval)
    with open(PROJECT_ROOT / "storage" / "runs" / "v3_rerank" / "v3_results.json", encoding="utf-8") as f:
        v3_data = json.load(f)

    v3v4 = []
    for v3r, v4r in zip(v3_data, [r for r in results if r["label"] == "fixed"]):
        v3v4.append({
            "question": v4r["question"],
            "v3_answer": v3r.get("pages", []),
            "v4_status": v4r["final_status"],
            "v4_answer": v4r["answer"][:200],
            "v4_verified": v4r["verification_result"].get("supported", False),
        })
    with open(out_dir / "v3_v4_comparison.json", "w", encoding="utf-8") as f:
        json.dump(v3v4, f, ensure_ascii=False, indent=2)

    meta = {
        "experiment": "v4_verified", "version": "V4",
        "total_cases": len(results),
        "verified": verified, "refused": refused, "retried": retried,
        "fixed_questions": len(questions),
        "edge_cases": len(extra_cases),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Verified: {verified}  Refused: {refused}  Retried: {retried}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
