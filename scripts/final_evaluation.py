#!/usr/bin/env python
"""Run the final, fair evaluation for V0-V4 on the same 100 questions.

This script deliberately does not read old experiment result files. It creates
fresh answers and retrieval results in storage/runs/final_eval and can resume
at a completed version using the checkpoint files in that directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
import src.eval.ragas_patch  # noqa: E402,F401

DATASET_PATH = PROJECT_ROOT / "data" / "eval_dataset" / "golden_100.json"
RUN_DIR = PROJECT_ROOT / "storage" / "runs" / "final_eval"
ARCHIVE_DIR = PROJECT_ROOT / "storage" / "runs" / "archive_before_final_eval"
TOP_K = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions() -> list[dict[str, Any]]:
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not questions:
        raise RuntimeError("Dataset is empty")
    required = {"question", "reference_answer", "reference_contexts", "gold_pages"}
    for index, question in enumerate(questions, 1):
        missing = required - question.keys()
        if missing:
            raise RuntimeError(f"Question {index} missing fields: {sorted(missing)}")
    return questions


def source_document_path() -> Path:
    candidates = sorted((PROJECT_ROOT / "data" / "raw_docs").glob("*.pdf"))
    preferred = [p for p in candidates if "Roborock G10S" in p.name]
    if not preferred:
        raise RuntimeError("Roborock G10S source PDF was not found")
    return preferred[0]


def metric_values(gold_pages: list[int], retrieved_pages: list[int]) -> dict[str, Any]:
    top = retrieved_pages[:TOP_K]
    gold = set(gold_pages)
    found = len(gold.intersection(top))
    reciprocal_rank = 0.0
    for rank, page in enumerate(top, 1):
        if page in gold:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "hit_at_5": int(bool(gold.intersection(top))),
        "recall_at_5": round(found / len(gold_pages), 4) if gold_pages else 1.0,
        "reciprocal_rank": round(reciprocal_rank, 4),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evaluated_question_count": len(results),
        "hit_at_5": round(sum(r["hit_at_5"] for r in results) / len(results), 4),
        "recall_at_5": round(sum(r["recall_at_5"] for r in results) / len(results), 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in results) / len(results), 4),
        "top1_hit_rate": round(
            sum(int(bool(r["gold_pages"] and r["retrieved_pages"] and r["retrieved_pages"][0] in r["gold_pages"])) for r in results)
            / len(results),
            4,
        ),
        "avg_retrieval_latency_s": round(
            sum(r["retrieval_latency_s"] for r in results) / len(results), 4
        ),
        "avg_generation_latency_s": round(
            sum(r["generation_latency_s"] for r in results) / len(results), 4
        ),
    }


def finite_or_none(value: Any) -> float | None:
    """Convert RAGAS NaN/inf values to JSON-safe null."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if math.isfinite(number) else None


def latest_v1_collection() -> str:
    from pymilvus import MilvusClient
    from dotenv import dotenv_values

    env = dotenv_values(str(PROJECT_ROOT / ".env"))
    uri = str(env.get("MILVUS_URI", "milvus.db"))
    uri = str(PROJECT_ROOT / uri) if not uri.startswith("http") else uri
    client = MilvusClient(uri)
    collections = sorted(c for c in client.list_collections() if c.startswith("v1_multimodal_kw_"))
    if not collections:
        collections = sorted(c for c in client.list_collections() if c.startswith("v1_multimodal_"))
    client.close()
    if not collections:
        raise RuntimeError("No V1 multimodal Milvus collection found")
    return collections[-1]


def retrieve_and_generate(
    version: str, questions: list[dict[str, Any]], retrieval_only: bool = False,
) -> list[dict[str, Any]]:
    from src.generation.generator import generate_answer
    from src.eval.doc_registry import resolve_doc_filter
    from src.retrieval.retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranked_retriever import RerankedRetriever

    collection = latest_v1_collection()
    bm25_path = str(PROJECT_ROOT / "storage" / "bm25")
    if version == "V0":
        retriever = DenseRetriever(collection_name="v0_naive_rag")
    elif version == "V1":
        retriever = DenseRetriever(collection_name=collection)
    elif version == "V2":
        retriever = HybridRetriever(collection_name=collection, bm25_index_path=bm25_path)
    else:
        retriever = RerankedRetriever(
            collection_name=collection,
            bm25_index_path=bm25_path,
            candidate_top_k=20,
            final_top_k=TOP_K,
        )

    verifier = None
    vqa = None
    if version == "V4":
        from src.workflow.verified_qa import VerifiedQA

        verifier_client = __import__("src.infra.llm_client", fromlist=["LLMClient"]).LLMClient()

        def verifier(question: str, answer: str, chunks: list[dict]) -> dict:
            context = "\n\n".join(
                f"[{i + 1}] page={c.get('page_number')}\n{c.get('content', '')[:500]}"
                for i, c in enumerate(chunks)
            )
            prompt = f"""严格判断 ANSWER 是否完全由 CONTEXT 支持。
QUESTION: {question}
CONTEXT:
{context}
ANSWER:
{answer}
只输出 JSON：{{"supported": true/false, "confidence": 0.0, "reason": "..."}}"""
            try:
                response, _ = verifier_client.chat([{"role": "user", "content": prompt}])
                start, end = response.find("{"), response.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(response[start:end])
                    return {
                        "supported": bool(parsed.get("supported", False)),
                        "confidence": float(parsed.get("confidence", 0.0)),
                        "reason": str(parsed.get("reason", "")),
                    }
            except Exception as exc:
                return {"supported": False, "confidence": 0.0, "reason": f"verify_error: {exc}"}
            return {"supported": False, "confidence": 0.0, "reason": "verify_parse_failed"}

        vqa = VerifiedQA(
            retriever=retriever,
            generator_fn=generate_answer,
            verifier_fn=verifier,
            max_retries=1,
        )

    results: list[dict[str, Any]] = []
    try:
        for question_id, question in enumerate(questions, 1):
            query = question["question"]
            doc_filter = resolve_doc_filter(question.get("source_document"))
            started = time.perf_counter()
            if version == "V4" and not retrieval_only:
                state = vqa.run(query, doc_filter=doc_filter)
                chunks = state.get("retrieved_chunks", [])
                generated = {
                    "answer": state.get("answer", ""),
                    "citations": state.get("citations", []),
                }
                verification = state.get("verification_result", {})
                final_status = state.get("final_status", "")
                retry_count = state.get("retry_count", 0)
                trace = state.get("trace", [])
            elif version == "V2":
                chunks = retriever.search(query, top_k=TOP_K, mode="hybrid", doc_filter=doc_filter)
                verification, final_status, retry_count, trace = {}, "", 0, []
                generated = (generate_answer(query, chunks) if not retrieval_only
                             else {"answer": ""})
            elif version == "V3":
                chunks = retriever.search(query, top_k=TOP_K, mode="reranked", doc_filter=doc_filter)
                verification, final_status, retry_count, trace = {}, "", 0, []
                generated = (generate_answer(query, chunks) if not retrieval_only
                             else {"answer": ""})
            else:
                chunks = retriever.search(query, top_k=TOP_K, doc_filter=doc_filter)
                verification, final_status, retry_count, trace = {}, "", 0, []
                generated = (generate_answer(query, chunks) if not retrieval_only
                             else {"answer": ""})
            total_latency = time.perf_counter() - started
            retrieval_latency = total_latency
            generation_latency = total_latency
            pages = [int(c.get("page_number", 0)) for c in chunks]
            record = {
                "question_id": question_id,
                "question": query,
                "source_document": question.get("source_document", ""),
                "gold_pages": question.get("gold_pages", []),
                "retrieved_pages": pages,
                "retrieved_sources": [c.get("source_file", "") for c in chunks],
                "retrieved_contexts": [c.get("content", "") for c in chunks],
                "answer": generated.get("answer", ""),
                "citations": generated.get("citations", []),
                "content_types": [c.get("content_type", "text") for c in chunks],
                "retrieval_latency_s": round(retrieval_latency, 4),
                "generation_latency_s": round(generation_latency, 4),
            }
            if version == "V4" and not retrieval_only:
                record.update({
                    "verification": verification,
                    "final_status": final_status,
                    "retry_count": retry_count,
                    "trace": trace,
                })
            record.update(metric_values(record["gold_pages"], pages))
            results.append(record)
            if question_id % 10 == 0:
                print(f"{version}: {question_id}/{len(questions)}")
    finally:
        retriever.close()
    return results


def evaluate_ragas(
    version: str,
    version_results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run official RAGAS metrics for all 100 records.

    Answer relevancy uses the project's explicit custom LLM judge because the
    configured endpoint does not expose an embeddings API compatible with the
    RAGAS implementation. It is still evaluated for all 100 questions and is
    labeled as custom in the manifest.
    """
    from datasets import Dataset
    from openai import OpenAI
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import context_precision, context_recall, faithfulness
    from src.config.settings import settings
    from src.eval.metrics import answer_relevancy

    ragas_llm = llm_factory(
        settings.llm_model,
        client=OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60.0,
            max_retries=0,
        ),
    )
    from ragas.run_config import RunConfig
    run_config = RunConfig(timeout=60, max_retries=1, max_wait=10, max_workers=2)
    batch_dir = RUN_DIR / "ragas_batches" / version.lower()
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 5
    row_map: dict[int, dict[str, Any]] = {}
    total_started = time.perf_counter()

    for start in range(0, len(version_results), batch_size):
        end = min(start + batch_size, len(version_results))
        batch_path = batch_dir / f"batch_{start + 1:03d}_{end:03d}.json"
        if batch_path.exists():
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            for row in batch["rows"]:
                for key in ("faithfulness", "context_precision", "context_recall", "answer_relevancy"):
                    row[key] = finite_or_none(row.get(key))
                row_map[int(row["question_id"])] = row
            print(f"{version} RAGAS: {end}/100 (checkpoint)", flush=True)
            continue

        batch_results = version_results[start:end]
        data = {
            "question": [r["question"] for r in batch_results],
            "answer": [r["answer"] for r in batch_results],
            "contexts": [r["retrieved_contexts"] or ["(no retrieved context)"] for r in batch_results],
            "ground_truth": [questions[start + i].get("reference_answer", "") for i in range(len(batch_results))],
        }
        print(f"{version} RAGAS: {start + 1}-{end}/100", flush=True)
        result = evaluate(
            Dataset.from_dict(data),
            metrics=[faithfulness, context_precision, context_recall],
            llm=ragas_llm,
            batch_size=batch_size,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=False,
        )
        frame_rows = result.to_pandas().to_dict("records")
        relevancy_values = [answer_relevancy(r["question"], r["answer"]) for r in batch_results]
        rows = []
        for record, row, relevancy_value in zip(batch_results, frame_rows, relevancy_values):
            rows.append({
                "question_id": record["question_id"],
                "faithfulness": finite_or_none(row.get("faithfulness")),
                "context_precision": finite_or_none(row.get("context_precision")),
                "context_recall": finite_or_none(row.get("context_recall")),
                "answer_relevancy": finite_or_none(relevancy_value),
            })
        batch_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        for row in rows:
            row_map[int(row["question_id"])] = row

    if len(row_map) != len(version_results):
        raise RuntimeError(f"{version} RAGAS incomplete: {len(row_map)}/{len(version_results)}")

    rows = [row_map[r["question_id"]] for r in version_results]
    def mean(key: str) -> float:
        values = [float(r[key]) for r in rows if finite_or_none(r.get(key)) is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    elapsed = time.perf_counter() - total_started
    metrics = {
        "evaluated_question_count": len(version_results),
        "faithfulness": mean("faithfulness"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "answer_relevancy": mean("answer_relevancy"),
        "avg_ragas_latency_s": round(elapsed / len(version_results), 4),
        "valid_sample_counts": {
            key: sum(1 for row in rows if row.get(key) is not None)
            for key in ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
        },
        "engine": {
            "faithfulness": "ragas==0.4.3",
            "context_precision": "ragas==0.4.3",
            "context_recall": "ragas==0.4.3",
            "answer_relevancy": "custom_llm_judge (embedding API unavailable)",
        },
    }
    for record, row in zip(version_results, rows):
        record["faithfulness"] = row.get("faithfulness")
        record["context_precision"] = row.get("context_precision")
        record["context_recall"] = row.get("context_recall")
        record["answer_relevancy"] = row.get("answer_relevancy")
    return metrics


def v5_metrics() -> dict[str, Any]:
    # Prefer the v8 multi-doc report (does not overwrite the original V5 report)
    candidates = [
        PROJECT_ROOT / "storage" / "runs" / "v8_multidoc" / "update_report.json",
        PROJECT_ROOT / "storage" / "runs" / "v5_incremental" / "update_report.json",
        ARCHIVE_DIR / "v5_incremental" / "update_report.json",
    ]
    for report in candidates:
        if report.exists():
            data = json.loads(report.read_text(encoding="utf-8"))
            counts = data.get("counts", data)
            keys = ["added_count", "unchanged_count", "modified_count", "deleted_count",
                    "reprocessed_pages", "reused_chunks", "embedded_chunks", "removed_chunks"]
            return {key: int(counts.get(key, counts.get(key.replace("_count", ""), 0))) for key in keys}
    raise RuntimeError("V5 update_report.json not found")


def archive_old_runs() -> None:
    if ARCHIVE_DIR.exists():
        raise RuntimeError(f"Archive already exists: {ARCHIVE_DIR}; refusing to overwrite it")
    ARCHIVE_DIR.mkdir(parents=True)
    runs = PROJECT_ROOT / "storage" / "runs"
    for item in list(runs.iterdir()):
        if item.name in {"archive_before_final_eval", "final_eval", ".gitkeep"}:
            continue
        shutil.move(str(item), str(ARCHIVE_DIR / item.name))


def write_results(version: str, results: list[dict[str, Any]]) -> None:
    (RUN_DIR / f"{version.lower()}_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Final evaluation (multi-doc aware)")
    parser.add_argument("--dataset", default="golden_100.json",
                        help="dataset filename in data/eval_dataset")
    parser.add_argument("--run-dir", default="storage/runs/final_eval",
                        help="output run directory (use a new dir for a new dataset)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip generation + RAGAS (fast retrieval metrics only)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="limit to first N questions (quick smoke test)")
    args = parser.parse_args()
    global DATASET_PATH, RUN_DIR
    DATASET_PATH = PROJECT_ROOT / "data" / "eval_dataset" / args.dataset
    RUN_DIR = PROJECT_ROOT / args.run_dir

    questions = load_questions()
    if args.max_questions:
        questions = questions[: args.max_questions]
    total = len(questions)
    modality_counts = dict(Counter(q.get("modality_required", "text") for q in questions))
    source_docs = sorted({q.get("source_document", "") for q in questions if q.get("source_document")})
    if not (RUN_DIR / "final_metrics.json").exists() and not ARCHIVE_DIR.exists():
        archive_old_runs()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": f"final_eval_{time.strftime('%Y%m%d_%H%M%S')}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "source_documents": source_docs,
        "total_questions": total,
        "modality_counts": modality_counts,
        "ragas_version": "0.4.3",
        "versions": {},
    }
    (RUN_DIR / "final_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for version in ["V0", "V1", "V2", "V3", "V4"]:
        output = RUN_DIR / f"{version.lower()}_results.json"
        if output.exists():
            print(f"{version}: checkpoint exists, loading")
            results = json.loads(output.read_text(encoding="utf-8"))
        else:
            results = retrieve_and_generate(version, questions, retrieval_only=args.retrieval_only)
            write_results(version, results)
        if len(results) != total:
            raise RuntimeError(f"{version} incomplete: {len(results)}/{total}")
        if args.retrieval_only:
            ragas = {}
        else:
            print(f"{version}: running RAGAS on {total} questions", flush=True)
            ragas = evaluate_ragas(version, results, questions)
        retrieval = summarize(results)
        summary = {"version": version, "retrieval_metrics": retrieval, "ragas_metrics": ragas}
        manifest["versions"][version] = summary
        (RUN_DIR / "final_metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["v5_incremental_metrics"] = v5_metrics()
    (RUN_DIR / "final_metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Final Evaluation",
        "",
        f"Evaluation Dataset: {DATASET_PATH.name}  ",
        f"Total Questions: {total}  ",
        f"Modality Counts: {modality_counts}  ",
        f"Source Documents: {source_docs}  ",
        "All V0-V4 use the same dataset and evaluation protocol.",
        "",
        "## Retrieval Metrics",
        "| Version | Hit@5 | Recall@5 | MRR | Top-1 Hit | Avg Retrieval Latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for version in ["V0", "V1", "V2", "V3", "V4"]:
        m = manifest["versions"][version]["retrieval_metrics"]
        lines.append(f"| {version} | {m['hit_at_5']:.4f} | {m['recall_at_5']:.4f} | {m['mrr']:.4f} | {m['top1_hit_rate']:.4f} | {m['avg_retrieval_latency_s']:.4f}s |")
    if not args.retrieval_only:
        lines += [
            "", "## RAGAS / Answer Metrics",
            "| Version | Faithfulness | Context Precision | Context Recall | Answer Relevancy | Avg Generation Latency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for version in ["V0", "V1", "V2", "V3", "V4"]:
            m = manifest["versions"][version]["ragas_metrics"]
            g = manifest["versions"][version]["retrieval_metrics"]
            lines.append(f"| {version} | {m['faithfulness']:.4f} | {m['context_precision']:.4f} | {m['context_recall']:.4f} | {m['answer_relevancy']:.4f} | {g['avg_generation_latency_s']:.4f}s |")
    lines += ["", "## V5 Incremental Metrics", "| Metric | Value |", "|---|---:|"]
    for key, value in manifest["v5_incremental_metrics"].items():
        lines.append(f"| {key} | {value} |")
    (RUN_DIR / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Final evaluation written to {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
