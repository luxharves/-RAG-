#!/usr/bin/env python
"""V6 poison test — prove the grounding verifier REJECTS hallucinated sentences.

Reads cached answers+chunks from storage/runs/v6_grounding/v6_results.json
(the Phase-1 run) and, for each answered fixed case, appends a fabricated
sentence. The verifier must flag the fabricated sentence as unsupported and
flip the overall answer to refused — demonstrating the rejection path works
on real embeddings (not just FakeEmbedder unit tests).

Usage:
    python scripts/run_v6_poison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os as _os
_os.environ["MILVUS_URI"] = "http://localhost:19530"

POISONS = [
    "本产品由核聚变反应堆提供动力。",
    "用户可通过脑机接口控制本产品。",
]


def main() -> None:
    from src.infra.reranker import Reranker
    from src.workflow.grounding import GroundingVerifier, CrossEncoderScorer

    results = json.loads(
        (PROJECT_ROOT / "storage" / "runs" / "v6_grounding" / "v6_results.json")
        .read_text(encoding="utf-8")
    )
    rr = Reranker()
    rr.load()
    verifier = GroundingVerifier(scorer=CrossEncoderScorer(rr), scorer_floor=0.1)

    report = []
    n_flagged = 0
    n_flipped = 0
    tested = 0
    for r in results:
        if r["label"] != "fixed" or r["final_status"] != "answered":
            continue
        chunks = r.get("retrieved_chunks", [])
        answer = r.get("answer", "")
        real = verifier.verify(r["question"], answer, chunks)
        if not real["supported"]:
            continue  # only poison answers that genuinely pass grounding
        tested += 1
        for poison in POISONS:
            poisoned = answer + " " + poison
            pv = verifier.verify(r["question"], poisoned, chunks)
            # The fabricated sentence should be the one containing the poison text
            hit = next(
                (se for se in pv["sentence_evidence"]
                 if se["status"] != "skipped_short" and poison[:10] in se["clean"]),
                None,
            )
            flagged = bool(hit and not hit["supported"])
            flipped = real["supported"] and not pv["supported"]
            n_flagged += int(flagged)
            n_flipped += int(flipped)
            report.append({
                "question": r["question"],
                "poison": poison,
                "poison_best_similarity": hit["best_similarity"] if hit else None,
                "poison_flagged_unsupported": flagged,
                "real_supported": real["supported"],
                "poisoned_supported": pv["supported"],
                "flipped_to_refused": flipped,
            })

    total = tested * len(POISONS)
    print(f"tested {tested} answered cases x {len(POISONS)} poisons = {total}")
    print(f"poison sentences flagged unsupported: {n_flagged}/{total}")
    print(f"answers flipped to refused: {n_flipped}/{total}")

    out = PROJECT_ROOT / "storage" / "runs" / "v6_grounding" / "poison_test.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
