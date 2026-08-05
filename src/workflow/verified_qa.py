"""LangGraph verified QA workflow.

Flow:
  Question → Retrieve → CheckRelevance
    → irrelevant: refuse (fallback)
    → relevant: Generate → Verify → Decide
        → answered: return
        → retry: back to Retrieve (max 1)
        → refused: return

final_status values:
  - answered: verified and supported
  - refused: could not answer (no evidence / out of scope / verify failed)
  - fallback: degraded answer (retriever/reranker unavailable)
  - retried: internal; should not appear in final output
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END


class QAState(TypedDict):
    question: str
    retrieved_chunks: list[dict]
    answer: str
    citations: list[dict]
    verification_result: dict
    retry_count: int
    final_status: str  # answered | refused | fallback
    trace: list[str]
    doc_filter: str | None  # source_file to restrict retrieval to one doc


# Default thresholds
MIN_RELEVANCE_SCORE = 0.05  # rerank_score below this → likely out of scope


def _check_question_relevance(chunks: list[dict], threshold: float = MIN_RELEVANCE_SCORE) -> bool:
    """Determine if any retrieved chunk is relevant to the question.

    Checks rerank_score (or retrieval_score as fallback) against threshold.
    """
    if not chunks:
        return False
    best = max(
        c.get("rerank_score") or c.get("retrieval_score") or 0.0
        for c in chunks
    )
    return best >= threshold


class VerifiedQA:
    """LangGraph-based QA with verification, retry, and scope checking."""

    def __init__(
        self,
        retriever,
        generator_fn,
        verifier_fn,
        max_retries: int = 1,
        relevance_threshold: float = MIN_RELEVANCE_SCORE,
    ):
        self.retriever = retriever
        self.generator_fn = generator_fn
        self.verifier_fn = verifier_fn
        self.max_retries = max_retries
        self.relevance_threshold = relevance_threshold
        self._graph = self._build_graph()

    # ── Nodes ──

    def _retrieve(self, state: QAState) -> dict:
        new_trace = list(state.get("trace", [])) + ["retrieve"]
        chunks = self.retriever.search(state["question"], top_k=5, mode="reranked",
                                       doc_filter=state.get("doc_filter"))
        return {
            "trace": new_trace,
            "retrieved_chunks": chunks,
            "citations": [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "source_file": c.get("source_file", ""),
                    "page_number": c.get("page_number", 0),
                    "content_type": c.get("content_type", "text"),
                }
                for c in chunks
            ],
        }

    def _check_relevance(self, state: QAState) -> dict:
        new_trace = list(state.get("trace", [])) + ["check_relevance"]
        chunks = state["retrieved_chunks"]
        relevant = _check_question_relevance(chunks, self.relevance_threshold)
        if not relevant:
            return {
                "trace": new_trace,
                "final_status": "refused",
                "answer": "根据现有说明书内容无法回答此问题。",
                "verification_result": {
                    "supported": False,
                    "confidence": 0.0,
                    "unsupported_claims": ["问题与说明书内容无关"],
                    "evidence_chunk_ids": [],
                    "reason": "所有检索结果相关度低于阈值，问题超出说明书范围",
                },
            }
        return {"trace": new_trace}

    def _generate(self, state: QAState) -> dict:
        new_trace = list(state.get("trace", [])) + ["generate"]
        result = self.generator_fn(
            state["question"], state["retrieved_chunks"]
        )
        return {"trace": new_trace, "answer": result["answer"]}

    def _verify(self, state: QAState) -> dict:
        new_trace = list(state.get("trace", [])) + ["verify"]
        # Pre-check: if generator itself refused, skip verification
        answer = state["answer"]
        refusal_phrases = ["根据现有说明书内容无法回答", "无法回答此问题"]
        if any(phrase in answer for phrase in refusal_phrases):
            return {
                "trace": new_trace,
                "verification_result": {
                    "supported": False,
                    "confidence": 0.0,
                    "unsupported_claims": ["生成器自行拒答"],
                    "evidence_chunk_ids": [],
                    "reason": "generator self-refused",
                },
            }
        result = self.verifier_fn(
            state["question"], state["answer"], state["retrieved_chunks"],
        )
        return {"trace": new_trace, "verification_result": result}

    def _decide(self, state: QAState) -> dict:
        new_trace = list(state.get("trace", [])) + ["decide"]
        supported = state["verification_result"].get("supported", False)
        retries = state["retry_count"]
        if supported:
            return {"trace": new_trace, "final_status": "answered"}
        elif retries < self.max_retries:
            return {
                "trace": new_trace,
                "retry_count": retries + 1,
            }
        else:
            return {
                "trace": new_trace,
                "final_status": "refused",
                "answer": "根据现有说明书内容无法回答此问题。",
            }

    @staticmethod
    def _route_after_check(state: QAState) -> str:
        if state.get("final_status") == "refused":
            return "refused"
        return "relevant"

    @staticmethod
    def _route_after_decide(state: QAState) -> str:
        status = state.get("final_status", "")
        if status == "answered":
            return "answered"
        elif status == "" and state.get("retry_count", 0) > 0:
            return "retry"  # no status set → retry path
        return "refused"

    # ── Build ──

    def _build_graph(self):
        builder = StateGraph(QAState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("check_relevance", self._check_relevance)
        builder.add_node("generate", self._generate)
        builder.add_node("verify", self._verify)
        builder.add_node("decide", self._decide)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "check_relevance")
        builder.add_conditional_edges(
            "check_relevance",
            self._route_after_check,
            {"relevant": "generate", "refused": END},
        )
        builder.add_edge("generate", "verify")
        builder.add_edge("verify", "decide")
        builder.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {"answered": END, "retry": "retrieve", "refused": END},
        )
        return builder.compile()

    def run(self, question: str, doc_filter: str | None = None) -> QAState:
        initial: QAState = {
            "question": question,
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "verification_result": {},
            "retry_count": 0,
            "final_status": "",
            "trace": [],
            "doc_filter": doc_filter,
        }
        return self._graph.invoke(initial)
