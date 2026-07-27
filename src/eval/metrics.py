"""LLM-as-Judge evaluation metrics for RAG quality.

Implements:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are retrieved contexts relevant to the question?
- Context Recall: Are all key facts from reference in the retrieved contexts?
"""

from __future__ import annotations

from src.infra.llm_client import LLMClient


def _ask_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return the response."""
    client = LLMClient()
    response, _ = client.chat([{"role": "user", "content": prompt}])
    return response.strip()


def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Score 0-1: how factually grounded is the answer in the contexts?

    Decomposes the answer into claims and checks each against contexts.
    """
    if not answer or not contexts:
        return 0.0

    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    prompt = f"""Evaluate whether the ANSWER is faithful to the CONTEXTS.

CONTEXTS:
{context_text}

ANSWER:
{answer}

Task:
1. Break the answer into individual factual claims.
2. For each claim, determine if it is supported by the contexts.
3. Count the number of supported claims and total claims.

Output ONLY a JSON object:
{{"supported": <int>, "total": <int>, "score": <float>}}"""

    try:
        response = _ask_llm(prompt)
        import json
        # Extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            return max(0.0, min(1.0, data.get("score", 0.0)))
    except Exception:
        pass
    return 0.0


def answer_relevancy(question: str, answer: str) -> float:
    """Score 0-1: how relevant is the answer to the question?"""
    if not answer:
        return 0.0

    prompt = f"""Evaluate whether the ANSWER is relevant to the QUESTION.

QUESTION: {question}

ANSWER: {answer}

Score from 0 to 1 where:
- 1.0: Answer fully addresses the question
- 0.5: Answer partially addresses the question
- 0.0: Answer is completely irrelevant or refuses to answer

Output ONLY a number (0.0 to 1.0):"""

    try:
        response = _ask_llm(prompt)
        # Extract numeric score
        for word in response.split():
            try:
                score = float(word.strip(".,;:"))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass
    except Exception:
        pass
    return 0.0


def context_precision(question: str, contexts: list[str]) -> float:
    """Score 0-1: what fraction of retrieved contexts are relevant?"""
    if not contexts:
        return 0.0

    context_text = "\n\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))

    prompt = f"""Evaluate whether each CONTEXT is relevant to the QUESTION.

QUESTION: {question}

CONTEXTS:
{context_text}

For each context, determine if it is relevant (1) or not (0).
Output ONLY a JSON list of 0s and 1s, one per context:
[0, 1, 0, 1, 1]"""

    try:
        response = _ask_llm(prompt)
        import json
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            scores = json.loads(response[start:end])
            relevant = sum(1 for s in scores if s == 1)
            return relevant / len(scores) if scores else 0.0
    except Exception:
        pass
    return 0.0


def context_recall(
    question: str, contexts: list[str], reference_contexts: list[str]
) -> float:
    """Score 0-1: what fraction of key reference facts are in the contexts?"""
    if not reference_contexts:
        return 0.0

    context_text = "\n\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    ref_text = "\n".join(reference_contexts)

    prompt = f"""Evaluate how much of the REFERENCE information is covered by the CONTEXTS.

QUESTION: {question}

REFERENCE (ground truth facts):
{ref_text}

CONTEXTS (retrieved):
{context_text}

What fraction (0.0-1.0) of the key facts in the REFERENCE appear in the CONTEXTS?
Output ONLY a number (0.0 to 1.0):"""

    try:
        response = _ask_llm(prompt)
        for word in response.split():
            try:
                score = float(word.strip(".,;:"))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass
    except Exception:
        pass
    return 0.0
