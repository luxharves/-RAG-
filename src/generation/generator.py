"""LLM answer generator with citation support."""

from __future__ import annotations

from src.infra.llm_client import LLMClient


SYSTEM_PROMPT = """你是一个智能硬件维保助手。请根据提供的说明书内容回答用户问题。

规则：
1. 只能根据提供的上下文回答，不要使用你自己的知识。
2. 如果上下文中没有相关信息，请明确说"根据现有说明书内容无法回答此问题"，不要编造。
3. 回答时请引用来源，格式为 [来源: 文件名, 第X页]。
4. 回答应简洁、准确、有帮助。
5. 如果多个来源提供相同信息，请合并引用。
"""


def _build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source_file", "unknown")
        page = c.get("page_number", "?")
        content = c.get("content", "")
        parts.append(f"[{i}] 来源: {source}, 第{page}页\n{content}")
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    temperature: float = 0.0,
) -> dict:
    """Generate an answer with citations from retrieved chunks.

    Returns:
        {answer, chunks_used, citations, model, usage}
    """
    if not retrieved_chunks:
        return {
            "answer": "根据现有说明书内容无法回答此问题。",
            "chunks_used": 0,
            "citations": [],
            "model": None,
            "usage": None,
        }

    context = _build_context(retrieved_chunks)
    user_message = f"上下文：\n\n{context}\n\n问题：{question}"

    client = LLMClient()
    response_text, raw = client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    # Extract citations from chunks used
    citations: list[dict] = []
    for c in retrieved_chunks:
        citations.append({
            "chunk_id": c.get("chunk_id", ""),
            "source_file": c.get("source_file", ""),
            "page_number": c.get("page_number", 0),
            "retrieval_score": c.get("retrieval_score"),
        })

    return {
        "answer": response_text,
        "chunks_used": len(retrieved_chunks),
        "citations": citations,
        "model": raw.get("model"),
        "usage": raw.get("usage"),
    }
