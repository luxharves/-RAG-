"""Phase 8 — FastAPI demo entry point.

Run: uvicorn main:app --host 127.0.0.1 --port 8000

NOTE: Milvus Lite requires SINGLE-PROCESS access.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="多模态 RAG 智能硬件维保助手",
    description="V0–V9 多模态可信问答：Hybrid Retrieval → Reranker → LangGraph Verify → 确定性接地 → 语义缓存",
    version="V9",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
