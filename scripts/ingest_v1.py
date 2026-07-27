#!/usr/bin/env python
"""Phase 3 — V1 Multimodal Ingestion.

Extracts tables and renders page images, generates VLM semantic descriptions,
creates multimodal Chunks, embeds with BGE-M3, stores in V1 Collection.

Usage: python scripts/ingest_v1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ⚠️ pymilvus workaround
import os as _os
from dotenv import dotenv_values as _dv
_os.environ["MILVUS_URI"] = "http://localhost:19530"
_ENV = _dv(str(PROJECT_ROOT / ".env"))
_REAL_MILVUS = _ENV.get("MILVUS_URI", "milvus.db")


def _render_page(pdf, page_num: int, out_dir: Path) -> Path:
    """Render a PDF page as a PNG image."""
    import fitz
    page = pdf[page_num - 1]  # 0-indexed
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for readability
    pix = page.get_pixmap(matrix=mat)
    out_path = out_dir / f"page_{page_num:02d}.png"
    pix.save(str(out_path))
    return out_path


def _describe_image(vlm, image_path: Path, page_num: int) -> str:
    """Use VLM to generate a semantic description of a page image."""
    prompt = (
        "请详细描述这张说明书页面中的图表、结构示意图和关键标注内容。"
        "包括：标注的部件名称、箭头指示的方向、表格内容、以及图示传达的操作信息。"
        "请用中文描述，保留所有技术术语和数字。"
    )
    try:
        result, _ = vlm.chat_with_image(str(image_path), prompt)
        return result.strip()
    except Exception as e:
        return f"[VLM description failed: {e}]"


def _extract_tables(pdf, page_num: int) -> list[str]:
    """Extract tables from a PDF page as structured text."""
    page = pdf[page_num - 1]
    tables = page.find_tables()
    if not tables or not tables.tables:
        return []
    results = []
    for t in tables.tables:
        rows = []
        for row in t.extract():
            rows.append(" | ".join(str(cell).strip() for cell in row if cell))
        results.append("\n".join(rows))
    return results


def main() -> None:
    import fitz

    t0 = time.perf_counter()
    pdf_path = list((PROJECT_ROOT / "data" / "raw_docs").glob("*.pdf"))[0]
    print(f"PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # ── 1. Load VLM ──
    from src.infra.vlm_client import VLMClient
    vlm = VLMClient()
    print(f"VLM: {vlm.model}")

    # ── 2. Render pages & describe ──
    img_dir = PROJECT_ROOT / "data" / "processed" / "v1_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    from src.ingestion.document import Chunk, _make_document_id

    doc_id = _make_document_id(str(pdf_path))
    doc_version = _dv(str(PROJECT_ROOT / ".env")).get("DOC_VERSION", "v1")

    multimodal_chunks: list[Chunk] = []
    img_count = 0
    table_count = 0

    # ── 2a. Image descriptions (diagram-heavy pages) ──
    DIAGRAM_PAGES = {5, 6, 7, 8, 10, 23}  # pages with significant diagrams
    print(f"\nRendering & describing {len(DIAGRAM_PAGES)} diagram pages...")
    for pn in DIAGRAM_PAGES:
        if pn > total_pages:
            continue
        try:
            img_path = _render_page(doc, pn, img_dir)
            description = _describe_image(vlm, img_path, pn)
            img_count += 1

            chunk = Chunk(
                document_id=doc_id,
                document_version=doc_version,
                page_number=pn,
                content=f"[图片语义描述 - 第{pn}页]\n{description}",
                content_type="image",
                source_file=str(pdf_path),
                image_path=str(img_path),
                seq=len(multimodal_chunks),
            )
            multimodal_chunks.append(chunk)
            print(f"  Page {pn}: described ({len(description)} chars)")
        except Exception as e:
            print(f"  Page {pn}: ERROR - {e}")

    # ── 2b. Tables ──
    print(f"\nExtracting tables...")
    for pn in range(1, total_pages + 1):
        tables = _extract_tables(doc, pn)
        for t_text in tables:
            if len(t_text.strip()) < 20:
                continue
            table_count += 1
            chunk = Chunk(
                document_id=doc_id,
                document_version=doc_version,
                page_number=pn,
                content=f"[表格 - 第{pn}页]\n{t_text}",
                content_type="table",
                source_file=str(pdf_path),
                seq=len(multimodal_chunks),
            )
            multimodal_chunks.append(chunk)
            print(f"  Page {pn}: table extracted ({len(t_text)} chars)")

    doc.close()

    # ── 3. Also include original text chunks ──
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.chunker import chunk_document
    text_doc = parse_pdf(str(pdf_path))
    text_chunks = chunk_document(text_doc, chunk_size=500, overlap=50)
    # Update content_type to text for clarity
    for c in text_chunks:
        c.content_type = "text"
    all_chunks = text_chunks + multimodal_chunks

    print(f"\nChunk summary:")
    print(f"  Text chunks:  {len(text_chunks)}")
    print(f"  Image chunks: {img_count}")
    print(f"  Table chunks: {table_count}")
    print(f"  Total:        {len(all_chunks)}")

    # ── 4. Embed ──
    from src.infra.embedder import Embedder
    print("\nLoading BGE-M3...")
    embedder = Embedder()
    embedder.load()
    print(f"  Device: {embedder.device}, Dim: {embedder.dim}")

    texts = [c.content for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embedder.encode_batch(texts)
    print(f"  Done: {len(vectors)} vectors")

    # ── 5. Store in V1 Collection ──
    milvus_path = _REAL_MILVUS if _REAL_MILVUS.startswith("http") else str(PROJECT_ROOT / _REAL_MILVUS)
    from pymilvus import MilvusClient
    client = MilvusClient(milvus_path)

    COLLECTION = f"v1_multimodal_{time.strftime('%Y%m%d_%H%M%S')}"
    # Don't try drop — just create with unique name to avoid Milvus Lite Windows issue
    client.create_collection(
        collection_name=COLLECTION,
        dimension=embedder.dim,
        metric_type="COSINE",
        auto_id=True,
    )

    data = []
    for chunk, vec in zip(all_chunks, vectors):
        data.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "page_number": chunk.page_number,
            "content": chunk.content,
            "content_type": chunk.content_type,
            "source_file": chunk.source_file,
            "image_path": chunk.image_path or "",
            "vector": vec,
        })

    res = client.insert(collection_name=COLLECTION, data=data)
    client.load_collection(COLLECTION)
    print(f"\nV1 Collection '{COLLECTION}': {res['insert_count']} rows inserted")

    # ── 6. Save metadata ──
    elapsed = time.perf_counter() - t0
    meta = {
        "experiment": "v1_multimodal",
        "version": "V1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "text_chunks": len(text_chunks),
        "image_chunks": img_count,
        "table_chunks": table_count,
        "total_chunks": len(all_chunks),
        "vlm_model": vlm.model,
        "embedding_model": embedder.model_name,
        "collection": COLLECTION,
        "diagram_pages_rendered": sorted(DIAGRAM_PAGES),
        "duration_seconds": round(elapsed, 1),
    }
    meta_path = PROJECT_ROOT / "storage" / "runs" / "v1_ingest" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    client.close()
    print(f"\nV1 Ingestion complete: {elapsed:.1f}s")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
