#!/usr/bin/env python
"""Fix V1 image chunks: add concise keyword summaries to improve BGE-M3 matching.

Does NOT re-run VLM — reads existing descriptions, prepends keywords, re-embeds.
"""

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

# ── Manual keyword summaries for each diagram page ──
KEYWORDS: dict[int, str] = {
    5: "关键词：机器人正面结构图、电源键、回充键、局部清扫键、童锁、指示灯、传感器位置",
    6: "关键词：机器人底部结构图、悬崖传感器防跌落、地毯识别传感器、万向轮、主轮、边刷、软胶主刷、碰撞缓冲传感器",
    7: "关键词：尘盒结构图、滤网、电控水箱、震动拖布支架、拖布粘贴区域、升降震动擦地模组",
    8: "关键词：基座结构图、污水箱、清水箱、集尘桶、充电弹片、高速自清洁刷、清洗槽滤网、状态指示灯",
    10: "关键词：基座安装步骤图、运输固定泡沫、基座底板组装、电源线连接、基座放置空间要求",
    23: "关键词：电池拆卸步骤图、报废处理、电池盖板螺丝、连接器插头、环保回收",
}

# ── Per-page functional summaries (key safety/usage info) ──
SUMMARIES: dict[int, str] = {
    5: "机器人正面包含：清扫/开关机键（短按清扫、长按开关机）、回充键（短按回充、长按回洗拖布）、局部清扫/童锁键（短按局部清扫、长按3秒童锁）。电源指示灯：白色电量≥20%、红色<20%、呼吸闪烁充电中、红色快闪异常。",
    6: "机器人底部装有悬崖传感器，可检测楼梯、台阶等高差环境，防止机器人跌落。地毯识别传感器自动识别地毯并触发拖布支架升降。底部还有万向轮、主轮、边刷、软胶主刷、集尘进风口。碰撞缓冲传感器位于前端。",
    7: "尘盒包含上置可水洗滤网和滤网上盖。电控水箱通过进水口滤网和自动补水口与基座连接。震动拖布支架不可拆卸，拖布通过插槽和粘贴区域固定。升降震动擦地模组提供擦地动力。",
    8: "基座包含：污水箱（上盖+卡扣）、清水箱（提手+上盖）、集尘桶（内置一次性尘袋）、充电弹片、高速自清洁刷（可拆卸卡扣）、清洗槽滤网。状态指示灯：呼吸闪烁=集尘/清洁拖布中、红色=异常、熄灭=未通电或充电中。",
    10: "基座安装步骤：1)取出底部高速自清洁刷组件运输固定泡沫；2)基座底板与主体组装，下压听到咔哒声；3)电源线插紧，多余线材收入线槽。基座放置：硬质水平地面靠墙，两侧0.5米、前方1.5米、上方1米以上空间。",
    23: "报废前拆电池步骤：1)不接触基座运行至低电量；2)关机；3)卸下电池盖板螺丝；4)取下盖板；5)按下卡扣拔出连接器插头取下电池。注意：确保电量用尽、断开基座、整组拆卸、勿损坏外壳。渗出物接触皮肤用大量清水冲洗并及时就医。",
}


def main() -> None:
    from pymilvus import MilvusClient
    from src.infra.embedder import Embedder

    milvus_path = str(PROJECT_ROOT / _ENV.get("MILVUS_URI", "milvus.db"))
    client = MilvusClient(milvus_path)

    # ── Find latest V1 collection ──
    v1_cols = sorted([c for c in client.list_collections() if c.startswith("v1_multimodal_2")])
    old_col = v1_cols[-1]
    print(f"Source collection: {old_col}")

    # Load V0 text chunks (we need them too)
    client.load_collection("v0_naive_rag")
    client.load_collection(old_col)

    # ── Get all chunks from old V1 ──
    all_chunks = client.query(
        collection_name=old_col,
        filter="chunk_id != \"\"",
        output_fields=["chunk_id", "document_id", "page_number", "content",
                       "content_type", "source_file", "image_path"],
        limit=100,
    )
    print(f"Read {len(all_chunks)} chunks from {old_col}")

    # ── Enhance image chunks with keywords ──
    enhanced = 0
    for chunk in all_chunks:
        ctype = chunk.get("content_type", "")
        if ctype == "image":
            pn = chunk.get("page_number", 0)
            kw = KEYWORDS.get(pn, "")
            summary = SUMMARIES.get(pn, "")
            old_content = chunk.get("content", "")

            # Prepend keyword + summary as a focused prefix
            prefix = f"{kw}\n{summary}\n\n--- 原始VLM描述 ---\n"
            chunk["content"] = prefix + old_content
            enhanced += 1
            print(f"  Enhanced page {pn} chunk ({len(chunk['content'])} chars)")

    print(f"Enhanced {enhanced} image chunks")

    # ── Re-embed ──
    embedder = Embedder()
    embedder.load()
    print(f"Embedder: {embedder.device}, dim={embedder.dim}")

    texts = [c["content"] for c in all_chunks]
    vectors = embedder.encode_batch(texts)

    # ── Insert into new V1 collection ──
    new_col = f"v1_multimodal_kw_{time.strftime('%Y%m%d_%H%M%S')}"
    client.create_collection(
        collection_name=new_col,
        dimension=embedder.dim,
        metric_type="COSINE",
        auto_id=True,
    )

    data = []
    for chunk, vec in zip(all_chunks, vectors):
        data.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": chunk.get("document_id", ""),
            "page_number": chunk.get("page_number", 0),
            "content": chunk["content"],
            "content_type": chunk.get("content_type", "text"),
            "source_file": chunk.get("source_file", ""),
            "image_path": chunk.get("image_path", ""),
            "vector": vec,
        })

    res = client.insert(collection_name=new_col, data=data)
    client.load_collection(new_col)
    print(f"\nNew collection: {new_col} ({res['insert_count']} rows)")

    # ── Verify Q18 ──
    from src.retrieval.retriever import DenseRetriever
    retriever = DenseRetriever(collection_name=new_col)
    results = retriever.search("机器人会不会从楼梯摔下去？", top_k=5)
    print(f"\nQ18: 机器人会不会从楼梯摔下去？")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] type={r.get('content_type', '?')}, page={r.get('page_number', '?')}, "
              f"score={r.get('retrieval_score', 0):.4f}")
        print(f"      {r['content'][:100]}...")

    # ── Verify table ──
    table_results = retriever.search("产品有害物质含量表", top_k=3)
    print(f"\nTable query: 产品有害物质含量表")
    for i, r in enumerate(table_results, 1):
        print(f"  [{i}] type={r.get('content_type', '?')}, page={r.get('page_number', '?')}")

    retriever.close()
    client.close()

    # ── Save new collection name ──
    meta_path = PROJECT_ROOT / "storage" / "runs" / "v1_ingest" / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["collection"] = new_col
        meta["keyword_enhanced"] = True
        meta["enhanced_image_chunks"] = enhanced
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. New collection: {new_col}")


if __name__ == "__main__":
    main()
