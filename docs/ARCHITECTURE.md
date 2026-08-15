# Architecture

## 1. 总体数据流

```text
Raw PDF
  -> Parser/OCR/VLM
  -> Normalized Chunks
  -> BGE-M3 Embedding
  -> Milvus + BM25
  -> Hybrid Fusion
  -> BGE-Reranker
  -> LLM Answer
  -> LangGraph Verify
  -> Cited Answer / Retry / Fallback
```

## 2. 离线链路

离线链路负责解析文档、生成标准 Chunk、计算 Hash、生成 Embedding，并写入向量库和关键词索引。离线链路不得直接生成最终回答。

## 3. 在线链路

在线链路负责问题规范化、召回、重排、回答生成、证据验证和结果返回。在线链路必须返回来源文件、页码和 Chunk ID。

## 4. 模块边界

- `ingestion`：只负责文档到标准 Chunk。
- `embedding`：只负责文本到向量。
- `retrieval`：只负责候选召回和排序。
- `generation`：只负责基于上下文生成答案。
- `workflow`：负责流程编排和风险路由。
- `api`：负责请求校验和响应封装。
- `eval`：负责离线评测，不改变线上回答逻辑。
- `infra.gateway`：负责 LLM 调用的超时、重试、熔断、多 provider 兜底链和统一降级响应（V7）。
- `retrieval`：支持 doc 级 metadata filtering（`doc_filter` 按 source_file 限制检索到单本说明书），多文档场景下避免跨文档页号冲突（V8 数据强化）。
- `infra.semantic_cache`：负责 `/query` 的精确 SHA256 缓存和 BGE-M3 语义缓存，使用 SQLite 持久化（V9）。
- `frontend`：React + Vite + TypeScript 演示端，提供问答、知识库、实验评估和系统状态页面。

## 5. 实验版本

```text
V0 文本 + Dense Retrieval
V1 V0 + Multimodal Parsing
V2 V1 + BM25/RRF Hybrid Retrieval
V3 V2 + BGE-Reranker
V4 V3 + LangGraph Verify
V5 V4 + Incremental Update
V6 V5 + Deterministic Grounding (Cross-Encoder)
V7 V6 + LLM Gateway Resilience
V8 V7 + Multi-document Retrieval and Document Filtering
V9 V8 + Exact/Semantic Query Cache
```

## 6. 评测数据边界

- `golden_100.json`：100 条固定回归集，保存 V0–V4 的完整检索和 RAGAS 结果，是当前正式 RAGAS 对比基线。
- `golden_extended.json`：123 条扩展集，包含第二本说明书和新增图片问题，用于多文档、多模态检索验证；当前已完成 retrieval-only 评测，未运行 123 题 RAGAS。
- 两套数据集和结果目录独立保存，不混合计算指标。

## 7. 运行时风险边界

- Milvus 使用 Milvus Lite，适合单机实验，当前为单进程运行模式。
- 增量更新对未变化文档复用全部 Chunk；文档发生变化时目前按文档级全量重处理。
- 语义缓存当前为进程内查询缓存，知识库更新后应在演示或生产流程中清理缓存，以避免复用旧答案。
