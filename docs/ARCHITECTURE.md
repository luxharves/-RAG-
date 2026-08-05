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
- `infra.gateway`：负责 LLM 调用的超时/重试/熔断/多 provider 兜底链（V7）。

## 5. 实验版本

```text
V0 文本 + Dense Retrieval
V1 V0 + Multimodal Parsing
V2 V1 + BM25/RRF Hybrid Retrieval
V3 V2 + BGE-Reranker
V4 V3 + LangGraph Verify
V5 V4 + Incremental Update
V6 V5 + Deterministic Grounding (Cross-Encoder)
```

