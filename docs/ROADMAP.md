# Roadmap

预计周期为 2–3 周，约 12–15 个工作日。每个阶段完成后必须经过验收。

| 阶段 | 目标 | 状态 |
|---|---|---|
| Phase 0 | 治理、依赖、模型和 Milvus 连通性 | PASS |
| Phase 1 | 文本 Naive RAG | PASS |
| Phase 2 | RAGAS 评测闭环 | PASS |
| Phase 3 | 多模态解析 | PASS |
| Phase 4 | Hybrid Retrieval | PASS |
| Phase 5 | BGE-Reranker | PASS |
| Phase 6 | LangGraph Verify | PASS |
| Phase 7 | 增量更新 | CONDITIONAL_PASS |
| Phase 8 | FastAPI、报告和展示 | PASS | 5 端点真实验证通过 |
| Phase 9A–9D | React 前端、问答、知识库、实验评估和视觉改造 | PASS | lint/build 通过，功能页已接入后端 |
| Phase 10 / V6 | 句级确定性接地验证 | PASS | 交叉编码器接地、投毒测试和拒答链路 |
| Phase 11 / V7 | LLM Gateway 容错 | PASS | 超时、重试、熔断、Provider 兜底 |
| Phase 12 / V8 | 多文档与扩展检索评测 | PASS | 123 题 retrieval-only；第二本说明书和 doc_filter |
| Phase 13 / V9 | 查询语义缓存 | PASS | SHA256 + BGE-M3 语义缓存，SQLite 持久化 |

## 当前结果边界

- 100 题正式 RAGAS 结果仍保存在 `storage/runs/final_eval/`。
- 123 题扩展评测保存在 `storage/runs/final_eval_extended_full/`，当前仅为检索评测，不包含 RAGAS。
- Phase 7 增量更新保留 `CONDITIONAL_PASS` 的技术限制：修改文档按文档级重新处理。

## 阶段推进规则

- `IN_PROGRESS` 阶段只能有一个。
- 未达到 `PASS` 不得进入下一阶段。
- 发现跨阶段需求时，先记录到 `DECISIONS.md`，不得临时扩大范围。
- 每次阶段结束都必须更新 `PROGRESS.md` 和 `SESSION_LOG.md`。
