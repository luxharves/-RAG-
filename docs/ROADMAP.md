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

## 阶段推进规则

- `IN_PROGRESS` 阶段只能有一个。
- 未达到 `PASS` 不得进入下一阶段。
- 发现跨阶段需求时，先记录到 `DECISIONS.md`，不得临时扩大范围。
- 每次阶段结束都必须更新 `PROGRESS.md` 和 `SESSION_LOG.md`。
