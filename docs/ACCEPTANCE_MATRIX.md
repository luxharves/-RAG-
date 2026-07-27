# Acceptance Matrix

## 判定结果

- `PASS`：核心验收项全部满足。
- `CONDITIONAL_PASS`：主流程可用，但必须先完成指定修复。
- `FAIL`：核心功能、测试或数据契约不满足。

## 阶段验收

| 阶段 | 核心验收证据 |
|---|---|
| Phase 0 | GPU Embedding、Reranker、Milvus、LLM、VLM Smoke Test 结果 |
| Phase 1 | PDF 入库、Top-5 命中率、引用答案、测试结果 |
| Phase 2 | 100 条 QA、RAGAS 指标、Recall@K、MRR、不可覆盖的 V0 结果 |
| Phase 3 | 图片/表格 Chunk、来源信息、多模态问题检索结果 |
| Phase 4 | BM25、Dense、RRF 融合结果及 V1/V2 对比 |
| Phase 5 | Rerank 前后顺序、分数和 V2/V3 对比 |
| Phase 6 | Graph 状态、Verify、重试、拒答和轨迹日志 |
| Phase 7 | Hash、增量处理数量、新旧版本和删除验证 |
| Phase 8 | FastAPI、Swagger、实验报告、README 和演示流程 |

## 通用完成条件

- 有实际运行命令和输出。
- 有测试结果，而不是只有代码说明。
- 有关键日志或生成文件。
- 有修改文件列表。
- 有已知问题和风险说明。

