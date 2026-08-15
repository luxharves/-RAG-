# Experiment Protocol

## 1. 控制变量

各实验必须固定：

- 评测问题集合
- 文档集合
- Chunk 配置
- LLM/VLM 模型
- Embedding 模型
- Temperature
- 最终返回 Top-K
- 评测提示词

## 2. 实验矩阵

| 版本 | Multimodal | Hybrid | Reranker | Verify |
|---|---:|---:|---:|---:|
| V0 | 0 | 0 | 0 | 0 |
| V1 | 1 | 0 | 0 | 0 |
| V2 | 1 | 1 | 0 | 0 |
| V3 | 1 | 1 | 1 | 0 |
| V4 | 1 | 1 | 1 | 1 |

> V5 = + SHA256 增量更新（非特征维）；V6 = + 确定性句级接地验证（BGE-Reranker 交叉编码器，非特征维）。V7–V9 是运行时和数据能力扩展，不纳入原始 V0–V4 检索特征增益表。

## 3. 扩展能力

| 版本 | 新增能力 | 评测方式 |
|---|---|---|
| V7 | LLM 超时、重试、熔断、Provider 兜底 | 故障注入和单元测试 |
| V8 | 多说明书、文档过滤、跨语言和图片问题 | `golden_extended.json`，123 题 retrieval-only |
| V9 | 精确/语义查询缓存 | 缓存命中率、命中延迟、LLM 调用节省 |

### 数据集使用规则

- `golden_100.json` 固定为 100 题，用于正式 V0–V4 检索 + RAGAS 对比，结果位于 `storage/runs/final_eval/`。
- `golden_extended.json` 固定为 123 题，用于 V8 多文档、多模态检索扩展，结果位于 `storage/runs/final_eval_extended_full/`。
- 扩展集当前只保存检索指标，不能将其描述为 123 题 RAGAS 评测。
- 历史结果不得覆盖；新增实验必须使用新的 run 目录。

## 4. 必测指标

- Recall@K
- MRR
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy
- 平均延迟
- Embedding 数量和增量更新成本

## 5. 结果保存

结果保存到 `storage/runs/<timestamp>-<experiment-name>/`，禁止覆盖旧实验。
