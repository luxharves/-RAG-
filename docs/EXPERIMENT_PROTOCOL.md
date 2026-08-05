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

> V5 = + SHA256 增量更新（非特征维）；V6 = + 确定性句级接地验证（BGE-Reranker 交叉编码器，非特征维）。

## 3. 必测指标

- Recall@K
- MRR
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy
- 平均延迟
- Embedding 数量和增量更新成本

## 4. 结果保存

结果保存到 `storage/runs/<timestamp>-<experiment-name>/`，禁止覆盖旧实验。

