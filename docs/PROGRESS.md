# Progress

## 当前状态

- 当前阶段：Phase 13 (V9 语义缓存)
- 阶段状态：PASS
- 当前版本：V9 语义缓存 + 多文档多模态 + FastAPI + React 全栈
- 最近完成：/query 两级缓存(精确 SHA256 + 语义余弦);缓存命中 ~31ms(全管线 ~20s+),精确重跑 100% 命中;前端缓存命中徽标
- 已知限制：Milvus Lite 单进程；交叉编码器对"主题相关编造"仍有边界（弗兰肯斯坦局限）；熔断状态仅进程内（uvicorn 重启重置，无 Redis）；Ecovacs 手册三语全量摄入（FR/ES 为噪音）

## V7–V9 当前状态

| 版本 | 能力 | 状态 | 证据 |
|---|---|---|---|
| V7 | LLM Gateway：超时、重试、熔断、Provider 兜底 | PASS | `tests/test_gateway.py`，故障注入 demo |
| V8 | 多文档、doc_filter、扩展图片/跨语言检索 | PASS | `golden_extended.json` 123 题 retrieval-only |
| V9 | 精确/语义查询缓存 | PASS | `storage/runs/v9_cache/cache_eval.json`，专项测试 |

### 评测边界

- 正式 RAGAS 对比：`golden_100.json`，100 题，结果位于 `storage/runs/final_eval/`。
- 多文档扩展检索：`golden_extended.json`，123 题，结果位于 `storage/runs/final_eval_extended_full/`。
- 123 题扩展评测当前未运行 RAGAS，不能与 100 题 RAGAS 指标直接合并。

## Phase 7 — 增量更新

### Hash 和版本策略
- 文件 SHA256 → `file_hash`
- `version = file_hash[:16]`
- `document_id = sha256(source_file)[:12]`（稳定）
- Chunk ID = `sha256(document_id|page|seq)[:16]`（同内容稳定）

### 全量 vs 增量对比

| 指标 | 首次（全量） | 第二次（增量） |
|---|---|---|
| added | 1 | 0 |
| unchanged | 0 | 1 |
| reprocessed_pages | 27 | 0 |
| embedded_chunks | 39 | 0 |
| reused_chunks | 0 | 39 |
| elapsed | 0.4s | 0.1s |

## Phase 3 — V1 多模态解析

| 指标 | 值 |
|---|---|
| PDF 总页数 | 27 |
| 渲染图示页 | 6 (p5/p6/p7/p8/p10/p23) |
| 表格 | 3 (p22×2 + p26×1) |
| Text Chunks | 39 |
| Image Chunks | 6 |
| Table Chunks | 3 |
| V1 总 Chunk | 48 |
| VLM | qwen3-vl-32b-thinking |
| V1 Collection | `v1_multimodal_20260718_202339` |

### V0 vs V1 对比（20 题固定回归集）

| 指标 | V0 | V1 (keyword-enhanced) |
|---|---|---|
| Hit Rate | 18/20 | **19/20** |
| Q18 (image) | MISS | **HIT** (p6 image Chunk rank 1, score 0.61) |
| Q19 (text) | MISS | MISS |

Q18 修复：为 image Chunk 增加关键词摘要前缀，BGE-M3 成功匹配"悬崖传感器防跌落"→"楼梯摔下去"。

## Phase 4 — V2 Hybrid Retrieval (Dense + BM25 + RRF)

| 模式 | Hit Rate | Recall@5 | MRR | Q18 | Q19 |
|---|---|---|---|---|---|
| Dense (V1) | 0.9500 (19/20) | 0.8750 | 0.6683 | HIT | MISS |
| BM25 | 0.7000 (14/20) | 0.6250 | 0.3933 | MISS | MISS |
| **Hybrid (V2)** | **0.9500 (19/20)** | **0.8500** | **0.7683** | **HIT** | **HIT** |

- Q19："开机后机器人不动怎么办"：Dense 和 BM25 均未命中，RRF 融合将 page 24 推入 Top-5。
- MRR 从 Dense 0.67 → Hybrid 0.77（+15%），排名质量显著提升。
- BM25 索引：48 docs，持久化至 `storage/bm25/`。

## Phase 5 — V3 Hybrid + BGE-Reranker-v2-m3

| 指标 | V2 Hybrid | V3 Reranked | Delta |
|---|---|---|---|
| Hit Rate | 0.9500 | 0.9500 | — |
| Top-1 Hit Rate | 0.6500 | **0.7000** | **+5.0%** |
| Recall@5 | 0.8500 | **0.8750** | +2.5% |
| MRR | 0.7683 | **0.7875** | **+1.9%** |
| 排名变化 | — | 20/20 题 | — |
| 平均耗时 | 0.28s | 23.15s | +22.87s |

- Reranker 模型：BAAI/bge-reranker-v2-m3 (GPU)
- 候选 Top-20 → Final Top-5
- 所有 20 题排序均发生了变化，Top-1 命中率从 65% 提升到 70%

## 数据集说明

Golden Dataset（`data/eval_dataset/golden_100.json`）：

- **100 条固定回归集**，V0–V4 共用。
- **标注来源**：3 条人工核验（Q9/Q18/Q19），97 条 AI 标注。
- **用途**：实验对比和相对增益评估，**不是严格人工终审数据集**。
- **内容**：`gold_pages`、`reference_answer`、`reference_contexts`、`modality_required`、`question_type`、`difficulty`。
- RAGAS 评测使用 **30 条分层抽样**（easy/medium/hard 各 10 条）。
- 严格人工审核全部 100 条可在后续阶段补充，当前数据集足以支撑 V0–V4 的相对比较。

## V0 Baseline 锁定指标

| 指标 | 值 | 引擎 |
|---|---|---|
| Recall@5 | 0.8990 | custom |
| MRR | 0.7663 | custom |
| Top-5 Hit Rate | 0.9293 (92/99) | custom |
| Faithfulness | 0.8799 | ragas==0.4.3 |
| Answer Relevancy | 0.9333 | custom LLM judge |
| Context Precision | 0.1006 | ragas==0.4.3 |
| Context Recall | 0.3167 | ragas==0.4.3 |

结果文件：`storage/runs/v0_baseline/phase2_metrics.json`

## 当前阻塞项

1. Docker 未安装；已确认 Milvus Lite 可用。
2. VLM 已可用（Qwen3-VL-32B），Phase 3 可直接使用。
3. 97/100 条 Golden Dataset 为 AI 标注——已确认不阻塞后续阶段。

## 验收记录

| 阶段 | 状态 | 结论 |
|---|---|---|
| 项目骨架 | PASS | `python -m pytest -q tests/smoke`：2 passed |
| Phase 0 | PASS | 全部 6/6 |
| Phase 1 | PASS | 20 题固定回归集、Recall@5=0.8148、Hit Rate=0.9474 |
| Phase 2 | CONDITIONAL_PASS | 100 条 Golden Dataset + ragas 0.4.3、Recall@5=0.8990；97 条 AI 标注 |
| Phase 3 | PASS | 6 图片 VLM + 3 表格入库；V0/V1 Hit Rate 19/20；Q18 V0✗→V1✓ |
| Phase 4 | PASS | BM25 + Dense RRF；19/20 Hit；Q19 Hybrid✓；降级链路完整；54/54 测试 |
| Phase 5 | PASS | BGE-Reranker-v2-m3 精排；Top-1 +5%；MRR +1.9%；20/20 排序变化；63/63 测试 |
| Phase 6 | PASS | LangGraph Verified QA；relevance check；边缘案例 refused；78/78 测试 |
| Phase 7 | CONDITIONAL_PASS | IncrementalIndexer；unchanged=0 embed；modified 全文档重处理；95/95 测试 |
| Phase 8 | PASS | FastAPI 5 端点 + Swagger；/health /query /ingest /evaluate /experiments；109/109 测试 |
| Phase 9A | PASS | React+Vite+TS 前端骨架；导航+仪表盘+系统状态+CORS；14/14 测试 |
| Phase 9B | PASS | QAPanel 接入 POST /query；答案/来源/验证展示；Q18 answered；109/109 测试 |
| Phase 9C | PASS | KnowledgeBase PDF上传+增量统计+文档列表；GET /documents；二次上传 embedded=0；6/6 测试 |
| Phase 9D | PASS | 实验评估页：V0-V5 指标对比+CSS图表+技术演进+案例+结论；GET /experiments；123/123 测试 |
| Phase 10 (V6) | PASS | 句级接地验证；交叉编码器替换余弦；投毒测试 82% 拦截；20+2 评测 17/20+2/2；158/158 测试 |
| Phase 11 (V7) | PASS | LLM 网关：60s 超时+重试+熔断+兜底链+兜底应答；LLMClient 门面零改调用点；19 网关测试；demo failover |
| Phase 12 (V8) | PASS | 多文档 doc_filter、Ecovacs 英文手册、9 条图片题；123 题 retrieval-only 扩展评测 |
| Phase 13 (V9) | PASS | `/query` 两级缓存；精确命中 12/12；命中延迟约 31ms；语义缓存专项测试通过 |
