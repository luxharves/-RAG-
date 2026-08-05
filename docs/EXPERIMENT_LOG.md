# Experiment Log

## V0 Baseline (Phase 2, locked)

| Run ID | Version | Dataset | Metrics | Status | Notes |
|---|---|---|---|---|---|
| v0_retrieval_20260718 | V0 | golden_100.json (99q text) | Recall@5=0.8990, MRR=0.7663, Hit@5=0.9293 | COMPLETED | dense BGE-M3 |
| v0_ragas_20260718 | V0 | golden_100.json (30q sample) | Faithfulness=0.8799, ContextPrecision=0.1006, ContextRecall=0.3167, AnswerRelevancy=0.9333 | COMPLETED | ragas 0.4.3 + custom |

### Provenance

| Source | Count | Notes |
|---|---|---|
| phase1_cached | 19 | from Phase 1 text questions (max=19) |
| phase2_generated | 80 | newly generated for Phase 2 |
| ragas_sampled | 30 | stratified (7 phase1 + 23 phase2) |

### Metric Details

| Metric | Value | Engine |
|---|---|---|
| Recall@5 | 0.8990 | custom |
| MRR | 0.7663 | custom |
| Top-5 Hit Rate | 0.9293 (92/99) | custom |
| Faithfulness | 0.8799 | ragas==0.4.3 |
| Answer Relevancy | 0.9333 | custom LLM judge |
| Context Precision | 0.1006 | ragas==0.4.3 |
| Context Recall | 0.3167 | ragas==0.4.3 |

### RAGAS Known Issues

- `IncompleteOutputException`: DeepSeek v4-flash occasionally hits max_tokens. Some scores may be underestimates.
- `answer_relevancy`: ragas needs embeddings API (unavailable via DeepSeek). Computed via custom LLM judge.

### Dataset Status

- Total: 100 questions (99 text + 1 image Q18)
- Human-reviewed: 3 (Q9, Q18, Q19)
- AI-annotated: 97
- **Dataset purpose**: experimental comparison and relative gain evaluation
- **Not a strictly human-verified final dataset**; human review can be completed incrementally

## V1 Multimodal (Phase 3)

| Run ID | Version | Collection | Chunks | V0 Hit | V1 Hit | Q18 | Status |
|---|---|---|---|---|---|---|---|
| v1_multimodal_20260718 | V1 | v1_multimodal_20260718_202339 | 48 (39t+6i+3tb) | 18/20 | 18/20 | MISS→MISS | COMPLETED |
| v1_multimodal_kw_20260718 | V1+KW | v1_multimodal_kw_20260718_204259 | 48 (39t+6i*+3tb) | 18/20 | **19/20** | **MISS→HIT** | COMPLETED |

*: image chunks with keyword summary prefix

### Key Finding
### V0/V1 Comparison (20 fixed questions)

| Question | Modality | V0 | V1 (keyword-enhanced) |
|---|---|---|---|
| Q18 (楼梯摔下去) | image | MISS | **HIT** (p6 image Chunk, rank 1, score 0.61) |
| Q19 (开机后不动) | text | MISS | MISS |
| Q1–Q17, Q20 | text | 17/17 HIT | 17/17 HIT |

### Key Findings
- Q18: VLM correctly described p6 diagram including "悬崖传感器...防止机器人跌落". With keyword summary prefix added to image chunks, BGE-M3 successfully matched semantic query.
- Q19: Persistent text retrieval failure across V0 and V1 — unrelated to multimodal.
- Image chunks don't degrade text retrieval quality (page ordering shifts slightly but hits unchanged).
- Table chunks are retrievable (e.g. "产品有害物质含量表" → p22 table Chunk rank 1).

## V2 Hybrid Retrieval (Phase 4)

| Run ID | Version | Modes | Hit Rate | MRR | Q18 | Q19 | Status |
|---|---|---|---|---|---|---|---|
| v2_dense_20260718 | V2 | Dense | 19/20 | 0.6683 | HIT | MISS | — |
| v2_bm25_20260718 | V2 | BM25 | 14/20 | 0.3933 | MISS | MISS | — |
| v2_hybrid_20260718 | V2 | **RRF** | **19/20** | **0.7683** | **HIT** | **HIT** | COMPLETED |

### Key Findings
- Q19 breakthrough: "开机后机器人不动怎么办" — Dense and BM25 both missed gold pages individually, but RRF fusion of top-20 results pushed page 24 into top-5.
- MRR +15% over Dense-only: rank-1 accuracy improved by fusion.
- BM25 alone is weak on this dataset (14/20) — Chinese tokenization and short manual make keyword overlap unreliable.
- BM25 index: 48 docs, jieba tokenization, persisted to `storage/bm25/bm25_index.pkl`.
- Fallback: if Dense or BM25 fails individually, Hybrid degrades to the working channel.

## V6 Deterministic Grounding (Phase V6)

| Run ID | Version | Cases | Status | Notes |
|---|---|---|---|---|
| v6_grounding | V6 | 20 fixed + 2 edge | RUNNING | 句级 BGE-M3 余弦接地,替换 LLM 自查 |

### Mechanism (打分器演化:余弦 → 交叉编码器)

**V6 初版用 BGE-M3 余弦**,投毒测试暴露 0 判别力;经决定性实验后**换成 BGE-Reranker 交叉编码器**(联合阅读 (句子, chunk)):

```
答案 → 拆句(。！？; ,括号/引用标记深度保护)
  → 每句去引用标记 → 交叉编码器逐对打分 (sentence, chunk)
  → 每句取最大分,≥ scorer_floor(0.1) 记为 supported
  → support_ratio = supported / 总句数(跳过 <5 字碎片句)
  → 低于 min_support_ratio(0.7) → 拒答/重试(复用 V4 LangGraph 重试链路)
  → 引用审计:LLM 声称的 [来源: 文件, 第X页] 与计算接地逐条核对
      → confirmed / unconfirmed / unmatched(幻觉引用)
```

### Key Findings

- 引用从"LLM 声称"变成"系统计算":答案每句是否被检索 chunk 支撑,由交叉编码器分数决定,可复现、可调阈值、可给逐句证据。
- `VERIFIER_MODE=llm` 可一键回到 V4 的 LLM-as-judge,便于对照;`GROUNDING_SCORER=cosine` 可回到初版余弦。
- 边界题(火星/核聚变)由 relevance 检查 + 接地双重拦截,保持 refused。
- **投毒测试是关键验证**(对真实答案追加编造句):
  - BGE-M3 余弦:0/34 拦截(0 判别力,相似度被主题词抬高)
  - BGE-Reranker 交叉编码器:28/34 拦截(82%),12/34 答案翻转拒答
  - 残余 6/34 漏网为"主题相关编造"(如"核聚变反应堆提供动力"vs 手机连接 chunk 里的功率内容)—— 交叉编码器是相关性打分器,非蕴含判别,这是"弗兰肯斯坦幻觉"的已知边界。
- 真实句交叉编码器分数 0.29–1.0(median 0.96),编造句大多 <0.1,间隔充裕。

### 评测结果 (20 固定 + 2 边界)

| 指标 | V6 |
|---|---|
| 固定题 answered | 17/20 |
| 边界题 refused | 2/2 |
| 到达验证的句子 | 45 |
| 句级支撑率 | 100% (avg support_ratio 1.0) |
| 交叉编码器分数分布 | 0.29 – 1.0 (median 0.96) |
| 接地误杀数 | 0 (3 个拒答均为检索相关性/LLM 自拒,非接地) |
| 投毒拦截率 | 28/34 (82%) |

- 20 固定题中 3 题 refused 均非接地所致:"如何设置虚拟墙"为检索相关性拒答(recall 失败),"机器人为什么一直回充""开机后机器人不动怎么办"为 LLM 自行拒答。
- 16 条投毒"被标记但答案未拒":答案多句时支撑率仍 ≥0.7,系统回答但把无支撑句列在 `unsupported_claims` —— 部分接地设计。

### 全量 100 题评测 (run: full)

| 指标 | V6 |
|---|---|
| answered | **95/100** |
| avg support_ratio | 0.9935 (94/96 例 =1.0) |
| retries 使用 | 5 |
| 交叉编码器分数 | median 0.95 (183 句, min 0.032) |

- 5 个拒答中 4 个先于接地(检索相关性 1 + LLM 自拒 3);**1 个为接地过度拒答**:"清洁尘盒时可以用洗涤剂吗" 的答案正确(说明书:"不要添加任何洗涤剂"),但短句"不可以用洗涤剂。"是否定释义,交叉编码器仅给 0.032,支撑率 0.5 < 0.7 → 拒答。记录为接地边界。

## V8 数据强化 — 扩图片题 + 第二本说明书(doc 级过滤)

| 项 | 内容 |
|---|---|
| 数据集 | `golden_extended.json` = 100 + 8 图片题 + 15 Ecovacs 题 = **123 题**(Roborock 108 / Ecovacs 15;text 114 / image 9) |
| 第二手册 | Ecovacs DEEBOT T30C Manual.pdf(91 页英文,三语 EN/FR/ES,EN p1-30)摄入进同一 collection + BM25(新增 ~335 chunk) |
| doc 过滤 | 检索链(Dense/BM25/Hybrid/Reranked/VerifiedQA)贯通 `doc_filter`,Milvus 按 `source_file` 过滤,BM25 打分后过滤 |
| 隔离验证 | `scripts/check_doc_filter.py`:Roborock query 过滤后 5/5 全回 Roborock;Ecovacs query 过滤后 5/5 全回 Ecovacs;unfiltered 混合 ✅ |

### 关键决策

- **独立 golden_extended.json,不动 golden_100.json** —— 保住 V0-V6 存档数字(95/100 仍有效);见 DECISIONS D-007。
- **Ecovacs 题中文题面 + 英文 reference_contexts** —— 直接测 BGE-M3 跨语言检索;见 DECISIONS D-008。
- Ecovacs 手册全量摄入含 FR/ES chunk,中文 query 靠英文 chunk 命中(可选后续只留英文页)。

### 评测 (123 题检索-only,完整 V0-V4)

| 版本 | Hit@5 | Recall@5 | MRR | Top-1 |
|---|---|---|---|---|
| V0 (Dense, 单文档) | 0.805 | 0.780 | 0.659 | 0.553 |
| V1 (+多模态+doc过滤) | 0.911 | 0.882 | 0.771 | 0.675 |
| V2 (+Hybrid) | 0.943 | 0.913 | 0.786 | 0.675 |
| V3/V4 (+Reranker) | **0.976** | **0.959** | **0.892** | 0.829 |

- 按文档拆分(V3/V4):Roborock 108 题 MRR=0.890;Ecovacs 15 题(中文题检索英文手册)**MRR=0.900** —— 跨语言检索成立。
- **8 张新图片题:Hit@5 / Recall@5 / MRR 全 1.000** —— VLM 图片描述完全可检索,多模态扩题成功。
- V0 低是因为 v0_naive_rag 是单文档 collection,对 Ecovacs 15 题无法检索(doc_filter 返回空)→ 拖低整体。V1+ 用多模态 collection + doc 过滤后大幅提升。
- 评测命令(后台长任务会被环境杀,V3/V4 用 `scripts/run_retrieval_batch.py` 分批 ~7 次跑完):
  `python scripts/final_evaluation.py --dataset golden_extended.json --run-dir storage/runs/final_eval_extended_full --retrieval-only`

## V9 语义缓存

`/query` 两级缓存(精确 SHA256 + 语义 BGE-M3 余弦 >0.9),命中直接返回,SQLite 持久化。

### 评测结果(预热 12 题 + 重跑 + 12 释义变体)

| 指标 | 值 |
|---|---|
| 精确重跑命中 | **12/12 (100%)** |
| 命中延迟 | **31ms** |
| 未命中(全管线) | ~20s 基线(评测时 DeepSeek 慢达 113s) |
| 释义变体命中(0.9 阈值) | 4/12 (33%) |
| 整体命中率(重跑+释义) | 67% |
| LLM 调用节省 | 12 次(精确重跑) |

### 关键发现:阈值标定是双峰分布

- 释义变体对原句的余弦 **双峰**:4 条近似重复(0.93-0.99,命中);8 条松散改写(0.63-0.75,**任何合理阈值都抓不住**)。
- 结论:**语义缓存抓"近似重复",不抓"完全改写"** —— 与生产实践一致(Redis 推荐阈值 0.85-0.95);threshold 0.9 正确(近重复命中、松散改写不误命中)。
- 精确命中 31ms vs 全管线 ~20s ≈ **600x 加速**;LLM 调用按命中数线性节省。

### 评测命令

`python scripts/eval_cache.py --warm 12` → `storage/runs/v9_cache/cache_eval.json`
