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
