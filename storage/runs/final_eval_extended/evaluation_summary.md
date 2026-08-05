# Final Evaluation

Evaluation Dataset: golden_extended.json  
Total Questions: 12  
Modality Counts: {'text': 12}  
Source Documents: ['Roborock G10S']  
All V0-V4 use the same dataset and evaluation protocol.

## Retrieval Metrics
| Version | Hit@5 | Recall@5 | MRR | Top-1 Hit | Avg Retrieval Latency |
|---|---:|---:|---:|---:|---:|
| V0 | 1.0000 | 0.8750 | 0.7083 | 0.5000 | 0.6837s |
| V1 | 1.0000 | 0.8750 | 0.6903 | 0.5000 | 0.8122s |
| V2 | 0.8333 | 0.7500 | 0.6528 | 0.5000 | 0.8065s |
| V3 | 0.9167 | 0.8333 | 0.7500 | 0.6667 | 25.5982s |
| V4 | 0.9167 | 0.8333 | 0.7500 | 0.6667 | 29.1058s |

## V5 Incremental Metrics
| Metric | Value |
|---|---:|
| added_count | 0 |
| unchanged_count | 4 |
| modified_count | 0 |
| deleted_count | 0 |
| reprocessed_pages | 0 |
| reused_chunks | 414 |
| embedded_chunks | 0 |
| removed_chunks | 0 |
