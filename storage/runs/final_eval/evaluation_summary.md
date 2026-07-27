# Final Evaluation

Evaluation Dataset: golden_100.json  
Total Questions: 100  
Text Questions: 99  
Image/Multimodal Questions: 1  
All V0-V4 use the same dataset and evaluation protocol.

## Retrieval Metrics
| Version | Hit@5 | Recall@5 | MRR | Top-1 Hit | Avg Retrieval Latency |
|---|---:|---:|---:|---:|---:|
| V0 | 0.9200 | 0.8900 | 0.7587 | 0.6400 | 3.1622s |
| V1 | 0.8700 | 0.8142 | 0.7288 | 0.6700 | 3.2324s |
| V2 | 0.9200 | 0.8775 | 0.8373 | 0.7900 | 3.3167s |
| V3 | 0.9500 | 0.9092 | 0.8583 | 0.8100 | 20.4676s |
| V4 | 0.9500 | 0.9092 | 0.8583 | 0.8100 | 16.7227s |

## RAGAS / Answer Metrics
| Version | Faithfulness | Context Precision | Context Recall | Answer Relevancy | Avg Generation Latency |
|---|---:|---:|---:|---:|---:|
| V0 | 0.9217 | 0.6949 | 0.8214 | 0.9100 | 3.1622s |
| V1 | 0.8799 | 0.6824 | 0.7727 | 0.8350 | 3.2324s |
| V2 | 0.9239 | 0.7624 | 0.8337 | 0.8960 | 3.3167s |
| V3 | 0.9533 | 0.8662 | 0.8698 | 0.9180 | 20.4676s |
| V4 | 0.8994 | 0.8525 | 0.8758 | 0.9070 | 16.7227s |

## V5 Incremental Metrics
| Metric | Value |
|---|---:|
| added_count | 0 |
| unchanged_count | 1 |
| modified_count | 0 |
| deleted_count | 0 |
| reprocessed_pages | 0 |
| reused_chunks | 39 |
| embedded_chunks | 0 |
| removed_chunks | 0 |
