# Data Contracts

## 1. Chunk

每个知识单元必须至少包含：

```json
{
  "chunk_id": "stable-id",
  "document_id": "document-id",
  "document_version": "sha256-prefix",
  "page_number": 1,
  "content": "normalized text",
  "content_type": "text|table|image",
  "source_file": "manual.pdf",
  "source_bbox": null,
  "image_path": null,
  "metadata": {}
}
```

`chunk_id` 必须在同一文档版本内稳定；`content_type` 不允许缺失。

## 2. Evaluation QA

```json
{
  "question": "设备无法开机怎么办？",
  "reference_answer": "根据说明书……",
  "reference_contexts": ["说明书原文片段"],
  "question_type": "troubleshooting",
  "source_document": "manual.pdf",
  "difficulty": "medium",
  "review_status": "human_reviewed"
}
```

## 3. Query result

```json
{
  "chunk_id": "stable-id",
  "content": "retrieved text",
  "source_file": "manual.pdf",
  "page_number": 1,
  "retrieval_channel": "dense|bm25|hybrid",
  "retrieval_score": 0.0,
  "rerank_score": null
}
```

## 4. Experiment run

每次实验必须保存配置快照、模型信息、检索上下文、生成答案、指标、耗时和错误信息，且不能覆盖既有运行结果。

