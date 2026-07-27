# Phase 0 Implementation Brief

## 目标

完成开发环境、依赖、模型适配器和 Milvus 的最小连通性验证，为 Phase 1 文本 Naive RAG 做准备。

## 允许修改

```text
requirements.txt 或 pyproject.toml
.env.example
src/config/
src/infra/
scripts/smoke_test.py
tests/smoke/
README.md
docs/PROGRESS.md
docs/SESSION_LOG.md
```

可以创建必要的 `__init__.py`，但不要实现完整 RAG 业务流程。

## 禁止实现

- PDF 业务解析和 Chunking。
- BM25、Hybrid Retrieval 或 Reranker 业务流程。
- Qwen2-VL 多模态解析流程。
- LangGraph Verify。
- RAGAS 评测。
- FastAPI 业务接口。

## 必须完成

1. 统一配置读取：环境变量和 `.env`，密钥不进入代码。
2. BGE-M3 最小加载和单文本 Embedding Smoke Test。
3. BGE-Reranker 最小加载和一组 Query/Document 打分 Smoke Test。
4. Milvus 建立临时 Collection、写入一条向量并查询回来。
5. OpenAI 兼容 LLM 调用 Smoke Test。
6. OpenAI 兼容 VLM 图片调用 Smoke Test；若没有可用图片，提供明确的 `NOT_RUN` 结果，不得伪造通过。
7. 将每项结果打印为结构化表格或 JSON，标记 `PASS`、`FAIL`、`NOT_RUN`。
8. 更新 README 的真实安装和运行命令。

## 验收命令

```powershell
python -m pytest -q tests/smoke
python scripts/smoke_test.py
```

## 通过条件

- 配置读取测试通过。
- BGE-M3、BGE-Reranker、Milvus、LLM 均有真实运行证据。
- Qwen2-VL 有真实运行证据；如果 API 或图片条件未准备好，必须标记为阻塞，不得进入 Phase 1 的多模态实现，但可以先继续文本 Baseline。
- 所有失败和未运行项记录在 `docs/PROGRESS.md`。
- 不得把本阶段的结构测试结果当成模型连通性结果。

## 完成报告格式

```text
Phase: 0
Status: PASS | CONDITIONAL_PASS | FAIL

Modified files:
- ...

Commands:
- ...

Results:
- Config: PASS/FAIL/NOT_RUN
- BGE-M3: PASS/FAIL/NOT_RUN
- BGE-Reranker: PASS/FAIL/NOT_RUN
- Milvus: PASS/FAIL/NOT_RUN
- LLM: PASS/FAIL/NOT_RUN
- Qwen2-VL: PASS/FAIL/NOT_RUN

Known issues:
- ...
```

