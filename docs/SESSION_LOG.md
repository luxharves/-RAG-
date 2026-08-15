# Session Log

## 2026-07-17：项目初始化

- 创建项目治理文档和实验规范。
- 确认项目目标为简历展示、快速闭环和实验对比。
- 确认默认部署方案：本地 BGE-M3/Reranker，API LLM/VLM。
- 未实现业务代码。
- 未宣称 Phase 0 通过。
- 项目结构 Smoke Test：2 passed。
- 新增外部实现 AI 协作协议和 Phase 0 实施任务书。

## 2026-07-18：Phase 0 实施

- 创建 `src/config/settings.py` — 统一配置（pydantic-settings + .env）。
- 创建 `src/infra/embedder.py` — BGE-M3 适配器。
- 创建 `src/infra/reranker.py` — BGE-Reranker 适配器。
- 创建 `src/infra/milvus_client.py` — Milvus 适配器（MilvusClient API, pymilvus 3.0）。
- 创建 `src/infra/llm_client.py` — OpenAI 兼容 LLM 客户端。
- 创建 `src/infra/vlm_client.py` — OpenAI 兼容 VLM 客户端。
- 创建 `scripts/smoke_test.py` — Phase 0 综合 Smoke Test。
- 创建 `tests/smoke/test_config.py` — 配置加载测试。
- 创建 `requirements.txt` — 项目依赖清单。
- 安装 PyTorch 2.6.0+cu124、sentence-transformers、pymilvus 3.0、milvus-lite、modelscope。
- 从 ModelScope 下载 BGE-M3 和 BGE-Reranker-v2-m3 到本地 `models/` 目录。
- 配置 .env：Milvus Lite (milvus.db)、DeepSeek LLM、Qwen3-VL endpoint。
- Phase 0 Smoke Test 结果：PASS（5/6 PASS，1 NOT_RUN）。
- 已知问题：
  - Qwen2-VL 未验证（无测试图片，NOT_RUN，不阻塞 Phase 1）。
  - BGE-M3 初次加载需 25s（模型加载到 GPU）。
  - Milvus Lite 在 Windows 上有临时文件冲突问题（drop_collection 时偶发，不影响功能）。
  - HuggingFace 直连下载不稳定，已改用 ModelScope 预下载。
- 清理临时下载文件和 Milvus 测试数据。
- 更新 README.md、PROGRESS.md、.env.example。

## 2026-07-18：Phase 3–7 实施

- Phase 3: 多模态解析 — 6 图片 VLM 描述 + 3 表格 + V1 Collection + Q18 V0✗→V1✓
- Phase 4: Hybrid Retrieval — BM25 + Dense RRF 融合 + Q19 Dense✗→Hybrid✓ + MRR +15%
- Phase 5: BGE-Reranker — 二阶段精排 + Top-1 +5% + 20/20 排序变化
- Phase 6: LangGraph Verified QA — relevance check + 重试/拒答 + 边缘案例 refused
- Phase 7: 增量更新 — IncrementalIndexer + 同Collection增删改 + Milvus delete + BM25增量
- 全量测试 95/95 passed
- V0–V6 实验产物完整保存于 storage/runs/
- Phase 10 (V6): 句级接地验证(交叉编码器)替换 LLM 自查;投毒测试 82% 拦截;全量 100 题 95/100;158/158 测试
- Phase 11 (V7): LLM 网关(超时/重试/熔断/兜底链);LLMClient 门面零改调用点;demo failover(死主→backup→熔断→探针)
- ROADMAP.md Phase 0–6 PASS, Phase 7 CONDITIONAL_PASS
- 已知限制：modified 全文档重处理（非页面/Chunk 级增量）；多模态 V1 图片/表格 Chunk 修改时未做局部保护
- Phase 8: FastAPI 5 端点 + Swagger
- 真实接口验证: /health(200) /experiments(200+ragas_metrics) /evaluate(200) /query(200,answered)
- Q18 "机器人会不会从楼梯摔下去？" → API 200 + p6 image rank1 + answered
- 4 端点全部验证: /health /query /evaluate /experiments
- .env 9 变量全部匹配代码
- 全量测试 109/109 passed（无锁冲突）
- Phase 8 最终判定: PASS

## 2026-08-05：V8 多文档与 V9 语义缓存

- 新增第二本说明书 Ecovacs DEEBOT T30C，并通过 `doc_filter` 贯通 Dense、BM25、Hybrid、Reranker 和 VerifiedQA 检索链路。
- 新增 `golden_extended.json`，共 123 题（114 text、9 image），完成 V0–V4 retrieval-only 扩展评测。
- 扩展集 V3/V4：Hit@5=0.9756、Recall@5=0.9593、MRR=0.8916、Top-1=0.8293。
- 明确：123 题扩展集未运行 RAGAS；100 题 RAGAS 结果继续保留在 `storage/runs/final_eval/`。
- 新增 V9 `/query` 两级缓存：精确 SHA256 + BGE-M3 语义余弦，SQLite 持久化。
- V9 缓存评测：精确命中 12/12，语义命中 4/12，平均命中延迟约 31ms。
- 新增语义缓存单元测试，配合网关和最终评测测试定向验证共 44 passed；前端 lint/build 通过。
- 已知限制：全量 pytest 包含慢速模型/检索路径，短时间窗口内未完成；语义缓存未自动绑定知识库版本，更新知识库后应清理缓存。
