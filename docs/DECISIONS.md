# Decision Log

## D-001：实验优先而非生产优先

- 状态：ACCEPTED
- 原因：项目主要用于简历和技术面展示，需要快速跑通闭环和控制变量实验。

## D-002：本地 Embedding/Reranker，API LLM/VLM

- 状态：ACCEPTED
- 原因：兼顾本地 GPU 的可复现性和大模型调用的开发速度。

## D-003：配置文件定义实验版本

- 状态：ACCEPTED
- 原因：避免通过 Git 分支复制代码，确保 V0–V6 可重复运行。

## D-004：先文本 Baseline，再引入高级能力

- 状态：ACCEPTED
- 原因：没有固定 Baseline 就无法证明后续模块的实际收益。

## D-005：开发阶段使用 Milvus Lite

- 状态：ACCEPTED
- 原因：当前环境未安装 Docker；Milvus Lite 足以支持单机实验，后续保留切换 Milvus Server 的适配层。

## D-006：本地模型采用显存保护配置

- 状态：PROVISIONAL
- 原因：当前 GPU 为 RTX 4060 Laptop、约 8GB 显存。实现时默认低 Batch 和半精度，并把模型设备、Batch Size 和最大长度配置化；若 BGE-Reranker-Large 无法稳定加载，必须记录证据后再讨论兼容替代方案。

## D-007：多文档数据集用独立 golden_extended.json，不追加进 golden_100.json

- 状态：ACCEPTED
- 原因：golden_100 的 V0-V6 存档评测数字和哈希都基于它，追加会使其失效（需 75+ min 重跑）。新建 golden_extended.json = 100 + 8 图片题 + 15 Ecovacs 题 = 123 题，保留历史数字的同时扩展多文档/多模态覆盖。

## D-008：Ecovacs 题用中文题面 + 英文 reference_contexts

- 状态：ACCEPTED
- 原因：Ecovacs 手册为英文（91 页三语，EN p1-30）。中文题 + 英文上下文直接测 BGE-M3 的跨语言检索能力（中文 query 命中英文 chunk），生成器仍用中文回答。

## D-009：保留 100 题 RAGAS 与 123 题扩展检索两套结果

- 状态：ACCEPTED
- 原因：100 题结果已经包含 V0–V4 的完整 RAGAS 指标；123 题扩展结果用于验证多文档和图片检索能力，当前只完成 retrieval-only。两套结果分目录保存，避免把不同评测协议混成一张表。

## D-010：V7–V9 作为工程能力扩展，不强行并入 V0–V4 增益实验

- 状态：ACCEPTED
- 原因：LLM Gateway、语义缓存和多文档数据增强分别属于可靠性、性能和数据覆盖能力。它们通过独立测试和专项指标证明，不改变已经锁定的 V0–V4 控制变量结果。

## D-011：语义缓存采用 SQLite 和进程内实现

- 状态：ACCEPTED
- 原因：项目定位为单机简历展示，SQLite 足以验证精确缓存、语义缓存和延迟收益；缓存不等同于分布式生产缓存，知识库更新后需要清理或失效旧缓存。
