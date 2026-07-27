# Decision Log

## D-001：实验优先而非生产优先

- 状态：ACCEPTED
- 原因：项目主要用于简历和技术面展示，需要快速跑通闭环和控制变量实验。

## D-002：本地 Embedding/Reranker，API LLM/VLM

- 状态：ACCEPTED
- 原因：兼顾本地 GPU 的可复现性和大模型调用的开发速度。

## D-003：配置文件定义实验版本

- 状态：ACCEPTED
- 原因：避免通过 Git 分支复制代码，确保 V0–V5 可重复运行。

## D-004：先文本 Baseline，再引入高级能力

- 状态：ACCEPTED
- 原因：没有固定 Baseline 就无法证明后续模块的实际收益。

## D-005：开发阶段使用 Milvus Lite

- 状态：ACCEPTED
- 原因：当前环境未安装 Docker；Milvus Lite 足以支持单机实验，后续保留切换 Milvus Server 的适配层。

## D-006：本地模型采用显存保护配置

- 状态：PROVISIONAL
- 原因：当前 GPU 为 RTX 4060 Laptop、约 8GB 显存。实现时默认低 Batch 和半精度，并把模型设备、Batch Size 和最大长度配置化；若 BGE-Reranker-Large 无法稳定加载，必须记录证据后再讨论兼容替代方案。
