# 多模态 RAG 智能硬件维保助手

这是一个面向简历展示和可复现实验的多模态 RAG 项目。项目将从文本向量检索 Baseline 开始，逐步加入多模态解析、Hybrid Retrieval、BGE-Reranker、LangGraph 验证节点和增量更新机制。

## 当前状态

- 当前阶段：Phase 1（文本 Naive RAG）— **IN_PROGRESS**
- 当前版本：V0 Baseline 已生成
- 当前事实来源：`docs/`
- 已完成：Config / BGE-M3 / BGE-Reranker / Milvus Lite / LLM / VLM + PDF 入库 + 20 题问答

## 环境准备

### 1. Python 环境

Python 3.11+，推荐使用 conda/venv。

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 配置环境变量

```powershell
cp .env.example .env
# 编辑 .env，填入真实的 LLM_API_KEY、VLM_API_KEY 等
```

### 4. 下载本地模型

如果网络可访问 HuggingFace，模型会在首次使用时自动下载。也可使用 ModelScope 预下载：

```powershell
python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-m3', cache_dir='./models')
snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='./models')
"
# 然后在 .env 中设置：
# EMBEDDING_MODEL=D:/Agentproject1/models/models/BAAI--bge-m3/snapshots/master
# RERANKER_MODEL=D:/Agentproject1/models/models/BAAI--bge-reranker-v2-m3/snapshots/master
```

### 5. 运行 Phase 0 验收

```powershell
python -m pytest -q tests/smoke
python scripts/smoke_test.py
```

## 协作规则

本项目严格按阶段推进。实现者只能执行当前阶段，完成后提交代码、测试输出和验收证据，经过验收后才能进入下一阶段。

详细规则见：

- `docs/PROJECT_CHARTER.md`
- `docs/ROADMAP.md`
- `docs/ACCEPTANCE_MATRIX.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/IMPLEMENTER_PROTOCOL.md`

## 预期运行方式

后续阶段将提供以下命令：

```powershell
python scripts/ingest.py --config configs/experiments/v0_naive.yaml
python scripts/query.py --config configs/experiments/v0_naive.yaml --question "设备无法开机怎么办？"
python scripts/run_experiment.py --config configs/experiments/v0_naive.yaml
```

上述命令在对应阶段完成前不可视为可用接口。
