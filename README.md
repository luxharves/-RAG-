# 多模态 RAG 智能硬件维保助手

基于 BGE-M3 + BM25 Hybrid Retrieval + BGE-Reranker + LangGraph 验证的多模态可信问答系统。

一个陌生人走进来就能看懂的完整项目介绍。

---

## 一、这个项目是干什么的？

你买了一个扫地机器人，附赠一本 27 页的 PDF 说明书。某天它坏了，你在说明书里翻来翻去找答案——很烦。

这个系统做的事：**你对着它用大白话问一个问题，它自动去说明书里找到相关段落，基于原文生成答案，并告诉你答案出自哪一页。如果说明书里没有答案，它会直接说"不知道"，绝不编造。**

比如：

> **你问**：「机器人会不会从楼梯摔下去？」
>
> **系统答**：「设备配备悬崖传感器，可自动检测高度差并转向，防止从楼梯跌落。——来源：说明书第6页」

---

## 二、核心能力

| 能力 | 说明 |
|---|---|
| 多模态检索 | 文本、表格、图片三种内容统一进入同一个向量空间，用同一句话检索 |
| 双通道混合检索 | BGE-M3 语义匹配 + BM25 关键词匹配，RRF 融合，互补长短 |
| 精排重排 | BGE-Reranker-v2-m3 逐对精读打分，把最相关的内容排到最前面 |
| 可信验证 | LangGraph 验证节点检查答案是否基于原文，证据不足就拒答或重试 |
| 增量更新 | SHA256 文件指纹检测变化，没变的文档跳过所有计算，零浪费 |
| 实验可复现 | 100 条 Golden Dataset 固定不变，V0-V4 控制变量实验；V6/V7/V9 通过专项测试和故障注入验证 |
| 多文档 + 元数据过滤 | 双说明书（Roborock 中文 + Ecovacs 英文）同库，检索按 source_file 过滤隔离；Ecovacs 跨语言检索 MRR 0.90，8 新图片题检索全中 |
| 语义缓存 | /query 两级缓存（精确 SHA256 + 语义余弦），重复/相似问题命中后 ~31ms 返回（全管线 ~20s+），省 LLM 调用 |

---

## 三、技术栈

```
语言:       Python 3.11+
后端框架:   FastAPI
前端:       React 19 + TypeScript + Vite
流程编排:   LangGraph StateGraph

Embedding:   BGE-M3 (本地 GPU, 1024 维)
Reranker:    BGE-Reranker-v2-m3 (本地 GPU)
向量数据库:  Milvus Lite (本地文件模式)
关键词检索:  BM25 (jieba 分词 + rank_bm25)
LLM:        DeepSeek V4 Flash (API)
VLM:        Qwen3-VL-32B (API, 图片语义描述)
PDF 解析:   PyMuPDF
评测框架:   RAGAS 0.4.3

GPU:        NVIDIA RTX 4060 Laptop (8 GB)
```

---

## 四、架构概览

```
                         ┌──────────────┐
                         │  React 前端   │
                         │  仪表盘/问答   │
                         │  知识库/实验   │
                         └──────┬───────┘
                                │ HTTP
                         ┌──────▼───────┐
                         │   FastAPI    │
                         │  后端 API    │
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
     ┌────────▼───────┐ ┌──────▼──────┐ ┌───────▼──────┐
     │   离线处理链路   │ │  在线检索    │ │   评测系统    │
     │               │ │             │ │              │
     │ PDF → 文本     │ │ Hybrid 粗筛 │ │ 100条 Dataset│
     │ + 图片VLM描述  │ │ + Rerank    │ │ RAGAS 指标   │
     │ → Chunk 切片   │ │ + Verify    │ │ 控制变量实验  │
     │ → BGE-M3 向量  │ │ → LLM 生成  │ │              │
     │ → Milvus+BM25  │ │             │ │              │
     └────────────────┘ └─────────────┘ └──────────────┘
```

---

## 五、技术演进：V0 → V9

每个版本解决朴素 RAG 的一个痛点，每个版本的效果提升都有数据支撑。

| 版本 | 做了什么 | 解决了什么问题 | 核心指标提升 |
|---|---|---|---|
| **V0** Baseline | PDF 文字提取 → BGE-M3 Dense 检索 → LLM 引用回答 | 关键词搜索太笨，"无法开机"搜不到"电源故障" | Recall@5 0.89 |
| **V1** +多模态 | 6 页图片调用 Qwen3-VL 看图说话 → 统一进入向量空间 | 说明书图片里的信息（传感器位置图等）之前完全搜不到 | 首次覆盖图片问题 |
| **V2** +Hybrid | BM25 关键词 + BGE-M3 语义双通道，RRF 融合排名 | 冷门术语（PTC、E07）纯语义检索会漏掉 | MRR **0.76→0.84** (+10.5%) |
| **V3** +Reranker | Hybrid 粗筛 20 条 → BGE-Reranker 逐对精读打分 → 取前 5 | 粗筛排序不够精确，最相关的内容可能排在第 3-5 名 | Recall@5 **0.88→0.91**，Context Precision **0.76→0.87** |
| **V4** +Verify | LangGraph 验证节点：逐条核对答案是否基于原文 → 拒答/重试 | LLM 还是会编造细节，"买原装配件199元"原文根本没有 | 越界问题正确拒答，Context Recall 0.88 |
| **V5** +增量 | SHA256 文件指纹 → 区分 added/unchanged/modified/deleted | 说明书换新版要全部重新处理，浪费算力 | 未变化文档 0 次 Embedding |
| **V6** +确定性接地 | 答案拆句 + BGE-Reranker 交叉编码器逐句核对 → 无支撑句子拒答/重试 + 引用审计 | V4 的验证靠 LLM"自查"不可复现 | 引用由系统"计算"而非 LLM"声称"；投毒测试拦截 82% 编造句（余弦为 0%） |
| **V7** +LLM Gateway | 超时、指数退避重试、熔断、Provider 故障转移和统一兜底 | 外部模型服务不稳定会阻塞或导致问答失败 | 故障注入覆盖重试、熔断、failover 和 fallback |
| **V8** +多文档扩展 | 第二本说明书、文档级过滤、跨语言问题和新增图片问题 | 多说明书场景下来源混淆、页码冲突和图片覆盖不足 | 123 题 retrieval-only：V3/V4 Hit@5 0.9756、MRR 0.8916 |
| **V9** +语义缓存 | SHA256 精确缓存 + BGE-M3 语义缓存 + SQLite 持久化 | 重复或近似问题重复调用大模型、延迟高 | 精确命中 12/12，命中延迟约 31ms，节省 12 次 LLM 调用 |

---

## 六、核心评测指标

基于 100 条固定 Golden Dataset（覆盖故障排查、日常维护、功能询问、初始设置），V0-V4 最终评测结果：

> 说明：下表是 100 题正式 RAGAS 评测。另有 123 题 `golden_extended.json` 多文档扩展集，但当前只完成 retrieval-only 评测，不与下表的 RAGAS 指标混合。

| 指标 | V0 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| **Recall@5** | 0.8900 | 0.8142 | 0.8775 | **0.9092** | 0.9092 |
| **MRR** | 0.7587 | 0.7288 | 0.8373 | **0.8583** | 0.8583 |
| **Top-1 Hit** | 0.64 | 0.67 | 0.79 | **0.81** | 0.81 |
| **Faithfulness** | 0.9217 | 0.8799 | 0.9239 | **0.9533** | 0.8994 |
| **Context Precision** | 0.6949 | 0.6824 | 0.7624 | **0.8662** | 0.8525 |
| **Context Recall** | 0.8214 | 0.7727 | 0.8337 | 0.8698 | **0.8758** |
| **Answer Relevancy** | 0.91 | 0.84 | 0.90 | **0.92** | 0.91 |
| **平均延迟** | 3.2s | 3.2s | 3.3s | 20.5s | 16.7s |

### V6 确定性接地验证（句级）

| 指标 | V6 |
|---|---|
| 全量 100 题 answered | **95/100** |
| 边界题 refused（火星/核聚变） | **2/2** |
| 句级支撑率 | **99.35%**（avg support_ratio） |
| 交叉编码器分数 | median 0.95（真实句） |
| 投毒测试拦截率 | **28/34（82%）**（BGE-M3 余弦为 0/34） |
| 接地过度拒答 | 1/100（短否定释义"不可以用洗涤剂"未匹配） |

机制：答案拆句 → 每句与检索 chunk 用 **BGE-Reranker 交叉编码器**逐对打分 → 无支撑句子标记/拒答 → 引用审计逐条核对 LLM 声称的 `[来源: 第X页]`。引用由**系统计算**而非 LLM 声称。
**投毒测试（对抗验证）**：对真实答案追加编造句（"本产品由核聚变反应堆提供动力。" 等），交叉编码器标记 82% 无支撑、35% 答案翻转拒答；残余漏网为"主题相关编造"——交叉编码器是相关性打分器，拦不住所有主题相关幻觉（弗兰肯斯坦边界）。另有 1/100 过度拒答（正确但被否定释义的短句）。两个边界均已诚实记录。

### 关键发现

- **V2 性价比最优**：MRR 提升 10.5%，仅增加 0.15s 延迟
- **V3 质量最优**：全部指标达到顶峰，代价是 6x 延迟（Reranker 逐对精读）
- **V4 最安全**：越界问题正确拒答，Context Recall 最高（重试机制补上了漏掉的 Chunk）
- **V5 零浪费**：未变化文档 39 个 Chunk 全部复用，0 次无效 Embedding
- **V6 可复现**：引用由 BGE-M3 余弦"计算"而非 LLM 声称，投毒测试证明能拦截编造句

### 指标解释

| 指标 | 一句话解释 |
|---|---|
| Recall@5 | 前 5 条结果里能找到正确答案的概率 |
| MRR | 第一条正确答案排在第几名（越靠前越高） |
| Top-1 Hit | 第一条结果就命中正确答案的概率 |
| Faithfulness | 生成的答案是否基于原文（而不是模型编的） |
| Context Precision | 返回的 5 条结果中有几条真的相关 |
| Context Recall | 标准答案的关键信息被检索到的段落覆盖了多少 |

---

## 七、项目结构

```
.
├── main.py                     # FastAPI 入口
├── requirements.txt            # Python 依赖
├── pyproject.toml              # 项目配置
├── .env.example                # 环境变量模板（无真实密钥）
├── README.md                   # 你正在看的文件
│
├── src/
│   ├── config/settings.py      # 统一配置（环境变量读取）
│   ├── infra/                  # 基础设施适配器
│   │   ├── embedder.py         #   BGE-M3 封装
│   │   ├── reranker.py         #   BGE-Reranker 封装
│   │   ├── llm_client.py       #   LLM API 客户端
│   │   ├── vlm_client.py       #   VLM API 客户端
│   │   └── milvus_client.py    #   Milvus 适配器
│   ├── ingestion/              # 文档摄取
│   │   ├── document.py         #   Document + Chunk 数据模型
│   │   ├── pdf_parser.py       #   PDF 文本/表格/图片提取
│   │   ├── chunker.py          #   文本切片
│   │   ├── manifest.py         #   文档清单 + SHA256 指纹
│   │   └── incremental.py      #   增量更新索引器
│   ├── retrieval/              # 检索模块
│   │   ├── retriever.py        #   Dense 检索器 (BGE-M3 + Milvus)
│   │   ├── bm25.py             #   BM25 关键词检索器
│   │   ├── hybrid_retriever.py #   混合检索 + RRF 融合
│   │   └── reranked_retriever.py # 二阶段精排 (Hybrid → Reranker)
│   ├── generation/generator.py # LLM 答案生成（含引用格式）
│   ├── workflow/verified_qa.py # LangGraph 可信问答流程
│   ├── workflow/grounding.py   # V6 确定性句级接地验证（BGE-M3 余弦）
│   ├── eval/                   # 评测系统
│   │   ├── metrics.py          #   LLM-as-Judge 指标
│   │   ├── ragas_patch.py      #   RAGAS 兼容补丁
│   │   └── doc_registry.py     #   source_document → source_file 映射（doc 过滤）
│   └── api/                    # FastAPI 接口
│       ├── routes.py           #   路由定义
│       └── deps.py             #   依赖注入
│
├── frontend/                   # React 19 前端
│   └── src/pages/
│       ├── Dashboard.tsx       #   仪表盘
│       ├── QAPanel.tsx         #   维保问答
│       ├── KnowledgeBase.tsx   #   知识库管理（PDF 上传）
│       ├── Experiments.tsx     #   实验评估对比
│       └── SystemStatus.tsx    #   系统状态
│
├── configs/experiments/        # 实验版本配置文件
│   ├── v1_multimodal.yaml
│   ├── v2_hybrid.yaml
│   ├── v3_rerank.yaml
│   └── v4_verified.yaml
│
├── scripts/                    # 运行脚本
│   ├── smoke_test.py           #   Phase 0 冒烟测试
│   ├── ingest.py               #   文档入库
│   ├── query.py                #   命令行问答
│   ├── compare_v3.py           #   V2 vs V3 对比
│   ├── incremental_update.py   #   增量更新
│   ├── final_evaluation.py     #   最终评测（支持 --dataset/--retrieval-only）
│   ├── run_v6_eval.py          #   V6 接地评测 + 阈值扫描 + V4 对比
│   ├── build_extended_dataset.py # 扩展数据集（+8 图片题 +15 Ecovacs 题）
│   └── check_doc_filter.py     #   doc 级过滤隔离性验证
│
├── docs/                       # 项目文档
│   ├── PROJECT_CHARTER.md      #   项目章程
│   ├── ARCHITECTURE.md         #   架构设计
│   ├── DATA_CONTRACTS.md       #   数据契约
│   ├── DECISIONS.md            #   决策日志
│   └── EXPERIMENT_PROTOCOL.md  #   实验协议
│
├── tests/                      # 测试
├── data/                       # 数据（说明书 PDF + 评测数据集）
└── storage/                    # 运行结果
```

---

## 八、快速开始

### 环境要求

- Python 3.11+
- NVIDIA GPU（可选，CPU 也能跑但慢）
- 操作系统：Windows / Linux / macOS

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```ini
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=你的Key
LLM_MODEL=deepseek-v4-flash

VLM_BASE_URL=https://你的VLM地址/v1
VLM_API_KEY=你的Key
VLM_MODEL=qwen3-vl-32b-thinking

MILVUS_URI=milvus.db
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-large
```

### 3. 运行冒烟测试

验证所有基础设施组件是否正常工作：

```bash
python -m pytest -q tests/smoke
python scripts/smoke_test.py
```

### 4. 启动后端

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

访问 http://127.0.0.1:8000/docs 查看 Swagger API 文档。

### 5. 启动前端（可选）

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173。

---

## 九、致谢

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — Embedding 模型
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — Reranker 模型
- [Milvus](https://milvus.io/) — 向量数据库
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 流程编排
- [RAGAS](https://docs.ragas.io/) — RAG 评测框架
