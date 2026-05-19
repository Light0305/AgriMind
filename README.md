# AgriMind — 作物智能会诊系统

**团队：禾智** | 西北农林科技大学 | 第28届中国机器人及人工智能大赛 · 人工智能创新赛

基于多智能体辩论协议（DDP）的可解释农业诊断系统，将作物诊断从图像分类问题重新定义为多模态推理问题。

---

## 核心创新

| 模块 | 说明 |
|---|---|
| **诊断辩论协议 (DDP)** | 初诊、质疑、仲裁三个 AI 专家结构化 2 轮辩论，输出可解释诊断报告 |
| **主动视觉问诊 (AVD)** | AI 主动评估图片信息充分度，智能引导用户拍摄关键视角（最多 3 轮） |
| **记忆增强** | 检索相似历史病例注入 Agent，证据链可追溯 |
| **选择性辩论** | 初诊与质疑达成共识时自动跳过冗余辩论轮次，节省 50% 推理成本 |
| **指南锚定仲裁** | RAG 检索植保防治知识，仲裁专家引用权威指南做出裁定 |
| **AgriReason 评测基准** | 自建农业多模态推理评测数据集，800 样本 × 5 任务类型 |
| **双模式推理** | 支持本地 GPU（4-bit Qwen2.5-VL-7B）和 DashScope API（云端 72B+） |

---

## 下载与安装

项目提供两种包：

### 代码包（~5MB）

包含所有源代码、前端、测试图片（50 张）、文档。**开箱即用，无需下载模型**——内置默认 API key，直接用 DashScope 云端推理。

```bash
git clone https://github.com/<your-org>/AgriMind.git
cd AgriMind
bash setup.sh   # 一键安装依赖 + 初始化知识库
source venv/bin/activate
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
cd ../frontend && npm install && npm run dev
```

> 默认使用 API 模式（零模型下载）。如您有自己的 DashScope key 或有本地 GPU，参见下方"切换推理模式"。

### 完整包（~16GB，含模型权重）

包含代码 + 微调后的 AgriMind-v2 模型（8.29B 参数，bf16 格式），下载即可运行。

**下载链接：**
- [Hugging Face](https://huggingface.co/<your-org>/agrimind-v2)（推荐，免费不限速）
- 百度网盘：[链接待上传]
- 阿里云盘：[链接待上传]

```bash
# 下载后解压
tar -xzf agrimind-full.tar.gz
cd AgriMind

# 本地 GPU 模式启动
source venv/bin/activate
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# 或 API 模式（无需 GPU）
AGRIMIND_API_KEY=sk-xxx uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> 推荐将完整包上传到 **Hugging Face**（免费、不限容量、支持 Git LFS 大文件），国内用户备用**百度网盘**。

---

## 硬件要求

| 最低配置 | 推荐配置 |
|---|---|
| NVIDIA GPU 8GB 显存 | NVIDIA RTX 4090 24GB |
| 系统内存 32GB | 系统内存 64GB+ |
| 磁盘 30GB | 磁盘 50GB+ |
| CUDA 12.0+ | CUDA 12.4+ |

> **支持的 GPU**：RTX 3070/3080/3090/4070/4080/4090, A100, V100 等。仅需 8GB 显存即可运行 4-bit 量化模型。

> **无 GPU 方案**：使用 API 模式，调用 DashScope Qwen-VL API，无需本地 GPU。在 Web 界面勾选"API 模式"即可。

---

## 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境
conda create -n agrimind python=3.11 -y
conda activate agrimind

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install transformers accelerate bitsandbytes peft
pip install fastapi uvicorn websockets chromadb Pillow sentence-transformers

# 如果使用 API 模式，还需安装：
pip install openai

# 前端（可选，仅需 Node.js 18+）
cd frontend && npm install
```

### 2. 下载模型

下载微调后的 AgriMind 模型到 `models/agrimind-v2/` 目录，或使用基础 Qwen2.5-VL-7B：

```bash
# 选项 A：使用官方 Qwen2.5-VL-7B（需要自行微调）
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5-vl-7b

# 选项 B：使用预训练的 AgriMind-v2 模型（如有提供）
# 将 agrimind-v2 放到 models/agrimind-v2/
```

### 3. 初始化知识库

```bash
cd backend
PYTHONPATH=. python app/rag/indexer.py
```

### 4. 启动后端

```bash
# 本地模式（需要 GPU）
cd backend
AGRIMIND_MODEL_PATH=models/agrimind-v2 uvicorn app.main:app --host 0.0.0.0 --port 8000

# API 模式（无需 GPU，需设置 API key）
AGRIMIND_API_KEY=sk-xxx uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. 启动前端（开发模式）

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173` 即可使用。

### 6. 使用测试图片

`test_images/` 目录包含 50 张来自 PlantVillage 数据集的作物病害图片，涵盖 24 个类别（病害 20 种 + 健康 4 种）。可用于测试系统功能。

---

## Docker 部署（一行启动）

```bash
docker-compose up -d
```

---

## 项目结构

```
robot/
├── backend/
│   ├── app/
│   │   ├── agents/              # DDP 三 Agent
│   │   │   ├── base.py          # Agent 基类
│   │   │   ├── proposer.py      # 初诊专家
│   │   │   ├── challenger.py   # 质疑专家
│   │   │   ├── arbiter.py      # 仲裁专家
│   │   │   └── ddp.py          # DDP 编排器（增强版 v2）
│   │   ├── api/
│   │   │   ├── main.py          # FastAPI 应用入口
│   │   │   ├── routes.py        # REST API 路由
│   │   │   └── websocket.py    # WebSocket 实时辩论推送
│   │   ├── avd/
│   │   │   ├── engine.py        # AVD 主动问诊引擎
│   │   │   └── session.py       # AVD 会话管理
│   │   ├── model/
│   │   │   └── inference.py     # VLM 推理封装（本地+API 双模式）
│   │   ├── rag/
│   │   │   ├── indexer.py       # ChromaDB 知识索引器
│   │   │   └── retriever.py     # 知识检索器
│   │   ├── retrieval/
│   │   │   └── similar_cases.py # 相似病例检索
│   │   ├── schemas.py           # Pydantic 数据模型
│   │   └── config.py            # 配置管理
│   └── tests/
│       ├── test_ddp.py          # DDP 辩论测试（18 项）
│       ├── test_avd.py          # AVD 引擎测试（28 项）
│       ├── test_api.py          # API 端点测试（12 项）
│       ├── test_rag.py          # RAG 检索测试（7 项）
│       ├── test_retrieval.py    # 病例检索测试（11 项）
│       ├── test_schemas.py      # 数据模型测试（36 项）
│       └── conftest.py          # 测试配置
├── frontend/
│   └── src/
│       ├── components/          # React 组件
│       │   ├── ImageUpload.tsx  # 图片上传
│       │   ├── DebateViewer.tsx # 辩论过程可视化
│       │   ├── DiagnosisReport.tsx # 诊断报告卡片
│       │   ├── ChatPanel.tsx    # 聊天面板
│       │   ├── ImageAnnotation.tsx # 图片标注
│       │   └── SimilarCases.tsx # 相似病例展示
│       ├── hooks/
│       │   ├── useDiagnosis.ts  # 诊断状态管理
│       │   └── useWebSocket.ts  # WebSocket 连接管理
│       └── pages/
│           ├── DiagnosisPage.tsx # 主诊断页
│           └── ComparePage.tsx  # 对比实验页
├── benchmark/
│   ├── generate_agrireason.py   # AgriReason 自动生成（800 样本）
│   ├── evaluate.py              # 模型评测
│   ├── ablation.py              # 消融实验编排器（6 配置）
│   ├── metrics.py               # 评测指标（ROUGE-L, ECE）
│   ├── semantic_eval.py         # 语义准确率评测（BGE-M3）
│   ├── baselines/
│   │   ├── resnet_baseline.py   # ResNet-50 基线
│   │   └── vlm_baselines.py     # VLM 零样本/CoT 基线
│   └── data/
│       └── agrireason_v1.json   # AgriReason v1 基准数据（800 样本）
├── training/
│   ├── train_qlora.py           # QLoRA 微调脚本
│   ├── merge_adapter.py         # 合并 LoRA → 完整模型
│   ├── configs/
│   │   └── qlora_qwen_vl.yaml   # 训练配置
│   └── scripts/
│       ├── prepare_data.py      # 数据预处理
│       ├── generate_instructions.py  # 单轮诊断数据生成
│       ├── generate_debate.py   # 辩论数据生成
│       └── generate_multiturn.py # 多轮问诊数据生成
├── models/                      # 模型文件（agrimind-v2, qwen2.5-vl-7b）
├── data/
│   ├── processed/               # 训练数据（6302 条 SFT）
│   │   ├── sft_single.jsonl     # 单轮诊断 2731 条
│   │   ├── sft_debate.jsonl     # 辩论数据 1509 条
│   │   └── sft_multiturn.jsonl  # 多轮问诊 2062 条
│   ├── chromadb/                # RAG 向量库（44 条植保知识）
│   └── raw/PlantVillage/        # PlantVillage 原始数据集（56K 图片）
├── checkpoints/                 # LoRA Adapter 检查点
│   ├── agrimind-qlora-v1/       # V1（5047 条，rank=16, 3 epoch）
│   └── agrimind-qlora-v2/       # V2（6302 条，rank=16, 3 epoch）
├── test_images/                 # 50 张测试图片（24 个类别）
├── demo/                        # Gradio 演示脚本
├── docs/                        # 方案书、PPT 提纲、设计文档
└── docker-compose.yml           # Docker 编排文件
```

---

## 复现实验

### 训练数据生成

```bash
# 1. 单轮诊断数据
python training/scripts/generate_instructions.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_single.jsonl --use-local

# 2. 辩论数据
python training/scripts/generate_debate.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_debate.jsonl --use-local

# 3. 多轮问诊数据（推荐 n=2 一致性过滤）
python training/scripts/generate_multiturn.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_multiturn.jsonl \
    --use-local --consistency-runs 2 --max-samples 2000
```

### QLoRA 训练

```bash
# 编辑 configs/qlora_qwen_vl.yaml 配置模型路径和输出目录
python training/train_qlora.py --config training/configs/qlora_qwen_vl.yaml

# 合并 Adapter → 完整模型
python training/merge_adapter.py \
    --base-model models/qwen2.5-vl-7b \
    --adapter checkpoints/agrimind-qlora-v2/checkpoint-final \
    --output models/agrimind-v2
```

### Benchmark + 消融实验

```bash
# 生成 AgriReason（800 样本）
python benchmark/generate_agrireason.py \
    --input data/processed/unified_dataset.jsonl \
    --output benchmark/data/agrireason_v1.json \
    --use-local --max-samples 800 --consistency-runs 2

# 运行完整消融实验（6 配置）
python benchmark/ablation.py \
    --benchmark benchmark/data/agrireason_v1.json \
    --base-model models/qwen2.5-vl-7b \
    --finetuned-model models/agrimind-v2 \
    --output-dir benchmark/results

# 语义准确率评估
python benchmark/semantic_eval.py
```

---

## 预期性能

在 AgriReason-800 基准上，`agrimind-v2` 的性能：

| 指标 | B4（微调直接推理） | O1（+DDP 辩论） |
|---|---|---|
| 标签准确率 | 42.1% | 45.1% |
| 语义准确率 (BGE-M3) | ~45% | ~56% |
| 校准误差 (ECE↓) | 0.43 | 0.30 |
| 鉴别覆盖率 | 61.4% | 70.4% |
| 推理质量 (ROUGE-L) | 0.26 | 0.17 |

> 注：开放域推理诊断不同于闭集分类，45% 精确匹配率已属较高水平。语义准确率更能反映模型实际诊断能力。

## 切换推理模式

系统支持三种推理模式，按优先级自动选择：

| 模式 | 条件 | 说明 |
|---|---|---|
| **API 模式**（默认） | 自动检测 `api_key` | 零模型下载，开箱即用 |
| 本地 GPU 模式 | 无 `api_key` 且 GPU 可用 | 需下载模型权重 |
| 自定义 API | 前端输入自己的 key | 替换默认测试 key |

**使用自己的 API key：**
```bash
# 方式 1：环境变量
export AGRIMIND_API_KEY=your_key
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 方式 2：前端界面勾选"API 模式"后输入
```

**使用本地 GPU 模型：**
```bash
# 清空 API key 环境变量
unset AGRIMIND_API_KEY
# 确保模型在 models/agrimind-v2/
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 测试

```bash
cd backend && python -m pytest tests/ -v
# 预期：112 passed in ~12s
```

---

## 技术栈

- **模型**：Qwen2.5-VL-7B + QLoRA (4-bit NF4)
- **后端**：FastAPI + WebSocket + asyncio
- **前端**：React 18 + TypeScript + TailwindCSS + Framer Motion
- **向量库**：ChromaDB + BGE-M3
- **推理**：PyTorch + Transformers + PEFT
- **部署**：Docker Compose

---

## 参赛信息

- 第28届中国机器人及人工智能大赛 · 人工智能创新赛（2026）
- 团队：禾智 | 西北农林科技大学
