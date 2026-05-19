# AgriMind —— 作物智能会诊系统

基于多智能体辩论协议（DDP）的可解释作物病害诊断系统。将作物病害诊断从传统图像分类问题重新定义为多模态推理与结构化辩论问题，通过三个 AI 专家的对抗验证机制输出可解释、可追溯的诊断报告。

> **团队：禾智** · 西北农林科技大学

---

## 核心技术

| 模块 | 说明 |
|---|---|
| **诊断辩论协议 (DDP)** | 初诊专家、质疑专家、仲裁专家进行结构化 2 轮辩论，每个诊断结论都经过对抗验证 |
| **主动视觉问诊 (AVD)** | AI 主动评估图片信息充分度，智能引导用户补拍关键视角（最多 3 轮追问） |
| **记忆增强推理** | 检索历史相似病例注入 Agent 上下文，增强诊断依据的可追溯性 |
| **选择性辩论** | 初诊与质疑达成共识时自动跳过冗余辩论轮次，节省约 50% 推理成本 |
| **指南锚定仲裁** | RAG 检索植保防治知识库，仲裁专家引用权威指南做出裁定 |
| **双模式推理** | 支持本地 GPU（4-bit Qwen2.5-VL-7B）和云端 API（DashScope）两种推理模式 |
| **AgriReason 评测基准** | 自建农业多模态推理评测数据集（800 样本 × 5 任务类型），支持消融实验 |

---

## 快速开始

### 环境要求

- **Python** 3.10+
- **Node.js** 18+（Web 界面）
- **GPU**（可选）：NVIDIA GPU 8GB+ 显存（本地推理模式）
- **无 GPU** 也可使用：通过 API 模式调用云端模型

### 一键安装

```bash
# 克隆项目
git clone https://github.com/Light0305/AgriMind.git
cd AgriMind

# Linux / macOS / WSL
bash setup.sh

# Windows（双击运行或命令行执行）
setup.bat
```

安装脚本会自动：
1. 创建 Python 虚拟环境
2. 检测 GPU 并安装对应版本的 PyTorch
3. 安装所有后端和前端依赖
4. 初始化 RAG 知识库

### 启动服务

```bash
# 终端 1：启动后端
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端
cd frontend && npm run dev
```

打开浏览器访问 **http://localhost:5173**

### 推理模式选择

| 模式 | 配置方式 | 说明 |
|---|---|---|
| **API 模式** | 在界面勾选「API 模式」并输入 DashScope API Key | 无需 GPU，无需下载模型，即开即用 |
| **本地 GPU 模式** | 将模型放置在 `models/agrimind-v2/` 目录 | 离线可用，不限调用次数 |

API Key 也可通过环境变量设置：
```bash
export AGRIMIND_API_KEY=sk-your-key-here    # Linux/macOS
set AGRIMIND_API_KEY=sk-your-key-here       # Windows
```

> DashScope API Key 申请地址：https://dashscope.console.aliyun.com/

---

## 下载说明

项目提供三个独立的下载包，托管在 [Hugging Face](https://huggingface.co/LightChuan/AgriMind)：

> 原始仓库：[Hugging Face](https://huggingface.co/LightChuan/AgriMind)

### 1. 代码包（~1.3MB）

包含完整源代码、前端、测试图片（50 张）和文档。使用 API 模式即可运行，无需下载模型。

- **GitHub**: https://github.com/Light0305/AgriMind
- **下载**: https://hf-mirror.com/LightChuan/AgriMind/resolve/main/AgriMind-code.tar.gz

### 2. 完整包（~13GB）

在代码包基础上，附带微调后的 AgriMind-v2 模型权重（Qwen2.5-VL-7B + QLoRA，8.29B 参数），支持本地 GPU 推理。

- **下载**: https://hf-mirror.com/LightChuan/AgriMind/resolve/main/AgriMind-full.tar.gz

下载后直接解压即可，模型文件位于 `models/agrimind-v2/` 目录下：
```bash
tar -xzf AgriMind-full.tar.gz
```

### 3. 附加数据包（~831MB，可选）

包含 PlantVillage 原始图片（56K 张）、训练数据（6302 条 SFT）和 AgriReason 评测数据。仅用于复现实验或重新训练，**不影响系统正常使用**。

- **下载**: https://hf-mirror.com/LightChuan/AgriMind/resolve/main/AgriMind-data.tar.gz

---

## 系统架构

```
用户上传图片
     │
     ▼
┌─────────────────────┐
│  结构化问诊 (4 题)   │  ← 采集种植地区、季节、天气、症状描述
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  主动视觉问诊 (AVD)  │  ← VLM 评估图片充分度，智能追问
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────────┐
│            诊断辩论协议 (DDP)                 │
│                                             │
│  ┌───────────┐  ┌───────────┐  ┌─────────┐ │
│  │ 初诊专家   │→│ 质疑专家   │→│ 仲裁专家 │ │
│  │ Proposer  │  │ Challenger│  │ Arbiter │ │
│  └───────────┘  └───────────┘  └─────────┘ │
│       ↑              ↑             ↑        │
│   相似病例        对抗验证      RAG 知识库   │
└─────────────────────┬───────────────────────┘
                      ▼
              ┌───────────────┐
              │   诊断报告     │  ← 结论 + 证据 + 鉴别诊断 + 防治建议
              └───────────────┘
```

---

## 项目结构

```
AgriMind/
├── backend/                     # FastAPI 后端
│   ├── app/
│   │   ├── agents/              # DDP 三智能体
│   │   │   ├── proposer.py      # 初诊专家
│   │   │   ├── challenger.py    # 质疑专家
│   │   │   ├── arbiter.py       # 仲裁专家
│   │   │   └── ddp.py           # DDP 编排器
│   │   ├── avd/                 # 主动视觉问诊
│   │   │   ├── engine.py        # AVD 评估引擎
│   │   │   └── session.py       # 会话状态管理
│   │   ├── api/
│   │   │   ├── routes.py        # REST API
│   │   │   └── websocket.py     # WebSocket 实时推送
│   │   ├── model/
│   │   │   └── inference.py     # VLM 推理 (本地 + API 双模式)
│   │   ├── rag/                 # RAG 知识检索
│   │   └── retrieval/           # 相似病例检索
│   ├── tests/                   # 单元测试 (112 项)
│   └── requirements.txt
├── frontend/                    # React 前端
│   └── src/
│       ├── components/          # UI 组件
│       ├── hooks/               # 状态管理 Hooks
│       └── pages/               # 页面
├── benchmark/                   # AgriReason 评测框架
│   ├── evaluate.py              # 模型评测
│   ├── ablation.py              # 消融实验 (6 配置)
│   └── baselines/               # 基线方法
├── training/                    # QLoRA 微调
│   ├── train_qlora.py           # 训练脚本
│   ├── merge_adapter.py         # 合并 LoRA 权重
│   └── scripts/                 # 数据生成
├── test_images/                 # 50 张测试图片 (24 类)
├── docs/                        # 技术文档
├── setup.sh                     # Linux/macOS 一键安装
├── setup.bat                    # Windows 一键安装
└── docker-compose.yml           # Docker 部署
```

---

## 硬件要求

### 本地 GPU 模式

| 项目 | 最低配置 | 推荐配置 |
|---|---|---|
| GPU | NVIDIA 8GB 显存 | NVIDIA RTX 4090 24GB |
| 内存 | 16GB | 32GB+ |
| 磁盘 | 30GB | 50GB+ |
| CUDA | 12.0+ | 12.4+ |

> 支持的 GPU：RTX 3060/3070/3080/3090/4060/4070/4080/4090、A100、V100 等。
> 使用 4-bit NF4 量化，仅需 8GB 显存即可运行 7B 模型。

### API 模式

无 GPU 要求。仅需网络连接和有效的 DashScope API Key。

---

## 实验结果

在自建 AgriReason-800 基准上的表现（agrimind-v2）：

| 配置 | 标签准确率 | 语义准确率 | 校准误差 (ECE↓) | 鉴别覆盖率 |
|---|---|---|---|---|
| 基础模型直接推理 | 29.3% | — | 0.52 | — |
| + QLoRA 微调 | 42.1% | ~45% | 0.43 | 61.4% |
| + DDP 辩论 | **45.1%** | **~56%** | **0.30** | **70.4%** |

> 注：开放域推理诊断不同于闭集分类（从 N 个固定类别中选择）。本系统在开放域场景下进行自由文本诊断，45% 的精确匹配率和 56% 的语义准确率已处于较高水平。

### 消融实验

| 编号 | 配置 | 准确率 | 说明 |
|---|---|---|---|
| B1 | Qwen2.5-VL-7B 原始 | 29.3% | 基线 |
| B2 | + Zero-shot CoT | 30.8% | 仅提示工程 |
| B3 | + 单轮 DDP | 33.2% | 未微调 + 辩论 |
| B4 | + QLoRA 微调 | 42.1% | 微调后直接推理 |
| O1 | + 微调 + DDP | **45.1%** | 完整系统 |
| O2 | + 微调 + DDP + RAG | 44.8% | 知识增强（小幅波动） |

---

## 复现实验

### 数据准备

```bash
# 1. 生成单轮诊断训练数据
python training/scripts/generate_instructions.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_single.jsonl

# 2. 生成辩论训练数据
python training/scripts/generate_multiturn.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_debate.jsonl
```

### QLoRA 训练

```bash
# 编辑训练配置
vim training/configs/qlora_qwen_vl.yaml

# 启动训练 (单卡 A100/4090，约 4-6 小时)
python training/train_qlora.py --config training/configs/qlora_qwen_vl.yaml

# 合并 LoRA 权重为完整模型
python training/merge_adapter.py \
    --base-model models/qwen2.5-vl-7b \
    --adapter checkpoints/agrimind-qlora-v2/checkpoint-final \
    --output models/agrimind-v2
```

### 评测

```bash
# 生成 AgriReason 评测数据 (800 样本)
python benchmark/generate_agrireason.py \
    --input data/processed/unified_dataset.jsonl \
    --output benchmark/data/agrireason_v1.json \
    --max-samples 800

# 运行消融实验 (6 配置)
python benchmark/ablation.py \
    --benchmark benchmark/data/agrireason_v1.json \
    --base-model models/qwen2.5-vl-7b \
    --finetuned-model models/agrimind-v2 \
    --output-dir benchmark/results
```

---

## 测试

```bash
cd backend && python -m pytest tests/ -v
# 预期: 112 passed
```

---

## Docker 部署

```bash
docker-compose up -d
# 后端: http://localhost:8000
# 前端: http://localhost:3000
```

---

## 技术栈

| 层级 | 技术 |
|---|---|
| 视觉语言模型 | Qwen2.5-VL-7B + QLoRA (4-bit NF4) |
| 后端 | FastAPI + WebSocket + asyncio |
| 前端 | React 19 + TypeScript + TailwindCSS 4 + Framer Motion |
| 向量数据库 | ChromaDB + BGE-M3 |
| 推理框架 | PyTorch + Transformers + BitsAndBytes |
| 容器化 | Docker Compose |

---

## 许可证

本项目仅用于学术研究和竞赛展示。
