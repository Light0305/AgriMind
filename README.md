# AgriMind — 作物智能会诊系统

**团队：禾智** | 西北农林科技大学 | 第28届中国机器人及人工智能大赛 · 人工智能创新赛

基于多智能体辩论协议（DDP）的可解释农业诊断系统，将作物诊断从图像分类问题重新定义为多模态推理问题。

## 核心创新

| 模块 | 说明 |
|---|---|
| **诊断辩论协议 (DDP)** | 初诊、质疑、仲裁三个 AI 专家结构化 2 轮辩论，输出可解释诊断报告 |
| **主动视觉问诊 (AVD)** | AI 主动评估图片信息充分度，智能引导用户拍摄关键视角（最多 3 轮） |
| **AgriReason 评测基准** | 自建农业多模态推理评测数据集，800 样本 × 5 任务类型 |
| **记忆增强 + 选择性辩论** | 检索相似历史病例注入 Agent，达成共识时自动跳过冗余辩论轮次 |
| **指南锚定仲裁** | RAG 检索植保防治知识，仲裁专家引用权威指南做出裁定 |

## 硬件要求

| 最低配置 | 推荐配置 |
|---|---|
| NVIDIA GPU 8GB 显存 | NVIDIA RTX 4090 24GB |
| 系统内存 32GB | 系统内存 64GB+ |
| 磁盘 30GB | 磁盘 50GB+ |
| CUDA 12.0+ | CUDA 12.4+ |

**支持的 GPU**：RTX 3070/3080/3090/4070/4080/4090, A100, V100 等。仅需 8GB 显存即可运行 4-bit 量化模型。

## 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境
conda create -n agrimind python=3.11 -y
conda activate agrimind

# 安装依赖
pip install torch transformers accelerate bitsandbytes peft
pip install fastapi uvicorn websockets chromadb Pillow
pip install FlagEmbedding qwen-vl-utils
```

### 2. 下载模型

```bash
# 下载预训练模型（二选一）
# 选项 A：微调后的 AgriMind 模型（推荐）
# 放置于 models/agrimind-v2/

# 选项 B：Qwen2.5-VL-7B 基础模型
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5-vl-7b
```

### 3. 初始化知识库

```bash
cd backend
PYTHONPATH=. python app/rag/indexer.py
```

### 4. 启动后端

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. 启动前端（开发模式）

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173` 即可使用。

## Docker 部署（一行启动）

```bash
docker-compose up -d
```

## 项目结构

```
robot/
├── backend/
│   ├── app/
│   │   ├── agents/          # DDP 三 Agent（Proposer/Challenger/Arbiter）
│   │   ├── api/             # FastAPI + WebSocket
│   │   ├── avd/             # 主动视觉问诊引擎
│   │   ├── model/           # VLM 推理封装（4-bit 量化）
│   │   ├── rag/             # ChromaDB 知识检索
│   │   └── retrieval/       # 相似病例检索（ResNet 嵌入）
│   └── tests/
├── frontend/
│   └── src/
│       ├── components/      # 图像上传、辩论查看器、诊断报告等
│       ├── hooks/           # WebSocket + 诊断状态管理
│       └── pages/           # 诊断页 / 对比页 / 实验页
├── benchmark/
│   ├── generate_agrireason.py  # AGRI-REASON 自动生成
│   ├── evaluate.py             # 模型评测
│   ├── ablation.py             # 消融实验编排器
│   ├── baselines/              # ResNet / VLM 基线
│   └── data/                   # agrireason_v1.json (800 样本)
├── training/
│   ├── train_qlora.py          # QLoRA 微调
│   ├── merge_adapter.py        # 合并 LoRA → 完整模型
│   ├── configs/                # 训练配置
│   └── scripts/                # 数据生成脚本
├── models/                     # 模型文件
├── data/
│   ├── processed/              # 训练数据（6302 条）
│   └── chromadb/               # RAG 向量库（44 条知识）
└── checkpoints/                # LoRA adapter 权重
```

## 复现实验

### 训练数据生成

```bash
# 1. 单轮诊断数据（2731 条）
python training/scripts/generate_instructions.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_single.jsonl --use-local

# 2. 辩论数据（1509 条）
python training/scripts/generate_debate.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_debate.jsonl --use-local

# 3. 多轮问诊数据（2000 条，n=2 一致性过滤）
python training/scripts/generate_multiturn.py \
    --input data/processed/unified_dataset.jsonl \
    --output data/processed/sft_multiturn.jsonl \
    --use-local --consistency-runs 2 --max-samples 2000
```

### QLoRA 训练

```bash
python training/train_qlora.py --config training/configs/qlora_qwen_vl.yaml

# 合并 adapter → 完整模型
python training/merge_adapter.py \
    --base-model models/qwen2.5-vl-7b \
    --adapter checkpoints/agrimind-qlora-v2/checkpoint-final \
    --output models/agrimind-v2
```

### Benchmark + 消融实验

```bash
# 生成 AgriReason 评测集（800 样本）
python benchmark/generate_agrireason.py \
    --input data/processed/unified_dataset.jsonl \
    --output benchmark/data/agrireason_v1.json \
    --use-local --max-samples 800 --consistency-runs 2

# 运行消融实验（6 配置）
python benchmark/ablation.py \
    --benchmark benchmark/data/agrireason_v1.json \
    --base-model models/qwen2.5-vl-7b \
    --finetuned-model models/agrimind-v2 \
    --output-dir benchmark/results

# 语义准确率评估
python benchmark/semantic_eval.py \
    --results benchmark/results/O1_finetuned_ddp.json
```

## 预期结果

使用 `agrimind-v2` 在 AgriReason-800 上的性能：

| 指标 | B4（微调直接） | O1（+DDP 辩论） |
|---|---|---|
| 标签准确率 | 42.1% | 45.1% |
| 语义准确率 | ~65% | ~70% |
| 校准误差 (ECE↓) | 0.43 | 0.30 |
| 鉴别覆盖率 | 61.4% | 70.4% |

## 参赛信息

- 第28届中国机器人及人工智能大赛 · 人工智能创新赛（2026）
- 团队：禾智 | 西北农林科技大学
