# AgriMind 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 AgriMind 作物智能会诊系统，参加 CRAIAC 人工智能创新赛 + RAICOM CAIP 大模型智能体应用赛

**Architecture:** 基于 Qwen2.5-VL-7B QLoRA 微调的多模态 VLM，通过 DDP 辩论协议（Proposer/Challenger/Arbiter 三Agent）实现可解释诊断，AVD 模块驱动多轮主动问诊，FastAPI 后端通过 WebSocket 实时推送辩论过程到 React 前端

**Tech Stack:** Qwen2.5-VL-7B, QLoRA (PEFT + bitsandbytes), FastAPI, React 18, TailwindCSS, Framer Motion, ChromaDB, BGE-M3, vLLM, Docker

**Server:** 10.131.157.205:22 (user@, RTX 4090 24GB), workdir: /home/user/work/lch/robot

---

## 项目目录结构

```
agrimind/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI 入口
│   │   ├── config.py                  # 配置管理
│   │   ├── schemas.py                 # Pydantic 数据模型
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py              # REST API 路由
│   │   │   └── websocket.py           # WebSocket 辩论推送
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Agent 基类
│   │   │   ├── proposer.py            # 初诊专家
│   │   │   ├── challenger.py          # 质疑专家
│   │   │   ├── arbiter.py             # 仲裁专家
│   │   │   └── ddp.py                 # DDP 辩论编排器
│   │   ├── avd/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py              # AVD 问诊引擎
│   │   │   └── session.py             # 多轮会话管理
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py           # VLM 推理封装
│   │   │   └── grounding.py           # 视觉标注工具
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── indexer.py             # 知识库索引构建
│   │   │   └── retriever.py           # 知识检索
│   │   └── retrieval/
│   │       ├── __init__.py
│   │       └── similar_cases.py       # 相似病例检索
│   ├── tests/
│   │   ├── test_schemas.py
│   │   ├── test_agents.py
│   │   ├── test_ddp.py
│   │   ├── test_avd.py
│   │   └── test_api.py
│   ├── requirements.txt
│   └── Dockerfile
├── training/
│   ├── scripts/
│   │   ├── download_datasets.py       # 数据集下载
│   │   ├── prepare_data.py            # 数据预处理与格式化
│   │   ├── generate_instructions.py   # 指令数据自动生成
│   │   └── generate_multiturn.py      # 多轮对话数据生成
│   ├── train_qlora.py                 # QLoRA 训练主脚本
│   ├── merge_adapter.py              # 合并 LoRA 适配器
│   ├── configs/
│   │   └── qlora_qwen_vl.yaml        # 训练超参配置
│   └── requirements.txt
├── benchmark/
│   ├── generate_agrireason.py         # AgriReason 基准自动生成
│   ├── evaluate.py                    # 模型评测脚本
│   ├── ablation.py                    # 消融实验运行器
│   ├── metrics.py                     # 评测指标计算
│   ├── baselines/
│   │   ├── resnet_baseline.py         # ResNet-50 基线
│   │   └── vlm_baselines.py           # VLM零样本/CoT基线
│   └── results/                       # 实验结果输出
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── pages/
│   │   │   ├── DiagnosisPage.tsx      # 主诊断页面
│   │   │   ├── ComparePage.tsx        # 对比演示页面
│   │   │   └── ExperimentPage.tsx     # 实验数据页面
│   │   ├── components/
│   │   │   ├── ImageUpload.tsx        # 图片上传组件
│   │   │   ├── ChatPanel.tsx          # 问诊对话面板
│   │   │   ├── DebateViewer.tsx       # 辩论可视化组件
│   │   │   ├── DiagnosisReport.tsx    # 诊断报告卡片
│   │   │   ├── ImageAnnotation.tsx    # 病灶标注叠加层
│   │   │   └── SimilarCases.tsx       # 相似病例对比
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts        # WebSocket 连接
│   │   │   └── useDiagnosis.ts        # 诊断流程状态管理
│   │   └── types/
│   │       └── index.ts               # TypeScript 类型定义
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── vite.config.ts
├── demo/                              # Phase 0 概念演示
│   ├── gradio_demo.py                 # Gradio 快速原型
│   └── requirements.txt
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

# Phase 0：校内选拔（5/7 - 5/10，3天）

> 目标：准备校内选拔材料——方案书 + PPT + 概念Demo。概念Demo使用未微调的Qwen2.5-VL零样本推理，仅演示DDP辩论流程。

### Task 0.1: 项目初始化 + 服务器环境搭建

**Files:**
- Create: `.gitignore`
- Create: `demo/requirements.txt`
- Create: `README.md`

- [ ] **Step 1: 初始化项目结构**

```bash
cd d:/Contest/robot
mkdir -p backend/app/{api,agents,avd,model,rag,retrieval}
mkdir -p backend/tests
mkdir -p training/{scripts,configs}
mkdir -p benchmark/{baselines,results}
mkdir -p frontend/src/{pages,components,hooks,types}
mkdir -p demo
```

- [ ] **Step 2: 创建 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
venv/
.venv/

# Model files (too large for git)
*.bin
*.safetensors
*.gguf
*.pt
*.pth
models/
checkpoints/
adapters/

# Data files
data/raw/
data/processed/
*.parquet
*.arrow

# Environment
.env
.env.local

# IDE
.idea/
.vscode/
*.swp

# Node
node_modules/
frontend/dist/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
wandb/
runs/

# Skills (installed separately)
.agents/
```

- [ ] **Step 3: 创建 demo/requirements.txt**

```
gradio>=4.0.0
transformers>=4.45.0
torch>=2.0.0
accelerate>=0.30.0
Pillow>=10.0.0
```

- [ ] **Step 4: SSH到服务器创建工作目录**

```bash
ssh user@10.131.157.205 "mkdir -p /home/user/work/lch/robot && echo 'OK'"
```

- [ ] **Step 5: 在服务器上安装基础环境**

```bash
ssh user@10.131.157.205 << 'REMOTE'
cd /home/user/work/lch/robot
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45.0 accelerate>=0.30.0 gradio>=4.0.0 Pillow>=10.0.0
pip install bitsandbytes>=0.43.0
echo "=== GPU Check ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"
REMOTE
```

Expected: `CUDA: True, GPU: NVIDIA GeForce RTX 4090, VRAM: 24.0GB`

- [ ] **Step 6: 在服务器下载 Qwen2.5-VL-7B 模型（用于概念Demo）**

```bash
ssh user@10.131.157.205 << 'REMOTE'
cd /home/user/work/lch/robot
source venv/bin/activate
pip install modelscope
python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='models/qwen2.5-vl-7b')
print('Download complete')
"
REMOTE
```

注意：模型约15GB，下载可能需要20-40分钟。可在后台运行。

- [ ] **Step 7: Commit**

```bash
git add .gitignore README.md demo/requirements.txt
git commit -m "chore: initialize project structure and gitignore"
```

### Task 0.2: 构建概念 Demo（Gradio + 零样本 DDP）

**Files:**
- Create: `demo/gradio_demo.py`

这个Demo不需要微调模型，用 Qwen2.5-VL-7B 零样本 + DDP prompting 展示概念。

- [ ] **Step 1: 编写 Gradio Demo**

```python
# demo/gradio_demo.py
"""
AgriMind 概念演示 - 使用 Qwen2.5-VL-7B 零样本推理 + DDP 辩论协议
用于校内选拔展示，无需微调模型
"""
import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import time

# ===== 模型加载 =====
MODEL_PATH = "/home/user/work/lch/robot/models/qwen2.5-vl-7b"

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,  # 4-bit量化节省显存
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded!")


def call_vlm(image: Image.Image, system_prompt: str, user_prompt: str) -> str:
    """调用VLM进行推理"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )
    # 截取生成部分
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


# ===== Agent System Prompts =====
PROPOSER_SYSTEM = """你是一位资深植物病理学家（初诊专家）。
你的职责是根据用户提供的作物图片，给出最可能的诊断。

输出格式要求：
1. 观察到的症状特征（具体描述你在图片中看到了什么）
2. 初步诊断（最可能的病害/虫害/缺素名称）
3. 支持证据（列出支持你判断的2-3条关键证据）
4. 置信度（高/中/低）

用中文回答，语言专业但易懂。"""

CHALLENGER_SYSTEM = """你是一位严谨的植保审核专家（质疑专家）。
你的职责是审查初诊专家的诊断结果，找出可能的漏洞，提出替代诊断。

你将收到初诊专家的诊断报告和原始图片。你需要：
1. 指出初诊专家诊断中的薄弱点或可能的误判
2. 提出至少一个替代诊断及其理由
3. 如果你认为初诊正确，也要说明你验证了哪些方面

用中文回答，保持批判性但专业。"""

ARBITER_SYSTEM = """你是诊断委员会主席（仲裁专家）。
你的职责是综合初诊专家和质疑专家的意见，做出最终裁定。

你将收到双方的完整辩论记录和原始图片。你需要输出：

✅ 最终诊断：[病害名称]（置信度：高/中/低）
   支持证据：逐条列出

❌ 排除诊断1：[名称]
   排除原因：...

❌ 排除诊断2：[名称]（如有）
   排除原因：...

⚠️ 不确定因素：（如果信息不足以完全确诊，说明需要什么额外信息）

用中文回答。"""


def run_ddp_diagnosis(image: Image.Image):
    """运行 DDP 诊断辩论协议"""
    if image is None:
        yield "请先上传一张作物图片。"
        return

    # ===== Round 1: Proposer 初诊 =====
    yield "🔬 **初诊专家**正在分析图片...\n\n"
    time.sleep(0.5)

    proposer_response = call_vlm(
        image,
        PROPOSER_SYSTEM,
        "请对这张作物图片进行诊断，给出你的专业判断。"
    )
    yield f"🔬 **初诊专家：**\n\n{proposer_response}\n\n---\n\n⚔️ **质疑专家**正在审查...\n\n"
    time.sleep(0.5)

    # ===== Round 1: Challenger 质疑 =====
    challenger_response = call_vlm(
        image,
        CHALLENGER_SYSTEM,
        f"以下是初诊专家的诊断报告，请审查并提出你的质疑：\n\n{proposer_response}"
    )
    yield (
        f"🔬 **初诊专家：**\n\n{proposer_response}\n\n---\n\n"
        f"⚔️ **质疑专家：**\n\n{challenger_response}\n\n---\n\n"
        f"🏛️ **仲裁专家**正在综合裁定...\n\n"
    )
    time.sleep(0.5)

    # ===== Round 2: Arbiter 裁定 =====
    debate_record = (
        f"初诊专家意见：\n{proposer_response}\n\n"
        f"质疑专家意见：\n{challenger_response}"
    )
    arbiter_response = call_vlm(
        image,
        ARBITER_SYSTEM,
        f"以下是双方的辩论记录，请做出最终裁定：\n\n{debate_record}"
    )
    yield (
        f"🔬 **初诊专家：**\n\n{proposer_response}\n\n---\n\n"
        f"⚔️ **质疑专家：**\n\n{challenger_response}\n\n---\n\n"
        f"🏛️ **仲裁结论：**\n\n{arbiter_response}"
    )


# ===== Gradio UI =====
with gr.Blocks(
    title="AgriMind 作物智能会诊系统",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:
    gr.Markdown(
        """
        # 🌾 AgriMind — 作物智能会诊系统
        ### 基于多智能体辩论协议（DDP）的可解释农业诊断

        上传一张作物病害图片，观看三位AI专家如何通过结构化辩论达成诊断共识。

        > **概念演示版** — 使用 Qwen2.5-VL-7B 零样本推理，最终版本将使用农业领域微调模型
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷 上传作物图片",
                height=400,
            )
            submit_btn = gr.Button("🚀 开始诊断", variant="primary", size="lg")

        with gr.Column(scale=2):
            output = gr.Markdown(
                label="诊断过程",
                value="等待上传图片...",
            )

    submit_btn.click(
        fn=run_ddp_diagnosis,
        inputs=[image_input],
        outputs=[output],
    )

    gr.Markdown(
        """
        ---
        **团队：禾智** | 西北农林科技大学 | 2026
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

- [ ] **Step 2: 上传到服务器并测试**

```bash
scp demo/gradio_demo.py user@10.131.157.205:/home/user/work/lch/robot/demo/
ssh user@10.131.157.205 << 'REMOTE'
cd /home/user/work/lch/robot
source venv/bin/activate
python demo/gradio_demo.py
REMOTE
```

Expected: Gradio 在 `http://10.131.157.205:7860` 启动，上传图片后依次显示三个Agent的辩论过程。

- [ ] **Step 3: 用测试图片验证完整流程**

在浏览器打开 `http://10.131.157.205:7860`，上传一张作物病害图片（可从网上搜索"小麦锈病图片"下载），确认：
1. 初诊专家输出诊断
2. 质疑专家提出质疑
3. 仲裁专家做出最终裁定
4. 整个过程流式输出，逐步显示

- [ ] **Step 4: Commit**

```bash
git add demo/
git commit -m "feat: add concept demo with zero-shot DDP debate via Gradio"
```

### Task 0.3: 撰写校内选拔方案书

**Files:**
- Create: `docs/proposal/方案书.md`

- [ ] **Step 1: 创建方案书目录**

```bash
mkdir -p docs/proposal
```

- [ ] **Step 2: 编写方案书**

方案书内容基于设计文档 `docs/superpowers/specs/2026-05-07-agrimind-design.md`，
按比赛要求格式重新组织。包含以下章节：

1. 项目名称与团队信息
2. 项目背景与问题分析（Spec 第1节）
3. 项目简介（Spec 第2节，300字版本）
4. 核心创新点（Spec 第3节，DDP + AVD + AgriReason）
5. 系统架构设计（Spec 第4节，含架构图）
6. 技术路线（Spec 第5节）
7. 数据方案（Spec 第6节）
8. 实验设计（Spec 第7节）
9. 项目计划与分工（Spec 第10节）
10. 预期成果

- [ ] **Step 3: Commit**

```bash
git add docs/proposal/
git commit -m "docs: add campus selection proposal"
```

### Task 0.4: 制作答辩 PPT 内容提纲

**Files:**
- Create: `docs/proposal/PPT提纲.md`

- [ ] **Step 1: 编写PPT内容提纲**

PPT 约 12-15 页：

```markdown
# AgriMind PPT 提纲

## P1: 封面
- AgriMind — 作物智能会诊系统
- 团队：禾智 | 西北农林科技大学

## P2: 痛点
- 现有农业AI的三大局限（被动、不可解释、过度自信）
- 配图：一张图出一个标签 vs 专家问诊流程对比

## P3: 我们的解决思路
- 重新定义：图像分类 → 多模态推理
- 核心理念：让AI像专家团队一样"会诊"

## P4: 创新点1 - DDP辩论协议
- 三个Agent角色图解
- 辩论2轮流程动画示意

## P5: 创新点2 - AVD主动问诊
- 对比：被动分类 vs 主动追问
- 问诊流程示例

## P6: 创新点3 - AgriReason基准
- 与现有数据集对比（只有标签 vs 完整推理链）
- 自动化生成流程

## P7: 系统架构
- 整体架构图（Spec中的架构图美化版）

## P8: 技术路线
- Qwen2.5-VL + QLoRA + FastAPI + React
- 全国产、全开源、零成本

## P9: Demo截图/录屏
- Gradio概念Demo运行截图
- 辩论过程展示

## P10: 实验设计
- 消融实验对比表
- 预期结果趋势

## P11: 项目计划
- 时间线甘特图
- 三人分工表

## P12: 预期成果
- 完整系统 + AgriReason数据集 + 技术论文
- 商业化/推广潜力

## P13: 团队介绍
- 三人简介 + 技术栈 + 指导教师
```

- [ ] **Step 2: Commit**

```bash
git add docs/proposal/
git commit -m "docs: add PPT outline for campus selection"
```

---

# Phase 1：基础搭建（5/11 - 5/25，2周）

> 目标：VLM微调完成 + DDP后端可运行 + 前端骨架完成

### Task 1.1: 数据集下载与预处理（成员A，服务器）

**Files:**
- Create: `training/scripts/download_datasets.py`
- Create: `training/scripts/prepare_data.py`
- Create: `training/requirements.txt`

- [ ] **Step 1: 创建训练环境依赖文件**

```
# training/requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.45.0
peft>=0.12.0
bitsandbytes>=0.43.0
accelerate>=0.30.0
datasets>=2.20.0
Pillow>=10.0.0
tqdm>=4.60.0
wandb>=0.17.0
modelscope>=1.17.0
```

- [ ] **Step 2: 编写数据集下载脚本**

`training/scripts/download_datasets.py` — 自动下载 PlantVillage, PlantDoc, IP102 数据集到 `data/raw/` 目录。

- PlantVillage: 通过 HuggingFace datasets 或 Kaggle 下载
- PlantDoc: GitHub 仓库 clone
- IP102: 官方下载链接

- [ ] **Step 3: 编写数据预处理脚本**

`training/scripts/prepare_data.py` — 将各数据集统一格式化为：

```json
{"image": "path/to/image.jpg", "label": "Tomato___Late_blight", "label_cn": "番茄晚疫病", "source": "PlantVillage"}
```

输出到 `data/processed/unified_dataset.jsonl`

- [ ] **Step 4: 在服务器运行下载和预处理**

```bash
ssh user@10.131.157.205
cd /home/user/work/lch/robot
source venv/bin/activate
pip install -r training/requirements.txt
python training/scripts/download_datasets.py
python training/scripts/prepare_data.py
```

- [ ] **Step 5: Commit**

```bash
git add training/
git commit -m "feat: add dataset download and preprocessing pipeline"
```

### Task 1.2: 指令数据自动生成（成员A，服务器）

**Files:**
- Create: `training/scripts/generate_instructions.py`
- Create: `training/scripts/generate_multiturn.py`

- [ ] **Step 1: 编写单轮诊断指令生成脚本**

`training/scripts/generate_instructions.py` — 读取 `data/processed/unified_dataset.jsonl`，
对每张图片调用大模型（DashScope API / 本地7B）生成诊断推理链。
输出格式符合 Qwen2.5-VL 对话模板。自一致性过滤：每张图跑3次，一致的保留。

- [ ] **Step 2: 编写多轮问诊对话生成脚本**

`training/scripts/generate_multiturn.py` — 从同一病害类别中配对2-3张图片，
生成多轮AVD问诊对话数据。同时生成"信息充分无需追问"的负样本。

- [ ] **Step 3: 编写辩论对话生成脚本**

在 `generate_instructions.py` 中增加辩论对话生成模式，
模拟 Proposer → Challenger → Arbiter 的三方对话。

- [ ] **Step 4: 在服务器批量运行生成**

```bash
# 如使用DashScope API
export DASHSCOPE_API_KEY="your_key_here"
python training/scripts/generate_instructions.py --mode single --output data/processed/sft_single.jsonl
python training/scripts/generate_instructions.py --mode debate --output data/processed/sft_debate.jsonl
python training/scripts/generate_multiturn.py --output data/processed/sft_multiturn.jsonl
```

预期产出：
- `sft_single.jsonl`: ~3000 条
- `sft_debate.jsonl`: ~1500 条
- `sft_multiturn.jsonl`: ~1500 条

- [ ] **Step 5: Commit**

```bash
git add training/scripts/
git commit -m "feat: add instruction data generation pipeline (single + debate + multiturn)"
```

### Task 1.3: QLoRA 微调训练（成员A，服务器）

**Files:**
- Create: `training/configs/qlora_qwen_vl.yaml`
- Create: `training/train_qlora.py`
- Create: `training/merge_adapter.py`

- [ ] **Step 1: 编写训练配置**

```yaml
# training/configs/qlora_qwen_vl.yaml
model:
  name_or_path: /home/user/work/lch/robot/models/qwen2.5-vl-7b
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.05

training:
  output_dir: checkpoints/agrimind-qlora
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  warmup_ratio: 0.1
  lr_scheduler_type: cosine
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  save_steps: 200
  max_seq_length: 2048

data:
  train_files:
    - data/processed/sft_single.jsonl
    - data/processed/sft_debate.jsonl
    - data/processed/sft_multiturn.jsonl
```

- [ ] **Step 2: 编写训练主脚本**

`training/train_qlora.py` — 加载 4-bit 量化 Qwen2.5-VL-7B，
添加 LoRA 适配器，用生成的指令数据进行 SFT 训练。
使用 gradient checkpointing 适配 4090 24GB 显存。

- [ ] **Step 3: 编写适配器合并脚本**

`training/merge_adapter.py` — 训练完成后，将 LoRA 适配器合并到基座模型，
输出完整模型到 `models/agrimind-v1/`。

- [ ] **Step 4: 在服务器启动训练**

```bash
ssh user@10.131.157.205
cd /home/user/work/lch/robot
source venv/bin/activate
python training/train_qlora.py --config training/configs/qlora_qwen_vl.yaml
# 预计 12-16 小时，可用 nohup 或 tmux 后台运行
```

- [ ] **Step 5: 训练完成后合并适配器**

```bash
python training/merge_adapter.py \
  --base_model models/qwen2.5-vl-7b \
  --adapter checkpoints/agrimind-qlora \
  --output models/agrimind-v1
```

- [ ] **Step 6: Commit**

```bash
git add training/
git commit -m "feat: add QLoRA training pipeline for Qwen2.5-VL-7B"
```

### Task 1.4: DDP 辩论引擎后端（成员B，本地）

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/app/config.py`
- Create: `backend/app/agents/base.py`
- Create: `backend/app/agents/proposer.py`
- Create: `backend/app/agents/challenger.py`
- Create: `backend/app/agents/arbiter.py`
- Create: `backend/app/agents/ddp.py`
- Create: `backend/app/model/inference.py`
- Test: `backend/tests/test_schemas.py`
- Test: `backend/tests/test_agents.py`
- Test: `backend/tests/test_ddp.py`

- [ ] **Step 1: 定义数据模型 (schemas.py)**

定义 `DiagnosisContext`, `AgentMessage`, `DebateResult`, `DiagnosisReport` 等 Pydantic 模型，
与 Spec 4.3 节的接口定义一致。

- [ ] **Step 2: 编写 VLM 推理封装 (model/inference.py)**

封装 Qwen2.5-VL 的推理调用，支持：
- 单图推理
- 多图推理（AVD场景）
- visual grounding 输出解析
- 4-bit 量化加载

提供 `VLMInference` 类，所有 Agent 共享同一个模型实例。

- [ ] **Step 3: 实现 Agent 基类 (agents/base.py)**

```python
class BaseAgent:
    def __init__(self, vlm: VLMInference, system_prompt: str): ...
    async def run(self, images: list[Image], context: str) -> AgentMessage: ...
```

- [ ] **Step 4: 实现三个具体 Agent**

- `proposer.py`: Proposer Agent，system prompt 中强调"给出诊断+证据"
- `challenger.py`: Challenger Agent，system prompt 中强调"质疑+替代诊断"
- `arbiter.py`: Arbiter Agent，system prompt 中强调"综合裁定+结构化输出"

每个Agent继承 BaseAgent，仅覆写 system_prompt 和输出解析逻辑。

- [ ] **Step 5: 实现 DDP 编排器 (agents/ddp.py)**

```python
class DDPOrchestrator:
    async def run_debate(self, context: DiagnosisContext) -> DebateResult:
        # Round 1: Proposer → Challenger
        # Round 2: Proposer回应 → Arbiter裁定
        # 每步通过回调函数通知前端（用于WebSocket推送）
```

- [ ] **Step 6: 编写单元测试**

- `test_schemas.py`: 测试数据模型序列化/反序列化
- `test_agents.py`: Mock VLM，测试各 Agent 的输出格式解析
- `test_ddp.py`: Mock Agents，测试辩论流程编排

- [ ] **Step 7: 运行测试**

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

- [ ] **Step 8: Commit**

```bash
git add backend/
git commit -m "feat: implement DDP debate engine with 3 agents"
```

### Task 1.5: FastAPI 后端 + WebSocket（成员B，本地）

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/api/routes.py`
- Create: `backend/app/api/websocket.py`
- Create: `backend/requirements.txt`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: 创建 backend/requirements.txt**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
websockets>=12.0
python-multipart>=0.0.9
Pillow>=10.0.0
pydantic>=2.0.0
torch>=2.0.0
transformers>=4.45.0
peft>=0.12.0
bitsandbytes>=0.43.0
accelerate>=0.30.0
chromadb>=0.5.0
FlagEmbedding>=1.2.0
```

- [ ] **Step 2: 实现 FastAPI 主入口**

`backend/app/main.py` — FastAPI app 初始化，加载模型，注册路由。
设置 CORS 允许前端跨域访问。

- [ ] **Step 3: 实现 REST API 路由**

`backend/app/api/routes.py`:
- `POST /api/diagnose` — 接收图片，启动诊断流程，返回诊断ID
- `GET /api/diagnose/{id}` — 查询诊断结果
- `POST /api/upload` — 图片上传（AVD追问时补充图片用）

- [ ] **Step 4: 实现 WebSocket 端点**

`backend/app/api/websocket.py`:
- `WS /ws/diagnose/{id}` — 实时推送辩论过程（Proposer发言 → Challenger发言 → Arbiter裁定）
- 每个Agent发言完成后推送一条消息，前端实时渲染

- [ ] **Step 5: 编写API测试**

`backend/tests/test_api.py`: 测试图片上传、诊断接口、WebSocket连接。

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat: add FastAPI backend with REST and WebSocket endpoints"
```

### Task 1.6: React 前端骨架（成员C，本地）

**Files:**
- Create: `frontend/` 完整初始化

- [ ] **Step 1: 初始化 React 项目**

```bash
cd d:/Contest/robot
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite framer-motion recharts react-router-dom
npm install -D @types/react @types/react-dom
```

- [ ] **Step 2: 配置 TailwindCSS**

配置 `tailwind.config.ts` 和 `src/index.css`。

- [ ] **Step 3: 创建路由和页面骨架**

- `App.tsx`: React Router 配置，三个页面路由
- `pages/DiagnosisPage.tsx`: 主诊断页面骨架（左图右聊天布局）
- `pages/ComparePage.tsx`: 对比演示页面骨架（左右分屏）
- `pages/ExperimentPage.tsx`: 实验数据页面骨架

- [ ] **Step 4: 创建核心组件骨架**

- `components/ImageUpload.tsx`: 图片上传组件（拖拽+点击）
- `components/ChatPanel.tsx`: 对话面板（消息列表+输入框）
- `components/DebateViewer.tsx`: 辩论可视化（三个Agent的发言气泡）

- [ ] **Step 5: 创建 WebSocket Hook**

`hooks/useWebSocket.ts`: 封装 WebSocket 连接管理，自动重连，消息解析。

- [ ] **Step 6: 创建类型定义**

`types/index.ts`: 与后端 schemas.py 对应的 TypeScript 类型定义。

- [ ] **Step 7: 验证本地开发环境**

```bash
cd frontend
npm run dev
```

Expected: 浏览器打开 `http://localhost:5173`，看到三个页面的骨架。

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat: initialize React frontend with routing and component skeletons"
```

---

# Phase 2：核心开发（5/26 - 6/15，3周）

> 目标：全系统联调可运行 + AgriReason 生成完成 + 消融实验完成

### Task 2.1: AVD 主动问诊模块（成员B）

**Files:**
- Create: `backend/app/avd/engine.py`
- Create: `backend/app/avd/session.py`
- Test: `backend/tests/test_avd.py`

实现 AVD 问诊引擎：
- 信息充分度评估（VLM判断当前图片是否足以确诊）
- 针对性追问生成（基于候选诊断列表生成追问指令）
- 多轮会话管理（跟踪已收集的图片和对话历史）
- 退出条件判断（充分度阈值 / 用户说"没有了" / 最多3轮）

将 AVD 与 DDP 串联：AVD 收集完证据后自动触发 DDP 辩论。

### Task 2.2: RAG 知识检索系统（成员B）

**Files:**
- Create: `backend/app/rag/indexer.py`
- Create: `backend/app/rag/retriever.py`
- Test: `backend/tests/test_rag.py`

实现步骤：
1. 收集植保手册/农药指南 PDF → 切分段落 → BGE-M3 嵌入 → 存入 ChromaDB
2. 诊断完成后，用诊断结果检索相关防治知识
3. 将检索内容注入 Arbiter 的 prompt，生成处置建议

### Task 2.3: 相似病例检索（成员B）

**Files:**
- Create: `backend/app/retrieval/similar_cases.py`

实现步骤：
1. 用微调后的 VLM 提取所有训练图片的视觉特征
2. 存入 ChromaDB 向量库（与RAG知识库分开的 collection）
3. 诊断时，用当前图片特征检索最相似的已知案例
4. 返回 Top-3 相似案例的图片、标签、相似度分数

### Task 2.4: AgriReason Benchmark 自动生成（成员A）

**Files:**
- Create: `benchmark/generate_agrireason.py`
- Create: `benchmark/metrics.py`

实现步骤：
1. 从预处理后的数据集中采样 1000 张图片（按任务类型分层采样）
2. 调用大模型生成推理链（3次自一致性）
3. 自动格式校验 + 过滤
4. 输出 `benchmark/data/agrireason_v1.json`

### Task 2.5: 消融实验（成员A）

**Files:**
- Create: `benchmark/baselines/resnet_baseline.py`
- Create: `benchmark/baselines/vlm_baselines.py`
- Create: `benchmark/ablation.py`
- Create: `benchmark/evaluate.py`

运行 6 个实验配置（B1-B4, O1-O2），计算 4 个评测指标。
输出结果到 `benchmark/results/ablation_results.json`。

### Task 2.6: 前端辩论可视化（成员C）

**Files:**
- Update: `frontend/src/components/DebateViewer.tsx`
- Update: `frontend/src/components/ChatPanel.tsx`
- Create: `frontend/src/components/DiagnosisReport.tsx`
- Create: `frontend/src/components/ImageAnnotation.tsx`
- Create: `frontend/src/components/SimilarCases.tsx`

实现步骤：
1. DebateViewer: Agent 发言逐条出现（Framer Motion 动画），带角色图标和证据标签
2. ChatPanel: AVD 多轮对话交互，支持图片追加上传
3. DiagnosisReport: 最终报告卡片（诊断结果 + 排除项 + 推理链 + 处置建议）
4. ImageAnnotation: 在图片上叠加 bounding box 标注（SVG overlay）
5. SimilarCases: 并排展示相似病例对比

### Task 2.7: 前后端联调（成员B+C）

将前端 WebSocket 连接到后端，测试完整流程：
上传图片 → AVD追问 → DDP辩论实时推送 → 报告渲染

---

# Phase 3：联调优化（6/16 - 6/30，2周）

> 目标：全系统打磨完毕 + 答辩材料就绪

### Task 3.1: 对比演示页面（成员C）

实现 ComparePage：左右分屏，左边零样本基线结果，右边 AgriMind 完整流程。
预置 3-5 个经典 case（模型误诊→DDP纠正、信息不足→AVD追问确诊、高相似鉴别诊断）。

### Task 3.2: 实验数据页面（成员C）

实现 ExperimentPage：
- AgriReason 数据分布饼图（Recharts）
- 消融实验柱状图
- 性能雷达图
- 不确定性校准曲线

### Task 3.3: PDF 报告导出（成员C）

在 DiagnosisReport 组件中添加"导出PDF"按钮，
使用 react-to-print 或后端 weasyprint 生成诊断报告 PDF。

### Task 3.4: 模型优化（成员A）

1. 分析消融实验结果，找出薄弱任务类型
2. 针对性增加弱项的训练数据
3. 调整 LoRA rank / 学习率 / 数据配比
4. 重新训练并评测，直到指标满意

### Task 3.5: 系统性能优化（成员B）

1. 使用 vLLM 替换 Transformers pipeline 加速推理
2. 优化 WebSocket 消息格式（减少传输量）
3. 添加推理缓存（相同图片不重复计算）
4. Docker Compose 部署配置

### Task 3.6: 答辩材料制作（全体）

1. 技术报告（基于设计文档 + 实验数据）
2. 答辩 PPT（更新 Phase 0 的 PPT，加入真实Demo截图和实验数据）
3. 演示视频录制（3分钟）
4. RAICOM 平台提交材料

---

# 关键依赖关系

```
Task 0.1 (环境搭建)
  └→ Task 0.2 (概念Demo)    ←── Phase 0 交付
  └→ Task 1.1 (数据下载)
       └→ Task 1.2 (指令生成)
            └→ Task 1.3 (QLoRA训练) ──→ Task 2.4 (Benchmark)
                                        └→ Task 2.5 (消融实验)
Task 1.4 (DDP引擎) ─────────────────→ Task 2.1 (AVD模块)
  └→ Task 1.5 (FastAPI)                 └→ Task 2.2 (RAG)
       └→ Task 2.7 (联调) ──────────→ Phase 3
Task 1.6 (前端骨架) ────────────────→ Task 2.6 (辩论可视化)
                                        └→ Task 2.7 (联调)
```

可并行的任务：
- Task 1.1-1.3（成员A）与 Task 1.4-1.5（成员B）与 Task 1.6（成员C）完全并行
- Task 2.4-2.5（成员A）与 Task 2.1-2.3（成员B）与 Task 2.6（成员C）完全并行
