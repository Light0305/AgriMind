"""
🌾 AgriMind — 作物智能会诊系统 概念Demo
DDP (Diagnostic Debate Protocol) 零样本多Agent辩论诊断

运行方式:
    python gradio_demo.py
    # 默认端口 7860，可通过 --port 指定
"""

import argparse
import time
from threading import Lock

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

# ──────────────────────────── 配置 ────────────────────────────

MODEL_PATH = "/home/user/work/lch/robot/models/qwen2.5-vl-7b"

GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
)

# ──────────────────────────── System Prompts ────────────────────────────

PROPOSER_SYSTEM = (
    "你是一位资深植物病理学家（初诊专家）。"
    "根据用户提供的作物图片，给出最可能的诊断。"
    "输出：(1)观察到的症状特征 (2)初步诊断 (3)支持证据2-3条 (4)置信度。"
    "用中文回答。"
)

CHALLENGER_SYSTEM = (
    "你是一位严谨的植保审核专家（质疑专家）。"
    "审查初诊专家的诊断，找出漏洞，提出至少一个替代诊断及理由。"
    "如果初诊正确，说明验证了哪些方面。"
    "用中文回答。"
)

ARBITER_SYSTEM = (
    "你是诊断委员会主席（仲裁专家）。"
    "综合双方意见做最终裁定。使用 ✅❌⚠️ 结构化格式输出。\n\n"
    "输出格式：\n"
    "✅ 最终诊断：[病害名称]（置信度：高/中/低）\n"
    "   支持证据：[1] ... [2] ... [3] ...\n\n"
    "❌ 排除诊断1：[病害名称]\n"
    "   排除原因：...\n\n"
    "⚠️ 不确定因素：（如有）\n\n"
    "用中文回答。"
)

# ──────────────────────────── 模型加载 ────────────────────────────

_model = None
_processor = None
_load_lock = Lock()


def get_model_and_processor():
    """懒加载模型和processor，只初始化一次。"""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    with _load_lock:
        if _model is not None:
            return _model, _processor

        print(f"[AgriMind] 正在加载模型: {MODEL_PATH} (4-bit 量化)...")
        t0 = time.time()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        elapsed = time.time() - t0
        print(f"[AgriMind] 模型加载完成，耗时 {elapsed:.1f}s")
        return _model, _processor


# ──────────────────────────── VLM 推理 ────────────────────────────


def vlm_chat(system_prompt: str, user_content: list, history: list[dict] | None = None) -> str:
    """
    调用VLM进行一次推理。

    Args:
        system_prompt: 系统提示词，定义角色
        user_content: 用户消息内容列表，可包含文本和图片
        history: 可选的对话历史（已完成的assistant轮次）
    Returns:
        模型生成的文本
    """
    model, processor = get_model_and_processor()

    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_content})

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs = [item["image"] for msg in messages for item in msg["content"] if isinstance(item, dict) and item.get("type") == "image"]

    inputs = processor(
        text=[text_input],
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **GENERATION_CONFIG)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result


# ──────────────────────────── DDP 辩论流程 ────────────────────────────

SECTION_DIVIDER = "\n\n---\n\n"


def run_ddp(image: Image.Image):
    """
    执行完整的 DDP 两轮辩论，通过 yield 逐步流式输出结果。
    """
    if image is None:
        yield "⚠️ 请先上传一张作物图片，再点击「开始诊断」。"
        return

    image_item = {"type": "image", "image": image}
    output = ""

    # ━━━━━━━━━━ Round 1 — 交锋 ━━━━━━━━━━

    # ── Step 1: Proposer 初诊 ──
    output += "## 🔬 Round 1 — 交锋\n\n"
    output += "### 👨‍⚕️ 初诊专家（Proposer）\n\n"
    output += "*正在分析图片...*"
    yield output

    proposer_r1 = vlm_chat(
        system_prompt=PROPOSER_SYSTEM,
        user_content=[
            image_item,
            {"type": "text", "text": "请仔细观察这张作物图片，给出你的诊断。"},
        ],
    )

    output = output.replace("*正在分析图片...*", proposer_r1)
    yield output

    # ── Step 2: Challenger 质疑 ──
    output += SECTION_DIVIDER
    output += "### 🔍 质疑专家（Challenger）\n\n"
    output += "*正在审查初诊结果...*"
    yield output

    challenger_r1 = vlm_chat(
        system_prompt=CHALLENGER_SYSTEM,
        user_content=[
            image_item,
            {"type": "text", "text": f"以下是初诊专家的诊断结果，请进行审查和质疑：\n\n{proposer_r1}"},
        ],
    )

    output = output.replace("*正在审查初诊结果...*", challenger_r1)
    yield output

    # ━━━━━━━━━━ Round 2 — 裁定 ━━━━━━━━━━

    output += SECTION_DIVIDER
    output += "## ⚖️ Round 2 — 裁定\n\n"

    # ── Step 3: Proposer 回应 ──
    output += "### 👨‍⚕️ 初诊专家（回应质疑）\n\n"
    output += "*正在回应质疑...*"
    yield output

    proposer_r2 = vlm_chat(
        system_prompt=PROPOSER_SYSTEM,
        user_content=[
            image_item,
            {"type": "text", "text": (
                f"你之前的诊断如下：\n\n{proposer_r1}\n\n"
                f"质疑专家提出了以下质疑：\n\n{challenger_r1}\n\n"
                "请回应质疑，必要时修改你的诊断。"
            )},
        ],
    )

    output = output.replace("*正在回应质疑...*", proposer_r2)
    yield output

    # ── Step 4: Arbiter 仲裁 ──
    output += SECTION_DIVIDER
    output += "### 🏛️ 仲裁专家（Arbiter）\n\n"
    output += "*正在综合裁定...*"
    yield output

    full_debate = (
        f"【初诊专家 — 第一轮诊断】\n{proposer_r1}\n\n"
        f"【质疑专家 — 质疑意见】\n{challenger_r1}\n\n"
        f"【初诊专家 — 回应质疑】\n{proposer_r2}"
    )

    arbiter_result = vlm_chat(
        system_prompt=ARBITER_SYSTEM,
        user_content=[
            image_item,
            {"type": "text", "text": f"以下是完整的辩论记录，请做出最终裁定：\n\n{full_debate}"},
        ],
    )

    output = output.replace("*正在综合裁定...*", arbiter_result)
    yield output


# ──────────────────────────── Gradio UI ────────────────────────────

TITLE_HTML = """
<div style="text-align:center; padding: 20px 0 10px 0;">
    <h1 style="margin:0; font-size:2.4em;">🌾 AgriMind — 作物智能会诊系统</h1>
    <p style="color:#555; font-size:1.1em; margin-top:8px;">
        基于 DDP（诊断辩论协议）的多Agent零样本作物病害诊断 · 概念验证Demo
    </p>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center; padding:16px 0 8px 0; color:#888; font-size:0.95em;">
    团队：禾智 &nbsp;|&nbsp; 西北农林科技大学 &nbsp;|&nbsp; 2026
</div>
"""


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="green")

    with gr.Blocks(theme=theme, title="AgriMind — 作物智能会诊系统") as demo:
        gr.HTML(TITLE_HTML)

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                img_input = gr.Image(
                    type="pil",
                    label="📷 上传作物图片",
                    height=400,
                    sources=["upload", "clipboard"],
                )
                submit_btn = gr.Button(
                    "🚀 开始诊断",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=3):
                output_md = gr.Markdown(
                    value="*上传图片并点击「开始诊断」，三位AI专家将展开辩论式会诊。*",
                    label="📋 辩论诊断过程",
                )

        submit_btn.click(fn=run_ddp, inputs=[img_input], outputs=[output_md])

        gr.HTML(FOOTER_HTML)

    return demo


# ──────────────────────────── 入口 ────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgriMind 作物智能会诊系统 Demo")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公开链接")
    args = parser.parse_args()

    app = build_ui()
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
