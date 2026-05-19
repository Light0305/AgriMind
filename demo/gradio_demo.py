"""
🌾 AgriMind — 作物智能会诊系统 概念Demo
DDP (Diagnostic Debate Protocol) 零样本多Agent辩论诊断 · 流式输出版

运行方式:
    python gradio_demo.py
    python gradio_demo.py --port 7860 --share
"""

import argparse
import time
from threading import Lock, Thread

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# ──────────────────────────── 配置 ────────────────────────────

MODEL_PATH = "/home/user/work/lch/robot/models/qwen2.5-vl-7b"

GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    min_new_tokens=150,
    temperature=0.7,
    do_sample=True,
)

# ──────────────────────────── System Prompts ────────────────────────────

PROPOSER_SYSTEM = (
    "你是一位资深植物病理学家（初诊专家）。"
    "根据用户提供的作物图片，给出最可能的诊断。"
    "请详细输出以下内容：\n"
    "(1)观察到的症状特征（详细描述你在图片中看到了什么）\n"
    "(2)初步诊断（最可能的病害名称）\n"
    "(3)支持证据（列出2-3条关键证据，每条展开说明）\n"
    "(4)置信度（高/中/低，并说明理由）\n"
    "用中文回答，内容要充分详实。"
)

CHALLENGER_SYSTEM = (
    "你是一位严谨的植保审核专家（质疑专家）。"
    "审查初诊专家的诊断结果，你需要详细完成以下任务：\n"
    "(1)指出初诊专家诊断中的薄弱点或可能的误判\n"
    "(2)提出至少一个替代诊断，并给出详细理由\n"
    "(3)如果初诊正确，详细说明你验证了哪些方面\n"
    "用中文回答，分析要深入、有理有据。"
)

ARBITER_SYSTEM = (
    "你是诊断委员会主席（仲裁专家）。"
    "综合初诊专家和质疑专家双方意见，做出最终裁定。\n\n"
    "请使用以下格式详细输出：\n\n"
    "✅ 最终诊断：[病害名称]（置信度：高/中/低）\n"
    "   支持证据：[1] （详细说明） [2] （详细说明） [3] （详细说明）\n\n"
    "❌ 排除诊断1：[病害名称]\n"
    "   排除原因：（详细解释为什么排除）\n\n"
    "❌ 排除诊断2：[病害名称]（如有）\n"
    "   排除原因：（详细解释）\n\n"
    "⚠️ 不确定因素：（详细说明，如果没有则写\"无\"）\n\n"
    "用中文回答，每一条都要展开论述。"
)

# ──────────────────────────── 模型加载 ────────────────────────────

_model = None
_processor = None
_load_lock = Lock()


def get_model_and_processor():
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
        print(f"[AgriMind] 模型加载完成，耗时 {time.time() - t0:.1f}s")
        return _model, _processor


# ──────────────────────────── VLM 推理（流式） ────────────────────────────

def vlm_chat_stream(system_prompt: str, user_content: list):
    """流式推理——逐 token 产出文本，用于实时显示。"""
    model, processor = get_model_and_processor()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs = [
        item["image"]
        for msg in messages
        for item in msg["content"]
        if isinstance(item, dict) and item.get("type") == "image"
    ]

    inputs = processor(
        text=[text_input],
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {**inputs, **GENERATION_CONFIG, "streamer": streamer}
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial

    thread.join()


# ──────────────────────────── HTML 构建 ────────────────────────────

def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


STEP_DEFS = [
    ("📷", "上传图片", "upload"),
    ("🔬", "初诊分析", "proposer"),
    ("🔍", "质疑审查", "challenger"),
    ("🏛️", "仲裁裁定", "arbiter"),
    ("📋", "诊断报告", "report"),
]


def build_page(active_step: int, steps_data: dict, loading_step: int | None = None) -> str:
    """构建步骤式界面。active_step=当前显示的tab, steps_data=已有内容, loading_step=转圈步骤"""

    # ── 步骤指示器 ──
    pip = []
    for i, (icon, label, _) in enumerate(STEP_DEFS):
        if i in steps_data:
            state = "completed"
        elif i == loading_step:
            state = "active"
        else:
            state = "pending"

        sel = " selected" if i == active_step else ""
        # 上传图片 tab 不可点击
        onclick = f'onclick="switchTab({i})"' if i > 0 and i in steps_data else ""
        check = '<span class="step-check">✓</span>' if state == "completed" and i > 0 else ""

        pip.append(
            f'<div class="step {state}{sel}" {onclick}>'
            f'<span class="step-icon">{icon}</span>'
            f'<span class="step-label">{label}</span>'
            f'{check}</div>'
        )
        if i < len(STEP_DEFS) - 1:
            acls = "completed" if i in steps_data else "pending"
            pip.append(f'<div class="step-arrow {acls}">›</div>')

    pipeline = f'<div class="pipeline">{"".join(pip)}</div>'

    # ── 内容面板 ──
    panels = []
    for i, (icon, label, role) in enumerate(STEP_DEFS):
        disp = "block" if i == active_step else "none"

        if i == loading_step and i not in steps_data:
            inner = (
                '<div class="panel-loading">'
                '<div class="loading-spinner"></div>'
                f'<div class="loading-text">{label}中，请稍候...</div>'
                '</div>'
            )
        elif i in steps_data:
            inner = steps_data[i]
        else:
            inner = '<div class="panel-empty">等待前序步骤完成</div>'

        panels.append(f'<div class="panel" id="panel-{i}" style="display:{disp}">{inner}</div>')

    content = f'<div class="panels">{"".join(panels)}</div>'

    return f'<div class="agrimind-app">{pipeline}{content}</div>'


def agent_html(role_cn: str, color: str, text: str, streaming: bool = False) -> str:
    """单个Agent的输出卡片"""
    escaped = escape_html(text)
    cursor = '<span class="typing-cursor">▌</span>' if streaming else ""
    return (
        f'<div class="agent-result">'
        f'<div class="agent-result-header" style="border-left:4px solid {color};"><strong>{role_cn}</strong></div>'
        f'<div class="agent-result-body">{escaped}{cursor}</div>'
        f'</div>'
    )


def report_html(text: str) -> str:
    escaped = escape_html(text)
    return (
        '<div class="report-card">'
        '<div class="report-title">📋 最终诊断报告</div>'
        f'<div class="report-body">{escaped}</div>'
        '</div>'
    )


# ──────────────────────────── DDP 辩论流程（流式） ────────────────────────────

def run_ddp(image: Image.Image):
    if image is None:
        yield build_page(0, {0: '<div class="panel-empty">📷 请先上传图片</div>'})
        return

    steps_data = {}
    image_item = {"type": "image", "image": image}

    # ━━━━━ Step 1: 初诊专家 ━━━━━
    yield build_page(1, steps_data, loading_step=1)

    proposer_text = ""
    for partial in vlm_chat_stream(
        PROPOSER_SYSTEM,
        [image_item, {"type": "text", "text": "请仔细观察这张作物图片，给出你的诊断。"}],
    ):
        proposer_text = partial
        steps_data[1] = agent_html("🔬 初诊专家 Proposer", "#2563eb", partial, streaming=True)
        yield build_page(1, steps_data)

    # 完成，去掉光标
    steps_data[1] = agent_html("🔬 初诊专家 Proposer", "#2563eb", proposer_text)
    yield build_page(1, steps_data)

    # ━━━━━ Step 2: 质疑专家 ━━━━━
    yield build_page(2, steps_data, loading_step=2)

    challenger_text = ""
    for partial in vlm_chat_stream(
        CHALLENGER_SYSTEM,
        [image_item, {"type": "text", "text": f"以下是初诊专家的诊断结果，请进行审查和质疑：\n\n{proposer_text}"}],
    ):
        challenger_text = partial
        steps_data[2] = agent_html("🔍 质疑专家 Challenger", "#ea580c", partial, streaming=True)
        yield build_page(2, steps_data)

    steps_data[2] = agent_html("🔍 质疑专家 Challenger", "#ea580c", challenger_text)
    yield build_page(2, steps_data)

    # ━━━━━ Step 3: 仲裁（含初诊回应 + 仲裁） ━━━━━
    yield build_page(3, steps_data, loading_step=3)

    # 3a: 初诊回应质疑（流式）
    response_text = ""
    for partial in vlm_chat_stream(
        PROPOSER_SYSTEM,
        [image_item, {"type": "text", "text": (
            f"你之前的诊断如下：\n\n{proposer_text}\n\n"
            f"质疑专家提出了以下质疑：\n\n{challenger_text}\n\n"
            "请回应质疑，必要时修改你的诊断。"
        )}],
    ):
        response_text = partial
        steps_data[3] = agent_html("🔬 初诊专家 回应质疑", "#2563eb", partial, streaming=True)
        yield build_page(3, steps_data)

    response_done = agent_html("🔬 初诊专家 回应质疑", "#2563eb", response_text)

    # 3b: 仲裁专家（流式）
    full_debate = (
        f"【初诊专家 — 第一轮诊断】\n{proposer_text}\n\n"
        f"【质疑专家 — 质疑意见】\n{challenger_text}\n\n"
        f"【初诊专家 — 回应质疑】\n{response_text}"
    )

    arbiter_text = ""
    for partial in vlm_chat_stream(
        ARBITER_SYSTEM,
        [image_item, {"type": "text", "text": f"以下是完整的辩论记录，请做出最终裁定：\n\n{full_debate}"}],
    ):
        arbiter_text = partial
        steps_data[3] = response_done + agent_html("🏛️ 仲裁专家 Arbiter", "#7c3aed", partial, streaming=True)
        yield build_page(3, steps_data)

    steps_data[3] = response_done + agent_html("🏛️ 仲裁专家 Arbiter", "#7c3aed", arbiter_text)
    yield build_page(3, steps_data)

    # ━━━━━ Step 4: 诊断报告 ━━━━━
    steps_data[4] = report_html(arbiter_text)
    yield build_page(4, steps_data)


# ──────────────────────────── CSS ────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700;900&display=swap');

html { overflow-x: hidden !important; }
body, .gradio-container {
    font-family: 'Noto Sans SC', system-ui, sans-serif !important;
    background: #f0f5f0 !important;
    overflow-x: hidden !important;
}
.gradio-container { max-width: 100% !important; }

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 30%, #047857 60%, #059669 100%);
    border-radius: 16px; padding: 36px 44px; margin-bottom: 20px;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(6, 78, 59, 0.3);
}
.hero-banner::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-size: 2.4em; font-weight: 900; color: #fff; margin: 0 0 6px 0; position: relative; z-index: 1; }
.hero-subtitle { font-size: 1.05em; color: rgba(255,255,255,0.75); font-weight: 300; position: relative; z-index: 1; }
.hero-tags { display: flex; gap: 8px; margin-top: 14px; position: relative; z-index: 1; flex-wrap: wrap; }
.hero-tag {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.15);
    color: #d1fae5; padding: 4px 12px; border-radius: 20px; font-size: 0.82em; font-weight: 500;
}

/* Pipeline */
.pipeline {
    display: flex; align-items: center; justify-content: center;
    padding: 14px 16px; margin-bottom: 16px;
    background: white; border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    flex-wrap: wrap; gap: 2px;
}
.step {
    display: flex; align-items: center; gap: 5px;
    padding: 7px 12px; border-radius: 8px;
    font-size: 0.85em; font-weight: 500; color: #9ca3af;
    transition: all 0.2s; user-select: none;
}
.step[onclick] { cursor: pointer; }
.step[onclick]:hover { background: #f3f4f6; }
.step.active { color: #047857; background: #ecfdf5; }
.step.completed { color: #059669; }
.step.selected { box-shadow: 0 0 0 2px #059669; background: #ecfdf5; color: #047857; }
.step-icon { font-size: 1.15em; }
.step-check { font-size: 0.7em; color: #059669; font-weight: 700; }
.step-arrow { color: #d1d5db; font-size: 1.3em; margin: 0 1px; }
.step-arrow.completed { color: #059669; }

/* Panels */
.panels { min-height: 300px; }
.panel {
    background: white; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    padding: 20px 24px; min-height: 280px;
    animation: fadeIn 0.25s ease;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.panel-empty { text-align: center; padding: 80px 20px; color: #9ca3af; font-size: 1.05em; }

/* Agent result */
.agent-result { margin-bottom: 16px; }
.agent-result:last-child { margin-bottom: 0; }
.agent-result-header {
    padding: 10px 16px; font-size: 0.93em;
    background: #f9fafb; border-radius: 8px 8px 0 0;
}
.agent-result-body {
    padding: 16px 20px; font-size: 0.9em; line-height: 1.8; color: #374151;
    border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px;
    min-height: 60px;
}

/* Typing cursor */
.typing-cursor {
    display: inline-block; color: #059669; font-weight: 400;
    animation: blink 0.7s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* Report */
.report-title {
    font-size: 1.15em; font-weight: 700; color: #047857;
    margin-bottom: 14px; padding-bottom: 10px; border-bottom: 2px solid #d1fae5;
}
.report-body { font-size: 0.93em; line-height: 1.8; color: #1f2937; }

/* Loading */
.panel-loading {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 80px 20px; gap: 14px;
}
.loading-spinner {
    width: 36px; height: 36px;
    border: 3px solid #e5e7eb; border-top-color: #059669;
    border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { color: #6b7280; font-size: 0.9em; }

/* Footer */
.footer-bar { text-align: center; padding: 16px; color: #6b7280; font-size: 0.85em; margin-top: 16px; }

/* Button */
button.primary {
    background: linear-gradient(135deg, #047857, #059669) !important;
    border: none !important; box-shadow: 0 4px 14px rgba(5,150,105,0.35) !important;
    font-weight: 700 !important; font-size: 1.05em !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.45) !important;
}
"""

CUSTOM_HEAD = """
<style>@import url("https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700;900&display=swap");</style>
<script>
function switchTab(idx) {
    document.querySelectorAll('.panel').forEach(function(p){ p.style.display='none'; });
    var el = document.getElementById('panel-'+idx);
    if(el) el.style.display='block';
    var steps = document.querySelectorAll('.step');
    steps.forEach(function(s){ s.classList.remove('selected'); });
    if(steps[idx]) steps[idx].classList.add('selected');
}
</script>
"""

# ──────────────────────────── Gradio UI ────────────────────────────

HERO_HTML = """
<div class="hero-banner">
    <div class="hero-title">🌾 AgriMind</div>
    <div class="hero-subtitle">作物智能会诊系统 — 基于诊断辩论协议的多Agent可解释诊断</div>
    <div class="hero-tags">
        <span class="hero-tag">🔬 DDP 诊断辩论协议</span>
        <span class="hero-tag">🤖 Qwen2.5-VL 多模态大模型</span>
        <span class="hero-tag">🎯 多Agent协作推理</span>
        <span class="hero-tag">📋 结构化诊断报告</span>
    </div>
</div>
"""

INITIAL_HTML = """
<div class="agrimind-app">
    <div class="pipeline">
        <div class="step selected"><span class="step-icon">📷</span><span class="step-label">上传图片</span></div>
        <div class="step-arrow pending">›</div>
        <div class="step"><span class="step-icon">🔬</span><span class="step-label">初诊分析</span></div>
        <div class="step-arrow pending">›</div>
        <div class="step"><span class="step-icon">🔍</span><span class="step-label">质疑审查</span></div>
        <div class="step-arrow pending">›</div>
        <div class="step"><span class="step-icon">🏛️</span><span class="step-label">仲裁裁定</span></div>
        <div class="step-arrow pending">›</div>
        <div class="step"><span class="step-icon">📋</span><span class="step-label">诊断报告</span></div>
    </div>
    <div class="panels"><div class="panel">
        <div class="panel-empty">📷 上传作物图片并点击「开始会诊」<br>
        <span style="font-size:0.85em;color:#b0b0b0;">三位AI植保专家将通过结构化辩论达成诊断共识</span></div>
    </div></div>
</div>
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AgriMind — 作物智能会诊系统", fill_width=True) as demo:
        gr.HTML(HERO_HTML)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("#### 📷 上传作物图片")
                img_input = gr.Image(
                    type="pil", label="拖拽或点击上传",
                    height=300, sources=["upload", "clipboard"],
                )
                submit_btn = gr.Button("🚀 开始会诊", variant="primary", size="lg")
                gr.Markdown(
                    "<div style='text-align:center;color:#9ca3af;font-size:0.8em;margin-top:4px;'>"
                    "支持 JPG / PNG · 建议拍摄清晰的病害部位特写</div>"
                )
            with gr.Column(scale=2, min_width=500):
                output_html = gr.HTML(value=INITIAL_HTML)

        submit_btn.click(fn=run_ddp, inputs=[img_input], outputs=[output_html])
        gr.HTML('<div class="footer-bar"><strong>团队：禾智</strong> · 西北农林科技大学 · 2026</div>')

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_ui()
    app.queue()
    app.launch(
        server_name=args.host, server_port=args.port,
        share=args.share, css=CUSTOM_CSS, head=CUSTOM_HEAD,
    )


if __name__ == "__main__":
    main()
