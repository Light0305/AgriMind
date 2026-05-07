#!/usr/bin/env python3
"""
VLM baseline experiments for AgriReason benchmark (B2, B3).
===========================================================
B2: Zero-shot direct inference (Qwen2.5-VL-7B, 4-bit quantized)
B3: Zero-shot + Chain-of-Thought prompting

Usage:
    # B2 — Zero-shot
    python vlm_baselines.py \
        --benchmark benchmark/data/agrireason_v1.json \
        --model-path models/qwen2.5-vl-7b \
        --mode zero-shot \
        --output benchmark/results/B2_zero_shot.json

    # B3 — CoT
    python vlm_baselines.py \
        --benchmark benchmark/data/agrireason_v1.json \
        --model-path models/qwen2.5-vl-7b \
        --mode cot \
        --output benchmark/results/B3_cot.json

    # B2 via DashScope API
    python vlm_baselines.py \
        --benchmark benchmark/data/agrireason_v1.json \
        --api-key sk-xxx \
        --api-model qwen2.5-vl-7b-instruct \
        --mode zero-shot \
        --output benchmark/results/B2_zero_shot.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 将 benchmark/ 加入路径
_BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BENCH_DIR))

from metrics import compute_all_metrics, compute_metrics_by_type  # noqa: E402

# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
PROMPT_ZERO_SHOT = "请诊断这张图片中的作物病害。给出诊断结论和置信度。\n\n诊断结论格式: \"诊断结论: XXX\"\n置信度格式: \"置信度: XX%\""

PROMPT_COT = """\
请一步步分析这张作物图片：
1. 首先描述你观察到的症状（颜色、形态、分布等具体特征）
2. 分析这些症状可能指向什么病害
3. 排除其他可能性，说明理由
4. 给出最终诊断

最终请明确给出:
诊断结论: XXX
置信度: XX%"""


# ---------------------------------------------------------------------------
# 结果解析 (复用 evaluate.py 的解析逻辑)
# ---------------------------------------------------------------------------
def parse_diagnosis(text: str) -> str:
    """从模型输出中提取诊断结论."""
    patterns = [
        r"诊断结论[：:][\s]*(.+?)(?:\n|$|。|，|,|置信度)",
        r"最终诊断[：:][\s]*(.+?)(?:\n|$|。|，|,|置信度)",
        r"诊断为[：:]*[\s]*(.+?)(?:\n|$|。|，|,)",
        r"确诊为[：:]*[\s]*(.+?)(?:\n|$|。|，|,)",
        r"判断为[：:]*[\s]*(.+?)(?:\n|$|。|，|,)",
        r"结论[：:][\s]*(.+?)(?:\n|$|。|，|,)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()

    last_part = text[-200:]
    for kw in ["诊断", "结论", "判断", "识别"]:
        idx = last_part.rfind(kw)
        if idx != -1:
            snippet = last_part[idx : idx + 50]
            return snippet.strip()

    return text.strip()[-50:]


def parse_confidence(text: str) -> float:
    """从模型输出中提取置信度 (0-1)."""
    patterns = [
        r"置信度[：:]\s*(\d+(?:\.\d+)?)\s*%",
        r"置信度[：:]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:的)?(?:置信|把握|确定|可能)",
        r"确信度[：:]\s*(\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val
    return 0.5


def parse_reasoning_chain(text: str) -> list[str]:
    """从模型输出中提取推理链."""
    # 按编号分段
    numbered = re.split(r"\n\s*\d+\s*[.、)\]]\s*", text)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 3:
        return numbered

    # 按关键词分段
    keywords = ["观察", "分析", "排除", "结论", "对比", "鉴别", "建议", "不确定", "关键", "症状"]
    segments: list[str] = []
    current = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        starts_new = False
        for kw in keywords:
            if stripped.startswith(kw) and len(stripped) > len(kw):
                if stripped[len(kw)] in "：:：":
                    starts_new = True
                    break
        if starts_new and current:
            segments.append(current.strip())
            current = stripped
        else:
            current += ("\n" if current else "") + stripped
    if current:
        segments.append(current.strip())
    if len(segments) >= 3:
        return segments

    # 按句号分割
    sentences = [s.strip() for s in text.split("。") if len(s.strip()) > 10]
    if len(sentences) >= 3:
        return sentences

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


# ---------------------------------------------------------------------------
# 模型接口
# ---------------------------------------------------------------------------
class LocalVLM:
    """本地 Qwen2.5-VL-7B (4-bit 量化, UN-finetuned base model)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        print(f"正在加载基础模型: {self.model_path}")
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as e:
            print(f"错误: 缺少依赖: {e}")
            print("请安装: pip install torch transformers bitsandbytes accelerate qwen-vl-utils")
            sys.exit(1)

        model_name = str(Path(self.model_path))
        # 如果是本地目录, 直接使用; 否则当作 HF hub model ID
        if not Path(model_name).exists():
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print("基础模型加载完成 (4-bit 量化, 未微调)。")

    def generate(self, prompt: str, image_path: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        abs_path = str(Path(image_path).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{abs_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2000,
                temperature=0.1,  # 评测用低温度
                do_sample=False,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        result = self.processor.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return result.strip()


class APIVLM:
    """DashScope API 远程模型 (未微调的基础 VLM)."""

    def __init__(self, api_key: str, model: str = "qwen2.5-vl-7b-instruct"):
        self.api_key = api_key
        self.model = model
        self._request_times: list[float] = []
        self._rpm_limit = 15

    def _rate_limit(self):
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rpm_limit:
            sleep_time = 60 - (now - self._request_times[0]) + 0.5
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_times.append(time.time())

    def generate(self, prompt: str, image_path: str) -> str:
        try:
            from dashscope import MultiModalConversation

            return self._generate_dashscope(prompt, image_path)
        except ImportError:
            return self._generate_openai_compat(prompt, image_path)

    def _generate_dashscope(self, prompt: str, image_path: str) -> str:
        from dashscope import MultiModalConversation

        self._rate_limit()
        abs_path = str(Path(image_path).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{abs_path}"},
                    {"text": prompt},
                ],
            }
        ]
        try:
            response = MultiModalConversation.call(
                model=self.model, messages=messages, api_key=self.api_key,
            )
            return response.output.choices[0].message.content[0]["text"].strip()
        except Exception as e:
            print(f"  ⚠ DashScope API 调用失败: {e}")
            return ""

    def _generate_openai_compat(self, prompt: str, image_path: str) -> str:
        import base64

        from openai import OpenAI

        self._rate_limit()
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"  ⚠ 无法读取图片 {image_path}: {e}")
            return ""

        ext = Path(image_path).suffix.lower()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ OpenAI 兼容 API 调用失败: {e}")
            return ""


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def evaluate_vlm(
    model,
    benchmark: list[dict],
    mode: str,
) -> list[dict]:
    """对 benchmark 进行 VLM 推理 (zero-shot 或 CoT)."""
    prompt_template = PROMPT_COT if mode == "cot" else PROMPT_ZERO_SHOT
    mode_cn = "Chain-of-Thought" if mode == "cot" else "Zero-shot"

    predictions: list[dict] = []
    for sample in tqdm(benchmark, desc=f"VLM 推理 ({mode_cn})", unit="sample"):
        image_path = sample.get("image_path", "")
        context = sample.get("context", "")
        question = sample.get("question", "")

        # 构建完整 prompt, 加入 benchmark 中的上下文
        if mode == "cot":
            prompt = prompt_template
            if context:
                prompt = f"背景信息: {context}\n\n{prompt}"
        else:
            prompt = prompt_template
            if context:
                prompt = f"背景信息: {context}\n\n{prompt}"

        try:
            response = model.generate(prompt, image_path)
        except Exception as e:
            print(f"  ⚠ 推理失败 [{sample['id']}]: {e}")
            response = ""

        diagnosis = parse_diagnosis(response)
        confidence = parse_confidence(response)
        reasoning_chain = parse_reasoning_chain(response) if mode == "cot" else []

        predictions.append({
            "id": sample["id"],
            "diagnosis": diagnosis,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
            "reasoning": response if mode == "cot" else "",
            "raw_response": response,
        })

    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VLM baselines for AgriReason benchmark (B2/B3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", type=str, default="benchmark/data/agrireason_v1.json",
        help="AgriReason benchmark JSON 文件",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="本地 Qwen2.5-VL-7B 基础模型路径 (未微调)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="DashScope API key (也可通过 DASHSCOPE_API_KEY 设置)",
    )
    parser.add_argument(
        "--api-model", type=str, default="qwen2.5-vl-7b-instruct",
        help="API 模型名 (default: qwen2.5-vl-7b-instruct)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["zero-shot", "cot"], default="zero-shot",
        help="推理模式: zero-shot (B2) 或 cot (B3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="评测结果输出路径",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="限制评测样本数 (测试用)",
    )
    args = parser.parse_args()

    # 确定输出路径
    if args.output is None:
        if args.mode == "cot":
            args.output = "benchmark/results/B3_cot.json"
        else:
            args.output = "benchmark/results/B2_zero_shot.json"

    # 确定实验 ID
    exp_id = "B3" if args.mode == "cot" else "B2"
    method = "Qwen2.5-VL + CoT" if args.mode == "cot" else "Qwen2.5-VL (zero-shot)"

    # 验证 benchmark
    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(f"错误: Benchmark 文件不存在: {bench_path}")
        sys.exit(1)

    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if args.max_samples is not None:
        benchmark = benchmark[: args.max_samples]

    print(f"加载 {len(benchmark)} 条 benchmark 样本")

    # 初始化模型
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")

    if args.model_path:
        model = LocalVLM(args.model_path)
        model_name = Path(args.model_path).name
    elif api_key:
        model = APIVLM(api_key=api_key, model=args.api_model)
        model_name = args.api_model
    else:
        print("错误: 请指定 --model-path 或 --api-key")
        sys.exit(1)

    mode_cn = "Chain-of-Thought" if args.mode == "cot" else "Zero-shot"
    print(f"\n实验: {exp_id} — {method}")
    print(f"模型: {model_name} (基础模型, 未微调)")
    print(f"模式: {mode_cn}")

    # 运行评测
    start_time = time.time()
    predictions = evaluate_vlm(model, benchmark, args.mode)
    elapsed = time.time() - start_time

    # 计算指标
    overall = compute_all_metrics(predictions, benchmark)
    by_type = compute_metrics_by_type(predictions, benchmark)

    # 保存结果 (与 evaluate.py 输出格式一致)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "experiment_id": exp_id,
            "method": method,
            "description": (
                f"{'Zero-shot + Chain-of-Thought' if args.mode == 'cot' else 'Zero-shot direct inference'} "
                f"with un-finetuned Qwen2.5-VL-7B (4-bit quantized)"
            ),
            "model": model_name,
            "mode": args.mode,
            "prompt": PROMPT_COT if args.mode == "cot" else PROMPT_ZERO_SHOT,
            "benchmark_file": str(bench_path),
            "n_samples": len(benchmark),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        },
        "overall_metrics": overall,
        "metrics_by_type": by_type,
        "predictions": predictions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'═' * 60}")
    print(f"  {exp_id} — {method} 评测结果")
    print(f"{'═' * 60}")
    print(f"  样本数:       {len(benchmark)}")
    print(f"  Top-1 准确率: {overall['top1_accuracy']:.4f}")
    print(f"  推理质量:     {overall['reasoning_quality']:.4f}")
    print(f"  校准误差:     {overall['calibration_error']:.4f}")
    print(f"  鉴别覆盖率:   {overall['differential_coverage']:.4f}")
    print(f"  耗时:         {elapsed:.1f}s ({elapsed / max(len(benchmark), 1):.2f}s/样本)")
    print(f"{'═' * 60}")

    if by_type:
        print(f"\n  ── 按任务类型 ──")
        header = f"    {'类型':<15}"
        for name in ["准确率", "推理质量", "ECE↓", "鉴别覆盖"]:
            header += f" {name:>10}"
        print(header)
        print(f"    {'─' * 58}")
        for task_type in ["disease", "pest", "differential", "nutrient", "uncertainty"]:
            if task_type not in by_type:
                continue
            m = by_type[task_type]
            row = f"    {task_type:<15}"
            row += f" {m.get('top1_accuracy', 0):>10.4f}"
            row += f" {m.get('reasoning_quality', 0):>10.4f}"
            row += f" {m.get('calibration_error', 0):>10.4f}"
            row += f" {m.get('differential_coverage', 0):>10.4f}"
            print(row)

    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
