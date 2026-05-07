#!/usr/bin/env python3
"""
generate_instructions.py — 自动生成指令微调数据 (单轮诊断 + 辩论对话)
====================================================================
利用大模型 (DashScope API 或本地 Qwen2.5-VL) 为每张作物图片生成:
  --mode single : 单轮诊断 + 推理链 (~3000 samples)
  --mode debate  : 三方辩论对话      (~1500 samples)

输出为 Qwen2.5-VL 对话格式 JSONL, 可直接用于 QLoRA 微调.

用法:
  # DashScope API 后端
  python generate_instructions.py \
      --input-file data/processed/unified_dataset.jsonl \
      --output data/generated/single_diagnosis.jsonl \
      --backend dashscope --mode single --num-samples 100

  # 本地模型后端
  python generate_instructions.py \
      --input-file data/processed/unified_dataset.jsonl \
      --output data/generated/debate.jsonl \
      --backend local --mode debate --num-samples 50
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_SINGLE = "你是一位资深植物病理学家，擅长通过图像症状进行精准诊断。请根据图片给出严谨的诊断推理过程。"

USER_SINGLE = "请诊断这张作物图片中的病害。"

GENERATION_PROMPT_SINGLE = """\
你是一位资深植物病理学家。请仔细观察这张作物图片，给出完整的诊断推理过程：
1. 观察到的症状特征（描述你在图片中看到的具体异常）
2. 这些特征指向什么（分析可能的病因）
3. 排除了什么（说明为什么不是其他病害）
4. 最终诊断结论及置信度

已知参考标签: {label_cn} ({label_en})
请**基于图片实际特征**给出专业的中文推理过程，保证结论与参考标签一致，但推理过程要自然、细致，不要直接照搬标签。
回复用纯文本，不要使用 markdown 格式标记。"""

SYSTEM_DEBATE = "你正在参与一场作物病害诊断辩论。你将扮演三位专家的角色，进行严谨的学术讨论。"

USER_DEBATE = "请对这张图片进行辩论式诊断。"

GENERATION_PROMPT_DEBATE = """\
你是三位植物病理学家，正在针对这张作物图片进行辩论式诊断。

请按以下格式生成对话（纯文本，不要用 markdown）:

【初诊专家】根据图像特征，给出初步诊断和依据
【质疑专家】对初诊提出合理质疑，指出可能的替代诊断或遗漏的特征
【仲裁专家】综合双方意见，做出最终裁决

已知参考标签: {label_cn} ({label_en})
要求：
- 初诊专家需描述具体的图像症状特征
- 质疑专家需提出至少一个合理的替代诊断
- 仲裁专家给出带置信度的最终裁决，最终结论应与参考标签一致
- 辩论过程要专业、自然，体现真实学术讨论
回复用纯文本，不要使用 markdown 格式标记。"""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str | None:
    """Read an image file and return base64 encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠ 无法读取图片 {image_path}: {e}")
        return None


def get_image_mime(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# DashScope backend
# ---------------------------------------------------------------------------
class DashScopeBackend:
    """Generate responses via DashScope OpenAI-compatible API."""

    def __init__(self, model: str = "qwen-vl-max"):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError:
            print("错误: 请安装 openai 库: pip install openai")
            sys.exit(1)

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self._request_times: list[float] = []
        self._rpm_limit = 15  # conservative rate limit

    def _rate_limit(self):
        """Simple sliding-window rate limiter."""
        now = time.time()
        # Remove timestamps older than 60s
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rpm_limit:
            sleep_time = 60 - (now - self._request_times[0]) + 0.5
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_times.append(time.time())

    def generate(self, prompt: str, image_path: str) -> str | None:
        """Call VLM with an image and prompt, return text response."""
        b64 = encode_image_base64(image_path)
        if b64 is None:
            return None

        mime = get_image_mime(image_path)
        self._rate_limit()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ API 调用失败: {e}")
            return None


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------
class LocalBackend:
    """Generate responses using local Qwen2.5-VL-7B in 4-bit."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load Qwen2.5-VL-7B-Instruct with 4-bit quantization."""
        print("正在加载本地模型 Qwen2.5-VL-7B-Instruct (4-bit) ...")
        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
                BitsAndBytesConfig,
            )
        except ImportError as e:
            print(f"错误: 缺少依赖: {e}")
            print("请安装: pip install torch transformers bitsandbytes accelerate")
            sys.exit(1)

        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print("模型加载完成。")

    def generate(self, prompt: str, image_path: str) -> str | None:
        """Run local inference with image + prompt."""
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
                    max_new_tokens=1500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )

            # Trim input tokens
            generated = output_ids[0][inputs["input_ids"].shape[1] :]
            result = self.processor.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return result.strip()

        except Exception as e:
            print(f"  ⚠ 本地推理失败: {e}")
            return None


# ---------------------------------------------------------------------------
# Data loading & resume
# ---------------------------------------------------------------------------
def load_input_data(input_file: str) -> list[dict]:
    """Load unified_dataset.jsonl records."""
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_outputs(output_file: str) -> set[str]:
    """Load already-generated image paths for resume support."""
    done = set()
    if not os.path.exists(output_file):
        return done
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Extract image path from the user message
                msgs = entry.get("messages", [])
                for msg in msgs:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "image":
                                    done.add(item.get("image", ""))
                        break
    except Exception:
        pass
    return done


# ---------------------------------------------------------------------------
# Self-consistency filtering
# ---------------------------------------------------------------------------
def extract_diagnosis_label(text: str, label_cn: str, label_en: str) -> str:
    """Extract main diagnosis conclusion from generated text.

    Returns a normalized label string for consistency comparison.
    Heuristic: if the reference label_cn appears in the text, that's the diagnosis.
    Otherwise try to find the last sentence mentioning 诊断/判断.
    """
    if label_cn in text:
        return label_cn
    if label_en.lower().replace("_", " ") in text.lower():
        return label_en
    # Try to extract from 最终诊断 / 判断为 patterns
    for keyword in ["最终诊断", "诊断为", "判断为", "确诊为", "结论"]:
        idx = text.find(keyword)
        if idx != -1:
            # Grab the surrounding context (~50 chars)
            snippet = text[idx : idx + 80]
            return snippet.strip()
    return text[-100:].strip()  # fallback: last 100 chars


def check_consistency(
    backend, prompt: str, image_path: str, label_cn: str, label_en: str, runs: int
) -> tuple[str | None, bool]:
    """Run generation `runs` times and keep only if all agree on diagnosis.

    Returns (best_response, is_consistent).
    """
    responses = []
    labels = []
    for _ in range(runs):
        resp = backend.generate(prompt, image_path)
        if resp is None:
            return None, False
        responses.append(resp)
        labels.append(extract_diagnosis_label(resp, label_cn, label_en))

    # Check if all extracted labels agree
    if len(set(labels)) == 1:
        return responses[0], True  # All agree — return first
    # Majority vote: return the response whose label appears most
    counter = Counter(labels)
    most_common_label, count = counter.most_common(1)[0]
    if count >= runs:
        idx = labels.index(most_common_label)
        return responses[idx], True
    return None, False


# ---------------------------------------------------------------------------
# Builders: construct Qwen2.5-VL conversation format
# ---------------------------------------------------------------------------
def build_single_record(
    image_path: str, response: str, source: str, label_en: str
) -> dict:
    """Build single-turn diagnosis record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_SINGLE},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": USER_SINGLE},
                ],
            },
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "source": source,
            "label": label_en,
            "task_type": "single_diagnosis",
        },
    }


def build_debate_record(
    image_path: str, response: str, source: str, label_en: str
) -> dict:
    """Build debate dialogue record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_DEBATE},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": USER_DEBATE},
                ],
            },
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "source": source,
            "label": label_en,
            "task_type": "debate",
        },
    }


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------
def generate_single(
    records: list[dict],
    backend,
    output_file: str,
    num_samples: int | None,
    consistency_runs: int,
):
    """Generate single-turn diagnosis + reasoning chain data."""
    done = load_existing_outputs(output_file)
    pending = [r for r in records if r["image_path"] not in done]

    if num_samples is not None:
        pending = pending[:num_samples]

    print(f"\n单轮诊断生成: 待处理 {len(pending)} 条 (已完成 {len(done)} 条)")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    generated = 0
    skipped = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for rec in tqdm(pending, desc="生成单轮诊断", unit="sample"):
            img_path = rec["image_path"]
            label_cn = rec["label_cn"]
            label_en = rec["label_en"]
            source = rec["source"]

            prompt = GENERATION_PROMPT_SINGLE.format(
                label_cn=label_cn, label_en=label_en
            )

            if consistency_runs > 1:
                response, consistent = check_consistency(
                    backend, prompt, img_path, label_cn, label_en, consistency_runs
                )
                if not consistent:
                    skipped += 1
                    continue
            else:
                response = backend.generate(prompt, img_path)

            if response is None:
                skipped += 1
                continue

            entry = build_single_record(img_path, response, source, label_en)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            generated += 1

    print(f"完成: 生成 {generated} 条, 跳过 {skipped} 条")
    print(f"输出: {output_file}")


def generate_debate(
    records: list[dict],
    backend,
    output_file: str,
    num_samples: int | None,
    consistency_runs: int,
):
    """Generate debate dialogue data."""
    done = load_existing_outputs(output_file)
    pending = [r for r in records if r["image_path"] not in done]

    if num_samples is not None:
        pending = pending[:num_samples]

    print(f"\n辩论对话生成: 待处理 {len(pending)} 条 (已完成 {len(done)} 条)")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    generated = 0
    skipped = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for rec in tqdm(pending, desc="生成辩论对话", unit="sample"):
            img_path = rec["image_path"]
            label_cn = rec["label_cn"]
            label_en = rec["label_en"]
            source = rec["source"]

            prompt = GENERATION_PROMPT_DEBATE.format(
                label_cn=label_cn, label_en=label_en
            )

            if consistency_runs > 1:
                response, consistent = check_consistency(
                    backend, prompt, img_path, label_cn, label_en, consistency_runs
                )
                if not consistent:
                    skipped += 1
                    continue
            else:
                response = backend.generate(prompt, img_path)

            if response is None:
                skipped += 1
                continue

            entry = build_debate_record(img_path, response, source, label_en)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            generated += 1

    print(f"完成: 生成 {generated} 条, 跳过 {skipped} 条")
    print(f"输出: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="自动生成指令微调数据 (单轮诊断 / 辩论对话)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # DashScope API — 单轮诊断 (测试 10 条)
  python generate_instructions.py \\
      --input-file data/processed/unified_dataset.jsonl \\
      --output data/generated/single_diagnosis.jsonl \\
      --backend dashscope --mode single --num-samples 10

  # 本地模型 — 辩论对话
  python generate_instructions.py \\
      --input-file data/processed/unified_dataset.jsonl \\
      --output data/generated/debate.jsonl \\
      --backend local --mode debate
""",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/unified_dataset.jsonl",
        help="输入: unified_dataset.jsonl 路径 (default: data/processed/unified_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["dashscope", "local"],
        default="dashscope",
        help="推理后端 (default: dashscope)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "debate"],
        required=True,
        help="生成模式: single=单轮诊断, debate=辩论对话",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="限制生成样本数 (用于测试, 默认全部)",
    )
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=1,
        help="自一致性过滤: 对每个样本生成 N 次, 仅保留诊断一致的 (default: 1 = 不过滤)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-vl-max",
        help="DashScope 模型名 (default: qwen-vl-max)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Validate input
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        print("请先运行 prepare_data.py 生成 unified_dataset.jsonl")
        sys.exit(1)

    print("=" * 60)
    print("指令数据自动生成")
    print("=" * 60)
    print(f"  输入文件:    {args.input_file}")
    print(f"  输出文件:    {args.output}")
    print(f"  后端:        {args.backend}")
    print(f"  模式:        {args.mode}")
    print(f"  样本限制:    {args.num_samples or '无 (全部)'}")
    print(f"  一致性检验:  {args.consistency_runs} 次")
    print()

    # Load data
    records = load_input_data(args.input_file)
    print(f"加载 {len(records)} 条输入记录")

    # Shuffle for diversity
    random.shuffle(records)

    # Initialize backend
    if args.backend == "dashscope":
        backend = DashScopeBackend(model=args.model)
    else:
        backend = LocalBackend()

    # Generate
    if args.mode == "single":
        generate_single(
            records, backend, args.output, args.num_samples, args.consistency_runs
        )
    else:
        generate_debate(
            records, backend, args.output, args.num_samples, args.consistency_runs
        )

    print("\n✓ 生成完成")


if __name__ == "__main__":
    main()
