#!/usr/bin/env python3
"""
generate_multiturn.py — 自动生成多轮问诊 (AVD) 对话数据
=========================================================
从 unified_dataset.jsonl 中按病害类别分组, 为同类别的多张图片
生成多轮追问→补充→诊断对话, 训练模型主动追问的能力.

生成两类样本:
  1. 多轮追问正样本 (~1500): 模型认为信息不足, 追问后给出诊断
  2. 无需追问负样本  (~500):  模型认为信息充分, 直接给出诊断

输出为 Qwen2.5-VL 对话格式 JSONL, 可直接用于 QLoRA 微调.

用法:
  python generate_multiturn.py \
      --input-file data/processed/unified_dataset.jsonl \
      --output data/generated/multiturn_avd.jsonl \
      --backend dashscope --num-samples 100
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_MULTITURN = "你是一位植物病理学家，擅长通过多轮问诊收集证据后做出诊断。当图片信息不足时，你会主动追问，要求用户补充特定角度或部位的照片。"

# Prompt for generating the FIRST turn (initial observation + follow-up request)
GENERATION_PROMPT_TURN1 = """\
你是一位植物病理学家，正在进行远程问诊。用户给你发了一张作物照片（第一张）。
已知这是 {crop} 的照片，最终诊断应为: {label_cn}。

请分析这张照片，然后表示信息不足，提出一个**具体的追问**：
- 要求用户拍摄特定部位或角度（如叶片背面、茎部、根部、果实等）
- 解释为什么需要这个补充信息

要求：
- 描述你在第一张图中观察到的初步症状
- 追问要有针对性，例如"请拍叶片背面——我需要确认孢子堆的颜色"
- 不要现在给出最终诊断
回复用纯文本，不要使用 markdown 格式标记。"""

# Prompt for generating the SECOND turn (after receiving additional photo)
GENERATION_PROMPT_TURN2 = """\
你是一位植物病理学家，正在进行远程问诊。

之前你观察了用户的第一张照片并追问了更多信息。你之前的回复是：
---
{turn1_response}
---

现在用户又发了一张补充照片（同一棵植物的另一个角度/部位）。

已知最终诊断为: {label_cn} ({label_en})

请基于两张照片的综合信息，给出：
1. 从补充照片中新观察到的特征
2. 结合两次观察的完整分析
3. 最终诊断结论及置信度

回复用纯文本，不要使用 markdown 格式标记。"""

# Prompt for "no follow-up needed" negative samples
GENERATION_PROMPT_NEGATIVE = """\
你是一位植物病理学家，正在进行远程问诊。用户给你发了一张作物照片。
已知最终诊断为: {label_cn} ({label_en})

这张照片信息充分，症状特征清晰可见，无需追问。
请直接给出完整的诊断：
1. 先明确说明"图片信息充分"
2. 描述观察到的症状特征
3. 给出诊断结论及置信度

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
        self._rpm_limit = 15

    def _rate_limit(self):
        """Simple sliding-window rate limiter."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rpm_limit:
            sleep_time = 60 - (now - self._request_times[0]) + 0.5
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_times.append(time.time())

    def generate(self, prompt: str, image_path: str) -> str | None:
        """Call VLM with single image + prompt."""
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
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
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

    def generate_with_context(
        self, prompt: str, image_path: str, context_messages: list[dict]
    ) -> str | None:
        """Call VLM with conversation history (for turn 2)."""
        b64 = encode_image_base64(image_path)
        if b64 is None:
            return None
        mime = get_image_mime(image_path)
        self._rate_limit()

        messages = list(context_messages)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
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

    def _run_inference(self, messages: list[dict]) -> str | None:
        """Run inference on a list of messages."""
        import torch
        from qwen_vl_utils import process_vision_info

        try:
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

            generated = output_ids[0][inputs["input_ids"].shape[1] :]
            result = self.processor.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return result.strip()
        except Exception as e:
            print(f"  ⚠ 本地推理失败: {e}")
            return None

    def generate(self, prompt: str, image_path: str) -> str | None:
        """Run local inference with single image + prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._run_inference(messages)

    def generate_with_context(
        self, prompt: str, image_path: str, context_messages: list[dict]
    ) -> str | None:
        """Run local inference with conversation history (for turn 2)."""
        messages = list(context_messages)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return self._run_inference(messages)


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
    """Load already-generated image paths for resume support.

    For multiturn data, we use the FIRST image path as the unique key.
    """
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
                msgs = entry.get("messages", [])
                for msg in msgs:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "image":
                                    done.add(item.get("image", ""))
                        break  # only first user message
    except Exception:
        pass
    return done


def group_by_disease(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by disease label (label_en) for multiturn pairing."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["label_en"]].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Builders: construct Qwen2.5-VL conversation format
# ---------------------------------------------------------------------------
def build_multiturn_record(
    image1_path: str,
    turn1_response: str,
    image2_path: str,
    turn2_response: str,
    source: str,
    label_en: str,
) -> dict:
    """Build multi-turn AVD positive sample."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MULTITURN},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1_path},
                    {"type": "text", "text": "这棵植物怎么了？"},
                ],
            },
            {"role": "assistant", "content": turn1_response},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2_path},
                    {"type": "text", "text": "这是补充拍摄的照片。"},
                ],
            },
            {"role": "assistant", "content": turn2_response},
        ],
        "metadata": {
            "source": source,
            "label": label_en,
            "task_type": "multiturn_avd",
        },
    }


def build_multiturn_3turn_record(
    image1_path: str,
    turn1_response: str,
    image2_path: str,
    turn2_response: str,
    image3_path: str,
    turn3_response: str,
    source: str,
    label_en: str,
) -> dict:
    """Build 3-image multi-turn AVD record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MULTITURN},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image1_path},
                    {"type": "text", "text": "这棵植物怎么了？"},
                ],
            },
            {"role": "assistant", "content": turn1_response},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2_path},
                    {"type": "text", "text": "这是补充拍摄的照片。"},
                ],
            },
            {"role": "assistant", "content": turn2_response},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image3_path},
                    {"type": "text", "text": "又拍了一张，请再看看。"},
                ],
            },
            {"role": "assistant", "content": turn3_response},
        ],
        "metadata": {
            "source": source,
            "label": label_en,
            "task_type": "multiturn_avd",
        },
    }


def build_negative_record(
    image_path: str, response: str, source: str, label_en: str
) -> dict:
    """Build no-follow-up negative sample."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MULTITURN},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "这棵植物怎么了？"},
                ],
            },
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "source": source,
            "label": label_en,
            "task_type": "multiturn_avd_negative",
        },
    }


# ---------------------------------------------------------------------------
# Generation: multiturn positive samples
# ---------------------------------------------------------------------------
def generate_multiturn_positive(
    groups: dict[str, list[dict]],
    backend,
    output_file: str,
    num_samples: int | None,
    min_group_size: int,
) -> int:
    """Generate multi-turn AVD positive samples.

    For each disease class with >= min_group_size images, pick 2-3 images
    and generate a multi-turn dialogue.
    """
    done = load_existing_outputs(output_file)

    # Build pairs/triples from groups
    pairs: list[tuple[list[dict], str]] = []
    for label_en, recs in groups.items():
        if len(recs) < min_group_size:
            continue

        random.shuffle(recs)
        # Create pairs (2 images) and occasional triples (3 images)
        i = 0
        while i + 1 < len(recs):
            if i + 2 < len(recs) and random.random() < 0.3:
                # 30% chance of 3-image dialogue
                pairs.append((recs[i : i + 3], label_en))
                i += 3
            else:
                pairs.append((recs[i : i + 2], label_en))
                i += 2

    random.shuffle(pairs)

    # Filter already-done
    pairs = [
        (imgs, label)
        for imgs, label in pairs
        if imgs[0]["image_path"] not in done
    ]

    if num_samples is not None:
        pairs = pairs[:num_samples]

    print(f"\n多轮追问正样本: 待处理 {len(pairs)} 组 (已完成 {len(done)} 组)")

    generated = 0
    skipped = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for imgs, label_en in tqdm(pairs, desc="生成多轮对话", unit="group"):
            img1 = imgs[0]
            img2 = imgs[1]
            crop = img1.get("crop", "作物")
            label_cn = img1["label_cn"]
            source = img1["source"]

            # --- Turn 1: initial observation + follow-up request ---
            prompt1 = GENERATION_PROMPT_TURN1.format(
                crop=crop, label_cn=label_cn
            )
            turn1_resp = backend.generate(prompt1, img1["image_path"])
            if turn1_resp is None:
                skipped += 1
                continue

            # --- Turn 2: after receiving supplementary photo ---
            prompt2 = GENERATION_PROMPT_TURN2.format(
                turn1_response=turn1_resp, label_cn=label_cn, label_en=label_en
            )

            # Build conversation context for turn 2
            if isinstance(backend, DashScopeBackend):
                b64 = encode_image_base64(img1["image_path"])
                if b64 is None:
                    skipped += 1
                    continue
                mime = get_image_mime(img1["image_path"])
                img_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            else:
                img_content = {
                    "type": "image",
                    "image": f"file://{img1['image_path']}",
                }

            context = [
                {
                    "role": "user",
                    "content": [
                        img_content,
                        {"type": "text", "text": "这棵植物怎么了？"},
                    ],
                },
                {"role": "assistant", "content": turn1_resp},
            ]
            turn2_resp = backend.generate_with_context(
                "这是补充拍摄的照片。请结合之前的分析给出诊断。\n" + prompt2,
                img2["image_path"],
                context,
            )

            if turn2_resp is None:
                skipped += 1
                continue

            # Build record
            if len(imgs) == 3:
                # 3-image dialogue — generate a third turn
                img3 = imgs[2]
                prompt3 = GENERATION_PROMPT_TURN2.format(
                    turn1_response=turn2_resp, label_cn=label_cn, label_en=label_en
                )
                turn3_resp = backend.generate(prompt3, img3["image_path"])
                if turn3_resp is None:
                    # Fall back to 2-turn
                    entry = build_multiturn_record(
                        img1["image_path"],
                        turn1_resp,
                        img2["image_path"],
                        turn2_resp,
                        source,
                        label_en,
                    )
                else:
                    entry = build_multiturn_3turn_record(
                        img1["image_path"],
                        turn1_resp,
                        img2["image_path"],
                        turn2_resp,
                        img3["image_path"],
                        turn3_resp,
                        source,
                        label_en,
                    )
            else:
                entry = build_multiturn_record(
                    img1["image_path"],
                    turn1_resp,
                    img2["image_path"],
                    turn2_resp,
                    source,
                    label_en,
                )

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            generated += 1

    print(f"  正样本: 生成 {generated} 条, 跳过 {skipped} 条")
    return generated


# ---------------------------------------------------------------------------
# Generation: negative samples (no follow-up needed)
# ---------------------------------------------------------------------------
def generate_negative_samples(
    records: list[dict],
    backend,
    output_file: str,
    num_negative: int,
) -> int:
    """Generate negative samples where single image is sufficient."""
    done = load_existing_outputs(output_file)
    pending = [r for r in records if r["image_path"] not in done]

    random.shuffle(pending)
    pending = pending[:num_negative]

    print(f"\n无需追问负样本: 待处理 {len(pending)} 条")

    generated = 0
    skipped = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for rec in tqdm(pending, desc="生成负样本", unit="sample"):
            img_path = rec["image_path"]
            label_cn = rec["label_cn"]
            label_en = rec["label_en"]
            source = rec["source"]

            prompt = GENERATION_PROMPT_NEGATIVE.format(
                label_cn=label_cn, label_en=label_en
            )
            response = backend.generate(prompt, img_path)
            if response is None:
                skipped += 1
                continue

            entry = build_negative_record(img_path, response, source, label_en)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            generated += 1

    print(f"  负样本: 生成 {generated} 条, 跳过 {skipped} 条")
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="自动生成多轮问诊 (AVD) 对话训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # DashScope API (测试 20 条)
  python generate_multiturn.py \\
      --input-file data/processed/unified_dataset.jsonl \\
      --output data/generated/multiturn_avd.jsonl \\
      --backend dashscope --num-samples 20

  # 本地模型
  python generate_multiturn.py \\
      --input-file data/processed/unified_dataset.jsonl \\
      --output data/generated/multiturn_avd.jsonl \\
      --backend local --num-samples 100
""",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/unified_dataset.jsonl",
        help="输入: unified_dataset.jsonl 路径",
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
        "--num-samples",
        type=int,
        default=None,
        help="多轮正样本数量限制 (默认全部可用组)",
    )
    parser.add_argument(
        "--num-negative",
        type=int,
        default=500,
        help="无需追问负样本数量 (default: 500)",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="每个病害类别最少需要的图片数 (default: 2)",
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
    print("多轮问诊 (AVD) 数据生成")
    print("=" * 60)
    print(f"  输入文件:      {args.input_file}")
    print(f"  输出文件:      {args.output}")
    print(f"  后端:          {args.backend}")
    print(f"  正样本限制:    {args.num_samples or '无 (全部)'}")
    print(f"  负样本数量:    {args.num_negative}")
    print(f"  最小分组大小:  {args.min_group_size}")
    print()

    # Load data
    records = load_input_data(args.input_file)
    print(f"加载 {len(records)} 条输入记录")

    # Group by disease
    groups = group_by_disease(records)
    eligible = {k: v for k, v in groups.items() if len(v) >= args.min_group_size}
    print(f"共 {len(groups)} 个病害类别, {len(eligible)} 个类别满足最小分组要求 (>={args.min_group_size})")

    total_pairs = sum(len(v) // 2 for v in eligible.values())
    print(f"预计可生成 ~{total_pairs} 组多轮对话")

    # Initialize backend
    if args.backend == "dashscope":
        backend = DashScopeBackend(model=args.model)
    else:
        backend = LocalBackend()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Generate positive samples
    n_pos = generate_multiturn_positive(
        eligible, backend, args.output, args.num_samples, args.min_group_size
    )

    # Generate negative samples
    n_neg = generate_negative_samples(
        records, backend, args.output, args.num_negative
    )

    # Summary
    print("\n" + "=" * 60)
    print("生成摘要")
    print("=" * 60)
    print(f"  多轮正样本:  {n_pos} 条")
    print(f"  无需追问负样本: {n_neg} 条")
    print(f"  合计:        {n_pos + n_neg} 条")
    print(f"  输出文件:    {args.output}")
    print("\n✓ 生成完成")


if __name__ == "__main__":
    main()
