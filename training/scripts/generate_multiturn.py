#!/usr/bin/env python3
"""
generate_multiturn.py — 自动生成多轮问诊 (AVD) 对话数据
=========================================================
从 unified_dataset.jsonl 中按病害类别分组, 为同类别的多张图片
生成多轮追问→补充→诊断对话, 训练模型主动追问的能力.

生成两类样本:
  1. 多轮追问正样本 (~1500): 模型认为信息不足, 追问后给出诊断
  2. 无需追问负样本  (~500):  模型认为信息充分, 直接给出诊断
     (主要从 PlantVillage 的清晰实验室照片中选取)

输出为 Qwen2.5-VL 对话格式 JSONL, 可直接用于 QLoRA 微调.

用法:
  # DashScope API
  python generate_multiturn.py \
      --input data/processed/unified_dataset.jsonl \
      --output data/generated/multiturn_avd.jsonl \
      --api-key sk-xxx --max-samples 100

  # 本地模型
  python generate_multiturn.py \
      --input data/processed/unified_dataset.jsonl \
      --output data/generated/multiturn_avd.jsonl \
      --use-local --max-samples 50
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_MULTITURN = (
    "你是一位植物病理学家，正在进行远程问诊。当图片信息不足以做出确切诊断时，"
    "你会主动追问，要求用户补充特定角度或部位的照片。当信息充分时，你会直接给出诊断。"
)

# Turn 1: initial observation + follow-up request
GENERATION_PROMPT_TURN1 = """\
你是一位植物病理学家，正在进行远程问诊。用户给你发了一张作物照片。
已知这是 {crop} 的照片，最终诊断应为: {label_cn}。

请分析这张照片，然后表示信息不充分，提出一个**具体的追问**：
- 要求用户拍摄特定部位或角度（如叶片背面、茎部基部、根部、果实近距离、病斑边缘等）
- 解释为什么需要这个补充信息

要求：
- 描述你在第一张图中观察到的2-3个初步症状
- 追问要有针对性, 例如"请拍叶片背面——我需要确认是否有孢子堆"
- 说明这个补充信息对确诊的意义
- 不要现在给出最终诊断
回复用纯文本，不要使用 markdown 格式标记。"""

# Turn 2: after receiving additional photo, give final diagnosis
GENERATION_PROMPT_TURN2 = """\
你是一位植物病理学家，正在进行远程问诊。

之前你观察了用户的第一张照片并追问了更多信息。你之前的回复是：
---
{turn1_response}
---

现在用户又发了一张补充照片（同一棵植物的另一个角度/部位）。

已知最终诊断为: {label_cn} ({label_en})

请基于两张照片的综合信息，给出：
1. 从补充照片中新观察到的关键特征
2. 结合两次观察的完整分析
3. 最终诊断结论及置信度

回复用纯文本，不要使用 markdown 格式标记。"""

# Negative: single clear image, no follow-up needed
GENERATION_PROMPT_NEGATIVE = """\
你是一位植物病理学家，正在进行远程问诊。用户给你发了一张作物照片。
已知最终诊断为: {label_cn} ({label_en})

这张照片信息充分、症状特征非常清晰，无需追问补充照片。
请直接给出完整的诊断：
1. 首先明确说明"图片信息充分，可以直接诊断"
2. 描述观察到的关键症状特征（至少3个）
3. 给出诊断结论及置信度

回复用纯文本，不要使用 markdown 格式标记。"""


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠ 无法读取图片 {image_path}: {e}")
        return None


def get_image_mime(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# DashScope backend (native MultiModalConversation API)
# ---------------------------------------------------------------------------
class DashScopeBackend:
    """Generate via DashScope MultiModalConversation API."""

    def __init__(self, api_key: str, model: str = "qwen2.5-vl-72b-instruct"):
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

    def generate(self, prompt: str, image_path: str) -> str | None:
        """Call VLM with single image + prompt."""
        try:
            from dashscope import MultiModalConversation
        except ImportError:
            return self._generate_openai_compat(prompt, image_path)

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
                model=self.model,
                messages=messages,
                api_key=self.api_key,
            )
            return response.output.choices[0].message.content[0]["text"].strip()
        except Exception as e:
            print(f"  ⚠ DashScope API 调用失败: {e}")
            return None

    def generate_with_context(
        self, prompt: str, image_path: str, context_messages: list[dict],
    ) -> str | None:
        """Call VLM with conversation history (for turn 2)."""
        try:
            from dashscope import MultiModalConversation
        except ImportError:
            return self._generate_with_context_openai(
                prompt, image_path, context_messages,
            )

        self._rate_limit()
        abs_path = str(Path(image_path).resolve())

        messages = list(context_messages)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"image": f"file://{abs_path}"},
                    {"text": prompt},
                ],
            }
        )
        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
            )
            return response.output.choices[0].message.content[0]["text"].strip()
        except Exception as e:
            print(f"  ⚠ DashScope API 调用失败: {e}")
            return None

    def _generate_openai_compat(self, prompt: str, image_path: str) -> str | None:
        """Fallback: OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI
        except ImportError:
            print("错误: 请安装 dashscope 或 openai 库")
            sys.exit(1)

        b64 = encode_image_base64(image_path)
        if b64 is None:
            return None
        mime = get_image_mime(image_path)
        self._rate_limit()

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
            print(f"  ⚠ OpenAI 兼容 API 调用失败: {e}")
            return None

    def _generate_with_context_openai(
        self, prompt: str, image_path: str, context_messages: list[dict],
    ) -> str | None:
        """Fallback turn-2 via OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI
        except ImportError:
            print("错误: 请安装 openai 库")
            sys.exit(1)

        b64 = encode_image_base64(image_path)
        if b64 is None:
            return None
        mime = get_image_mime(image_path)
        self._rate_limit()

        # Convert context to OpenAI format
        oai_messages: list[dict] = []
        for msg in context_messages:
            if msg["role"] == "assistant":
                oai_messages.append(
                    {"role": "assistant", "content": msg["content"]}
                )
            elif msg["role"] == "user":
                content_parts = msg.get("content", [])
                oai_parts = []
                for part in content_parts:
                    if "image" in part:
                        # Re-encode from path
                        img_p = part["image"]
                        if img_p.startswith("file://"):
                            img_p = img_p[7:]
                        ctx_b64 = encode_image_base64(img_p)
                        if ctx_b64:
                            ctx_mime = get_image_mime(img_p)
                            oai_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{ctx_mime};base64,{ctx_b64}",
                                },
                            })
                    elif "text" in part:
                        oai_parts.append({"type": "text", "text": part["text"]})
                oai_messages.append({"role": "user", "content": oai_parts})

        # Append current turn
        oai_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        })

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                max_tokens=1500,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ OpenAI 兼容 API 调用失败: {e}")
            return None


# ---------------------------------------------------------------------------
# Local backend (Qwen2.5-VL-7B, 4-bit)
# ---------------------------------------------------------------------------
class LocalBackend:
    """Generate responses using local Qwen2.5-VL-7B in 4-bit."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
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
            print("请安装: pip install torch transformers bitsandbytes accelerate qwen-vl-utils")
            sys.exit(1)

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
        print("模型加载完成。")

    def _run_inference(self, messages: list[dict]) -> str | None:
        import torch
        from qwen_vl_utils import process_vision_info

        try:
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
                    max_new_tokens=1500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            result = self.processor.decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return result.strip()
        except Exception as e:
            print(f"  ⚠ 本地推理失败: {e}")
            return None

    def generate(self, prompt: str, image_path: str) -> str | None:
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
        return self._run_inference(messages)

    def generate_with_context(
        self, prompt: str, image_path: str, context_messages: list[dict],
    ) -> str | None:
        abs_path = str(Path(image_path).resolve())
        messages = list(context_messages)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{abs_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return self._run_inference(messages)


# ---------------------------------------------------------------------------
# Data loading & resume
# ---------------------------------------------------------------------------
def load_input_data(input_file: str) -> list[dict]:
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_outputs(output_file: str) -> set[str]:
    """Load already-generated image paths for resume support.

    Uses the FIRST image path in the first user message as unique key.
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
    """Group records by disease label (label_cn) for multiturn pairing."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["label_cn"]].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Self-consistency filtering for multiturn
# ---------------------------------------------------------------------------
def extract_diagnosis(text: str, label_cn: str, label_en: str) -> str:
    """Extract diagnosis from text for consistency check."""
    if label_cn in text:
        return label_cn
    if label_en.lower().replace("_", " ") in text.lower():
        return label_en
    for kw in ["最终诊断", "诊断为", "判断为", "确诊为", "结论"]:
        idx = text.find(kw)
        if idx != -1:
            return text[idx: idx + 80].strip()
    return text[-100:].strip()


def run_with_consistency(
    gen_func, label_cn: str, label_en: str, n_runs: int = 3,
) -> str | None:
    """Run a generation function n_runs times, return result only if
    the extracted diagnosis agrees across runs.

    gen_func: callable() -> str | None
    """
    responses: list[str] = []
    labels: list[str] = []
    for _ in range(n_runs):
        resp = gen_func()
        if resp is None:
            return None
        responses.append(resp)
        labels.append(extract_diagnosis(resp, label_cn, label_en))

    if len(set(labels)) == 1:
        return responses[0]

    counter = Counter(labels)
    most_common, count = counter.most_common(1)[0]
    if count >= n_runs - 1:
        return responses[labels.index(most_common)]

    return None  # inconsistent


# ---------------------------------------------------------------------------
# Builders: Qwen2.5-VL SFT conversation format
# ---------------------------------------------------------------------------
def build_multiturn_record(
    image1_path: str,
    turn1_response: str,
    image2_path: str,
    turn2_response: str,
    source: str,
    label_en: str,
) -> dict:
    """Build multi-turn AVD positive sample (2 turns)."""
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
                    {"type": "text", "text": "这是背面"},
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


def build_negative_record(
    image_path: str, response: str, source: str, label_en: str,
) -> dict:
    """Build no-follow-up negative sample (信息充分, 直接诊断)."""
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
    max_samples: int | None,
    consistency_runs: int,
) -> int:
    """Generate multi-turn AVD positive samples.

    For each disease class with 2+ images, pair images and generate
    a 2-turn dialogue: initial observation → follow-up request → diagnosis.
    """
    done = load_existing_outputs(output_file)

    # Build image pairs from same-disease groups
    pairs: list[tuple[dict, dict, str]] = []  # (img1_rec, img2_rec, label_cn)
    for label_cn, recs in groups.items():
        if len(recs) < 2:
            continue
        random.shuffle(recs)
        for i in range(0, len(recs) - 1, 2):
            pairs.append((recs[i], recs[i + 1], label_cn))

    random.shuffle(pairs)

    # Filter already-done
    pairs = [
        (a, b, lbl)
        for a, b, lbl in pairs
        if a["image_path"] not in done
    ]

    if max_samples is not None:
        pairs = pairs[:max_samples]

    print(f"\n多轮追问正样本: 待处理 {len(pairs)} 组 (已完成 {len(done)} 组)")

    generated = 0
    skipped = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for img1, img2, label_cn in tqdm(pairs, desc="生成多轮对话", unit="group"):
            crop = img1.get("crop", "作物")
            label_en = img1["label_en"]
            source = img1["source"]

            # --- Turn 1: initial observation + follow-up request ---
            prompt1 = GENERATION_PROMPT_TURN1.format(
                crop=crop, label_cn=label_cn,
            )

            if consistency_runs > 1:
                turn1_resp = run_with_consistency(
                    lambda p=prompt1, ip=img1["image_path"]: backend.generate(p, ip),
                    label_cn, label_en, consistency_runs,
                )
            else:
                turn1_resp = backend.generate(prompt1, img1["image_path"])

            if turn1_resp is None:
                skipped += 1
                continue

            # --- Turn 2: supplementary photo → final diagnosis ---
            prompt2 = GENERATION_PROMPT_TURN2.format(
                turn1_response=turn1_resp,
                label_cn=label_cn,
                label_en=label_en,
            )

            # Build conversation context for turn 2
            abs1 = str(Path(img1["image_path"]).resolve())
            context = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{abs1}"},
                        {"text": "这棵植物怎么了？"},
                    ],
                },
                {"role": "assistant", "content": turn1_resp},
            ]

            full_prompt2 = "这是补充拍摄的照片。\n" + prompt2

            if consistency_runs > 1:
                turn2_resp = run_with_consistency(
                    lambda p=full_prompt2, ip=img2["image_path"], ctx=context: (
                        backend.generate_with_context(p, ip, ctx)
                    ),
                    label_cn, label_en, consistency_runs,
                )
            else:
                turn2_resp = backend.generate_with_context(
                    full_prompt2, img2["image_path"], context,
                )

            if turn2_resp is None:
                skipped += 1
                continue

            entry = build_multiturn_record(
                img1["image_path"], turn1_resp,
                img2["image_path"], turn2_resp,
                source, label_en,
            )
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            generated += 1

    print(f"  正样本: 生成 {generated} 条, 跳过 {skipped} 条")
    return generated


# ---------------------------------------------------------------------------
# Generation: negative samples (PlantVillage clear lab photos)
# ---------------------------------------------------------------------------
def generate_negative_samples(
    records: list[dict],
    backend,
    output_file: str,
    num_negative: int,
    consistency_runs: int,
) -> int:
    """Generate negative samples where a single clear image suffices.

    Prioritizes PlantVillage images (clear lab photos with uniform backgrounds)
    since they are unambiguous and don't need follow-up questions.
    """
    done = load_existing_outputs(output_file)

    # Prioritize PlantVillage (clean lab photos), then others
    pv_records = [r for r in records if r["source"] == "PlantVillage" and r["image_path"] not in done]
    other_records = [r for r in records if r["source"] != "PlantVillage" and r["image_path"] not in done]

    random.shuffle(pv_records)
    random.shuffle(other_records)

    # Take up to 80% from PlantVillage, rest from others
    pv_count = min(len(pv_records), int(num_negative * 0.8))
    other_count = min(len(other_records), num_negative - pv_count)
    pending = pv_records[:pv_count] + other_records[:other_count]

    # If we still need more, fill from whatever is left
    if len(pending) < num_negative:
        remaining = pv_records[pv_count:] + other_records[other_count:]
        pending += remaining[: num_negative - len(pending)]

    random.shuffle(pending)

    print(f"\n无需追问负样本: 待处理 {len(pending)} 条 (PlantVillage 优先)")

    generated = 0
    skipped = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for rec in tqdm(pending, desc="生成负样本", unit="sample"):
            img_path = rec["image_path"]
            label_cn = rec["label_cn"]
            label_en = rec["label_en"]
            source = rec["source"]

            prompt = GENERATION_PROMPT_NEGATIVE.format(
                label_cn=label_cn, label_en=label_en,
            )

            if consistency_runs > 1:
                response = run_with_consistency(
                    lambda p=prompt, ip=img_path: backend.generate(p, ip),
                    label_cn, label_en, consistency_runs,
                )
            else:
                response = backend.generate(prompt, img_path)

            if response is None:
                skipped += 1
                continue

            # Validate: the response should mention "信息充分" or similar
            if not any(kw in response for kw in ["信息充分", "直接诊断", "可以确定", "清晰可见", "特征明显"]):
                # Retry once — the model may not have followed the format
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
      --input data/processed/unified_dataset.jsonl \\
      --output data/generated/multiturn_avd.jsonl \\
      --api-key sk-xxx --max-samples 20

  # 本地 7B 模型
  python generate_multiturn.py \\
      --input data/processed/unified_dataset.jsonl \\
      --output data/generated/multiturn_avd.jsonl \\
      --use-local --max-samples 50
""",
    )
    parser.add_argument(
        "--input",
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
        "--api-key",
        type=str,
        default=None,
        help="DashScope API key (也可通过 DASHSCOPE_API_KEY 环境变量设置)",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        default=False,
        help="使用本地 Qwen2.5-VL-7B 模型 (回退方案, 需要 GPU)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1500,
        help="多轮正样本数量限制 (default: 1500)",
    )
    parser.add_argument(
        "--num-negative",
        type=int,
        default=500,
        help="无需追问负样本数量 (default: 500)",
    )
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=3,
        help="自一致性过滤: 对每个样本生成 N 次, 仅保留诊断一致的 (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-vl-72b-instruct",
        help="DashScope 模型名 (default: qwen2.5-vl-72b-instruct)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve API key
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not args.use_local and not api_key:
        print("错误: 请通过 --api-key 或 DASHSCOPE_API_KEY 环境变量提供 API key")
        print("      或使用 --use-local 回退到本地模型")
        sys.exit(1)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        print("请先运行 prepare_data.py 生成 unified_dataset.jsonl")
        sys.exit(1)

    backend_cn = "本地 7B" if args.use_local else f"DashScope ({args.model})"

    print("=" * 60)
    print("多轮问诊 (AVD) 数据生成")
    print("=" * 60)
    print(f"  输入文件:      {args.input}")
    print(f"  输出文件:      {args.output}")
    print(f"  后端:          {backend_cn}")
    print(f"  正样本限制:    {args.max_samples}")
    print(f"  负样本数量:    {args.num_negative}")
    print(f"  一致性检验:    {args.consistency_runs} 次/样本")
    print()

    # Load data
    records = load_input_data(args.input)
    print(f"加载 {len(records)} 条输入记录")

    # Group by disease
    groups = group_by_disease(records)
    eligible = {k: v for k, v in groups.items() if len(v) >= 2}
    print(
        f"共 {len(groups)} 个病害类别, "
        f"{len(eligible)} 个类别有 2+ 张图片可配对"
    )
    total_pairs = sum(len(v) // 2 for v in eligible.values())
    print(f"预计可生成 ~{total_pairs} 组多轮对话")

    # Initialize backend
    if args.use_local:
        backend = LocalBackend()
    else:
        backend = DashScopeBackend(api_key=api_key, model=args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Generate positive samples (multiturn follow-up)
    n_pos = generate_multiturn_positive(
        eligible, backend, args.output, args.max_samples, args.consistency_runs,
    )

    # Generate negative samples (PlantVillage clear photos → direct diagnosis)
    n_neg = generate_negative_samples(
        records, backend, args.output, args.num_negative, args.consistency_runs,
    )

    # Summary
    print("\n" + "=" * 60)
    print("生成摘要")
    print("=" * 60)
    print(f"  多轮正样本:     {n_pos} 条")
    print(f"  无需追问负样本:  {n_neg} 条")
    print(f"  合计:            {n_pos + n_neg} 条")
    print(f"  输出文件:        {args.output}")
    print("\n✓ 生成完成")


if __name__ == "__main__":
    main()
