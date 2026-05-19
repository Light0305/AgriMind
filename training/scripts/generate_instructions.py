#!/usr/bin/env python3
"""
generate_instructions.py — 自动生成指令微调数据 (单轮诊断 + 辩论对话)
====================================================================
利用大模型 (DashScope API 或本地 Qwen2.5-VL-7B) 为每张作物图片生成:
  --mode single : 单轮诊断推理链 (~3000 samples)
  --mode debate  : 三方辩论对话  (~1500 samples)

输出为 Qwen2.5-VL 对话格式 JSONL, 可直接用于 QLoRA 微调.

用法:
  # DashScope API (默认)
  python generate_instructions.py \
      --mode single \
      --input data/processed/unified_dataset.jsonl \
      --output data/generated/single_diagnosis.jsonl \
      --api-key sk-xxx --max-samples 100 --workers 5

  # 本地 7B 模型回退
  python generate_instructions.py \
      --mode debate \
      --input data/processed/unified_dataset.jsonl \
      --output data/generated/debate.jsonl \
      --use-local --max-samples 50
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_SINGLE = (
    "你是一位资深植物病理学家，擅长通过图像症状对作物病害进行精准诊断。"
    "你会从观察特征出发，进行逐步推理，给出带有依据的诊断结论。"
)

USER_SINGLE = "请诊断这张作物图片中的病害。"

GENERATION_PROMPT_SINGLE = """\
你是资深植物病理学家，请对这张作物图片进行诊断。给出完整推理链：
1. 观察到的症状特征（描述你在图片中看到的具体异常表现）
2. 这些特征指向什么（分析可能的病因、发生机制）
3. 排除了哪些其他可能性，为什么（列出至少1-2个鉴别诊断并说明排除理由）
4. 最终结论及置信度

已知参考标签: {label_cn} ({label_en})
请**基于图片实际特征**给出专业的中文推理过程，保证结论与参考标签一致，但推理过程要自然、细致。
回复用纯文本，不要使用 markdown 格式标记。"""

SYSTEM_DEBATE = (
    "你正在参与一场作物病害诊断辩论。三位植物病理学专家将分别从不同角度分析同一张"
    "图片，通过初诊、质疑和仲裁三个环节，给出最终诊断结论。"
)

USER_DEBATE = "请对这张图片进行辩论式诊断。"

GENERATION_PROMPT_DEBATE = """\
你是三位植物病理学家，正在针对这张作物图片进行辩论式诊断。

请按以下格式生成对话:

【初诊专家】根据图像特征，给出初步诊断及依据（描述具体症状，提出诊断假设）
【质疑专家】对初诊提出合理质疑（指出可能遗漏的特征，提出至少一个替代诊断）
【仲裁专家】综合双方意见，结合关键鉴别特征做出最终裁决（给出置信度）

已知参考标签: {label_cn} ({label_en})
要求：
- 初诊专家需描述至少3个具体的图像症状特征
- 质疑专家需提出至少一个有依据的替代诊断
- 仲裁专家给出带置信度的最终裁决，最终结论应与参考标签一致
- 辩论过程要专业、自然，体现真实学术讨论
回复用纯文本，不要使用 markdown 格式标记。"""


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str | None:
    """Read an image file and return base64 encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[WARN]无法读取图片 {image_path}: {e}")
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
# DashScope backend (native MultiModalConversation API)
# ---------------------------------------------------------------------------
class DashScopeBackend:
    """Generate responses via DashScope MultiModalConversation API."""

    def __init__(self, api_key: str, model: str = "qwen2.5-vl-72b-instruct"):
        self.api_key = api_key
        self.model = model
        self._lock = threading.Lock()
        self._request_times: list[float] = []
        self._rpm_limit = 15  # conservative RPM

    def _rate_limit(self):
        """Sliding-window rate limiter (thread-safe)."""
        with self._lock:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            if len(self._request_times) >= self._rpm_limit:
                sleep_time = 60 - (now - self._request_times[0]) + 0.5
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._request_times.append(time.time())

    def generate(self, prompt: str, image_path: str) -> str | None:
        """Call VLM with an image and prompt, return text response."""
        try:
            from dashscope import MultiModalConversation
        except ImportError:
            # Fallback to OpenAI-compatible endpoint
            return self._generate_openai_compat(prompt, image_path)

        self._rate_limit()

        # DashScope native API supports local file paths directly
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
            print(f"[WARN]DashScope API 调用失败: {e}")
            return None

    def _generate_openai_compat(self, prompt: str, image_path: str) -> str | None:
        """Fallback: OpenAI-compatible API when dashscope SDK is unavailable."""
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
            print(f"[WARN]OpenAI 兼容 API 调用失败: {e}")
            return None


# ---------------------------------------------------------------------------
# Local backend (Qwen2.5-VL-7B, 4-bit)
# ---------------------------------------------------------------------------
class LocalBackend:
    """Generate responses using local Qwen2.5-VL-7B in 4-bit quantization."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.processor = None
        project_root = Path(__file__).resolve().parent.parent.parent
        default_path = project_root / "models" / "qwen2.5-vl-7b"
        self.model_path = model_path or str(default_path)
        self._lock = threading.Lock()
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

        model_name = self.model_path if os.path.isdir(self.model_path) else "Qwen/Qwen2.5-VL-7B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        max_memory = {0: "6GB", "cpu": "16GB"}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print("模型加载完成。")

    def generate(self, prompt: str, image_path: str) -> str | None:
        """Run local inference with image + prompt (thread-safe via lock)."""
        import torch
        from qwen_vl_utils import process_vision_info

        with self._lock:
            try:
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

                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                result = self.processor.decode(
                    generated, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return result.strip()
            except Exception as e:
                print(f"[WARN]本地推理失败: {e}")
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
    """Extract main diagnosis from generated text for consistency comparison.

    Returns a normalized label string. If the reference label appears in text,
    we consider it a match. Otherwise fall back to pattern extraction.
    """
    if label_cn in text:
        return label_cn
    if label_en.lower().replace("_", " ") in text.lower():
        return label_en
    # Try known concluding patterns
    for keyword in ["最终诊断", "诊断为", "判断为", "确诊为", "结论", "仲裁"]:
        idx = text.find(keyword)
        if idx != -1:
            snippet = text[idx: idx + 80]
            return snippet.strip()
    return text[-100:].strip()


def run_with_consistency(
    backend, prompt: str, image_path: str,
    label_cn: str, label_en: str, n_runs: int = 3,
) -> str | None:
    """Run generation n_runs times; return the response only if all runs
    agree on the diagnosis (self-consistency filtering).

    Returns None if inconsistent.
    """
    responses: list[str] = []
    labels: list[str] = []

    for _ in range(n_runs):
        resp = backend.generate(prompt, image_path)
        if resp is None:
            return None
        responses.append(resp)
        labels.append(extract_diagnosis_label(resp, label_cn, label_en))

    # All agree -> consistent
    if len(set(labels)) == 1:
        return responses[0]

    # Majority vote: need at least n_runs-1 agreeing
    counter = Counter(labels)
    most_common_label, count = counter.most_common(1)[0]
    if count >= n_runs - 1:
        idx = labels.index(most_common_label)
        return responses[idx]

    return None  # inconsistent, discard


# ---------------------------------------------------------------------------
# Builders: Qwen2.5-VL SFT conversation format
# ---------------------------------------------------------------------------
def build_single_record(
    image_path: str, response: str, source: str, label_en: str,
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
    image_path: str, response: str, source: str, label_en: str,
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
# Worker: process one sample (called from thread pool)
# ---------------------------------------------------------------------------
def _process_one_sample(
    rec: dict,
    backend,
    mode: str,
    consistency_runs: int,
) -> dict | None:
    """Process a single sample. Returns a formatted record or None on failure."""
    img_path = rec["image_path"]
    label_cn = rec["label_cn"]
    label_en = rec["label_en"]
    source = rec["source"]

    if mode == "single":
        prompt = GENERATION_PROMPT_SINGLE.format(
            label_cn=label_cn, label_en=label_en,
        )
    else:
        prompt = GENERATION_PROMPT_DEBATE.format(
            label_cn=label_cn, label_en=label_en,
        )

    # Self-consistency: run N times, keep only if consistent
    if consistency_runs > 1:
        response = run_with_consistency(
            backend, prompt, img_path, label_cn, label_en, consistency_runs,
        )
    else:
        response = backend.generate(prompt, img_path)

    if response is None:
        return None

    if mode == "single":
        return build_single_record(img_path, response, source, label_en)
    else:
        return build_debate_record(img_path, response, source, label_en)


# ---------------------------------------------------------------------------
# Main generation loop with concurrent workers
# ---------------------------------------------------------------------------
def generate(
    records: list[dict],
    backend,
    output_file: str,
    mode: str,
    max_samples: int | None,
    consistency_runs: int,
    workers: int,
):
    """Generate instruction data with concurrent API calls.

    Args:
        records: Input dataset records.
        backend: DashScopeBackend or LocalBackend instance.
        output_file: Path to output JSONL.
        mode: 'single' or 'debate'.
        max_samples: Cap on samples to generate.
        consistency_runs: Number of runs per sample for self-consistency.
        workers: Max concurrent API calls.
    """
    mode_cn = "单轮诊断" if mode == "single" else "辩论对话"

    # Resume: skip already-generated
    done = load_existing_outputs(output_file)
    pending = [r for r in records if r["image_path"] not in done]

    if max_samples is not None:
        pending = pending[:max_samples]

    print(f"\n{mode_cn}生成: 待处理 {len(pending)} 条 (已完成 {len(done)} 条)")
    if not pending:
        print("  无待处理样本, 跳过。")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    generated = 0
    skipped = 0
    write_lock = threading.Lock()

    # For local backend, force workers=1 (GPU is single-threaded)
    if isinstance(backend, LocalBackend):
        workers = 1

    pbar = tqdm(total=len(pending), desc=f"生成{mode_cn}", unit="sample")

    with open(output_file, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_sample, rec, backend, mode, consistency_runs,
                ): rec
                for rec in pending
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    with write_lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        generated += 1
                else:
                    skipped += 1
                pbar.update(1)

    pbar.close()
    print(f"完成: 生成 {generated} 条, 跳过(不一致/失败) {skipped} 条")
    print(f"输出: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="自动生成指令微调数据 (单轮诊断推理链 / 辩论对话)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # DashScope API — 单轮诊断 (测试 10 条, 5 并发)
  python generate_instructions.py \\
      --mode single \\
      --input data/processed/unified_dataset.jsonl \\
      --output data/generated/single_diagnosis.jsonl \\
      --api-key sk-xxx --max-samples 10 --workers 5

  # 本地 7B 模型 — 辩论对话
  python generate_instructions.py \\
      --mode debate \\
      --input data/processed/unified_dataset.jsonl \\
      --output data/generated/debate.jsonl \\
      --use-local --max-samples 50

  # 3 次自一致性过滤
  python generate_instructions.py \\
      --mode single \\
      --input data/processed/unified_dataset.jsonl \\
      --output data/generated/single_filtered.jsonl \\
      --api-key sk-xxx --consistency-runs 3
""",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "debate"],
        required=True,
        help="生成模式: single=单轮诊断推理链, debate=三方辩论对话",
    )
    parser.add_argument(
        "--input",
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
        "--local-model-path",
        type=str,
        default=None,
        help="本地模型路径 (默认: models/qwen2.5-vl-7b 相对于项目根)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制生成样本数 (用于测试, 默认: single=3000, debate=1500)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="并发 API 调用数 (最大 5, default: 5, 本地模型自动设为 1)",
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

    # Cap workers at 5 as per spec
    args.workers = min(args.workers, 5)

    # Default max_samples by mode
    if args.max_samples is None:
        args.max_samples = 3000 if args.mode == "single" else 1500

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

    mode_cn = "单轮诊断推理链" if args.mode == "single" else "三方辩论对话"
    backend_cn = "本地 7B" if args.use_local else f"DashScope ({args.model})"

    print("=" * 60)
    print("指令数据自动生成")
    print("=" * 60)
    print(f"  模式:          {mode_cn}")
    print(f"  输入文件:      {args.input}")
    print(f"  输出文件:      {args.output}")
    print(f"  后端:          {backend_cn}")
    print(f"  最大样本数:    {args.max_samples}")
    print(f"  并发数:        {args.workers}")
    print(f"  一致性检验:    {args.consistency_runs} 次/样本")
    print()

    # Load data
    records = load_input_data(args.input)
    print(f"加载 {len(records)} 条输入记录")

    # Filter: skip "healthy" labels for debate mode (debates on healthy plants
    # are nonsensical), keep all for single mode
    if args.mode == "debate":
        before = len(records)
        records = [
            r for r in records
            if "healthy" not in r["label_en"].lower()
            and "健康" not in r["label_cn"]
        ]
        print(f"辩论模式: 过滤健康样本 {before} -> {len(records)} 条")

    # Shuffle for class diversity
    random.shuffle(records)

    # Initialize backend
    if args.use_local:
        backend = LocalBackend(model_path=args.local_model_path)
    else:
        backend = DashScopeBackend(api_key=api_key, model=args.model)

    # Generate
    generate(
        records=records,
        backend=backend,
        output_file=args.output,
        mode=args.mode,
        max_samples=args.max_samples,
        consistency_runs=args.consistency_runs,
        workers=args.workers,
    )

    print("\n[OK] 生成完成")


if __name__ == "__main__":
    main()
