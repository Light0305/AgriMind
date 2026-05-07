#!/usr/bin/env python3
"""
evaluate.py — 在 AgriReason Benchmark 上评测模型
================================================
加载 benchmark 数据集和模型, 运行推理, 计算全部 4 个指标。
支持两种评测模式:
  direct — 直接单轮推理
  ddp    — DDP 三方辩论协议 (Debate-Driven Protocol)

用法:
  # 直接推理
  python evaluate.py \\
      --benchmark benchmark/data/agrireason_v1.json \\
      --model-path models/agrimind-v1 \\
      --output benchmark/results/agrimind_v1_results.json \\
      --mode direct

  # DDP 辩论协议
  python evaluate.py \\
      --benchmark benchmark/data/agrireason_v1.json \\
      --model-path models/agrimind-v1 \\
      --output benchmark/results/agrimind_v1_ddp_results.json \\
      --mode ddp

  # 使用 DashScope API 评测远程模型
  python evaluate.py \\
      --benchmark benchmark/data/agrireason_v1.json \\
      --api-key sk-xxx \\
      --api-model qwen2.5-vl-72b-instruct \\
      --output benchmark/results/qwen72b_results.json
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

from metrics import (
    compute_all_metrics,
    compute_metrics_by_type,
)

# ---------------------------------------------------------------------------
# 推理提示词
# ---------------------------------------------------------------------------
EVAL_PROMPT_DIRECT = """\
请对这张作物图片进行诊断分析。

{context}
问题: {question}

要求:
1. 仔细观察图片中的症状特征
2. 给出逐步推理过程
3. 在最后明确给出诊断结论, 格式为 "诊断结论: XXX"
4. 给出置信度 (0-100%), 格式为 "置信度: XX%"

注意: 如果图片质量不佳或症状不典型, 请如实表达不确定性。"""

EVAL_PROMPT_DDP_INITIAL = """\
你是初诊专家。请对这张作物图片进行初步诊断。

{context}
问题: {question}

要求:
1. 描述你观察到的至少3个具体症状特征
2. 基于症状提出初步诊断假设
3. 给出初步诊断及依据"""

EVAL_PROMPT_DDP_CHALLENGE = """\
你是质疑专家。以下是初诊专家的分析:

---
{initial_diagnosis}
---

{context}
请对初诊专家的诊断提出合理质疑:
1. 指出初诊中可能遗漏的特征
2. 提出至少一个替代诊断假设
3. 说明替代假设的依据"""

EVAL_PROMPT_DDP_ARBITRATE = """\
你是仲裁专家。以下是初诊专家和质疑专家的讨论:

初诊意见:
---
{initial_diagnosis}
---

质疑意见:
---
{challenge}
---

{context}
请综合双方意见做出最终裁决:
1. 评估初诊和质疑各自的合理性
2. 指出关键鉴别特征
3. 给出最终诊断结论, 格式为 "诊断结论: XXX"
4. 给出置信度 (0-100%), 格式为 "置信度: XX%\""""


# ---------------------------------------------------------------------------
# 结果解析
# ---------------------------------------------------------------------------
def parse_diagnosis(text: str) -> str:
    """从模型输出中提取诊断结论."""
    # 尝试匹配 "诊断结论: XXX" 格式
    patterns = [
        r"诊断结论[：:]\s*(.+?)(?:\n|$|。|，|,|置信度)",
        r"最终诊断[：:]\s*(.+?)(?:\n|$|。|，|,|置信度)",
        r"诊断为[：:]*\s*(.+?)(?:\n|$|。|，|,)",
        r"确诊为[：:]*\s*(.+?)(?:\n|$|。|，|,)",
        r"判断为[：:]*\s*(.+?)(?:\n|$|。|，|,)",
        r"结论[：:]\s*(.+?)(?:\n|$|。|，|,)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()

    # 兜底: 取最后 50 字符中可能的诊断
    last_part = text[-200:]
    for kw in ["诊断", "结论", "判断", "识别"]:
        idx = last_part.rfind(kw)
        if idx != -1:
            snippet = last_part[idx:idx + 50]
            return snippet.strip()

    return text.strip()[-50:]


def parse_confidence(text: str) -> float:
    """从模型输出中提取置信度 (0-1)."""
    # 匹配 "置信度: XX%" 或 "置信度: 0.XX"
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

    # 默认 0.5 (未表达置信度)
    return 0.5


def parse_reasoning_chain(text: str) -> list[str]:
    """从模型输出中提取推理链 (复用 generate_agrireason 中的逻辑)."""
    # 按编号分段
    numbered = re.split(r'\n\s*\d+\s*[.、)\]]\s*', text)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 3:
        return numbered

    # 按关键词分段
    keywords = ["观察", "分析", "排除", "结论", "对比", "鉴别", "建议", "不确定", "关键", "质疑"]
    segments = []
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

    # 兜底
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


# ---------------------------------------------------------------------------
# 模型接口
# ---------------------------------------------------------------------------
class LocalModel:
    """本地 Qwen2.5-VL 模型 (QLoRA 微调后)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        print(f"正在加载模型: {self.model_path}")
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as e:
            print(f"错误: 缺少依赖: {e}")
            print("请安装: pip install torch transformers peft bitsandbytes accelerate qwen-vl-utils")
            sys.exit(1)

        base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        model_path = Path(self.model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            base_model_name, trust_remote_code=True
        )

        # 检查是否是 LoRA adapter 目录
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            # LoRA adapter → 加载 base + merge
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            self.model = PeftModel.from_pretrained(
                base_model, str(model_path)
            )
        else:
            # 完整模型目录
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        self.model.eval()
        print("模型加载完成。")

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
                max_new_tokens=2000,
                temperature=0.1,  # 评测用低温度
                do_sample=False,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return result.strip()


class APIModel:
    """DashScope API 远程模型."""

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
                model=self.model,
                messages=messages,
                api_key=self.api_key,
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
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".bmp": "image/bmp",
                ".webp": "image/webp"}.get(ext, "image/jpeg")

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
                            {"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{b64}"}},
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
# 评测逻辑
# ---------------------------------------------------------------------------
def evaluate_direct(
    model, benchmark: list[dict],
) -> list[dict]:
    """直接推理模式: 每样本单轮推理."""
    predictions = []
    for sample in tqdm(benchmark, desc="Direct 推理", unit="sample"):
        prompt = EVAL_PROMPT_DIRECT.format(
            context=sample.get("context", ""),
            question=sample.get("question", ""),
        )
        image_path = sample.get("image_path", "")

        try:
            response = model.generate(prompt, image_path)
        except Exception as e:
            print(f"  ⚠ 推理失败 [{sample['id']}]: {e}")
            response = ""

        diagnosis = parse_diagnosis(response)
        confidence = parse_confidence(response)
        reasoning_chain = parse_reasoning_chain(response)

        predictions.append({
            "id": sample["id"],
            "diagnosis": diagnosis,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
            "reasoning": response,
            "raw_response": response,
        })

    return predictions


def evaluate_ddp(
    model, benchmark: list[dict],
) -> list[dict]:
    """DDP 辩论协议模式: 三阶段辩论."""
    predictions = []
    for sample in tqdm(benchmark, desc="DDP 辩论推理", unit="sample"):
        context_str = sample.get("context", "")
        question_str = sample.get("question", "")
        image_path = sample.get("image_path", "")

        # Phase 1: 初诊
        prompt_initial = EVAL_PROMPT_DDP_INITIAL.format(
            context=context_str, question=question_str,
        )
        try:
            initial_response = model.generate(prompt_initial, image_path)
        except Exception as e:
            print(f"  ⚠ 初诊失败 [{sample['id']}]: {e}")
            initial_response = ""

        # Phase 2: 质疑
        prompt_challenge = EVAL_PROMPT_DDP_CHALLENGE.format(
            initial_diagnosis=initial_response,
            context=context_str,
        )
        try:
            challenge_response = model.generate(prompt_challenge, image_path)
        except Exception as e:
            print(f"  ⚠ 质疑失败 [{sample['id']}]: {e}")
            challenge_response = ""

        # Phase 3: 仲裁
        prompt_arbitrate = EVAL_PROMPT_DDP_ARBITRATE.format(
            initial_diagnosis=initial_response,
            challenge=challenge_response,
            context=context_str,
        )
        try:
            arbitrate_response = model.generate(prompt_arbitrate, image_path)
        except Exception as e:
            print(f"  ⚠ 仲裁失败 [{sample['id']}]: {e}")
            arbitrate_response = ""

        # 从仲裁结果提取最终诊断
        final_text = arbitrate_response
        diagnosis = parse_diagnosis(final_text)
        confidence = parse_confidence(final_text)

        # 合并三阶段为推理链
        reasoning_chain = []
        if initial_response:
            reasoning_chain.append(f"初诊: {initial_response}")
        if challenge_response:
            reasoning_chain.append(f"质疑: {challenge_response}")
        if arbitrate_response:
            reasoning_chain.append(f"仲裁: {arbitrate_response}")

        full_reasoning = (
            f"【初诊】\n{initial_response}\n\n"
            f"【质疑】\n{challenge_response}\n\n"
            f"【仲裁】\n{arbitrate_response}"
        )

        predictions.append({
            "id": sample["id"],
            "diagnosis": diagnosis,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
            "reasoning": full_reasoning,
            "raw_response": {
                "initial": initial_response,
                "challenge": challenge_response,
                "arbitrate": arbitrate_response,
            },
        })

    return predictions


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------
def print_summary(
    overall: dict[str, float],
    by_type: dict[str, dict[str, float]],
    mode: str,
    model_name: str,
    n_samples: int,
):
    """打印评测结果摘要表格."""
    print(f"\n{'═' * 70}")
    print(f"  AgriReason Benchmark 评测结果")
    print(f"{'═' * 70}")
    print(f"  模型:      {model_name}")
    print(f"  模式:      {'直接推理' if mode == 'direct' else 'DDP 辩论协议'}")
    print(f"  样本数:    {n_samples}")
    print(f"  时间:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─' * 70}")

    # 总体指标
    print(f"\n  ── 总体指标 ──")
    print(f"    {'指标':<30} {'值':>10}")
    print(f"    {'─' * 42}")
    metric_names = {
        "top1_accuracy": "Top-1 准确率",
        "reasoning_quality": "推理质量 (ROUGE-L)",
        "calibration_error": "校准误差 (ECE) ↓",
        "differential_coverage": "鉴别诊断覆盖率",
    }
    for key, name in metric_names.items():
        val = overall.get(key, 0.0)
        if key == "calibration_error":
            print(f"    {name:<30} {val:>10.4f}")
        else:
            print(f"    {name:<30} {val:>10.4f}")

    # 按类型分组
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

    print(f"\n{'═' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="在 AgriReason Benchmark 上评测模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 本地模型 — 直接推理
  python evaluate.py \\
      --benchmark benchmark/data/agrireason_v1.json \\
      --model-path models/agrimind-v1 \\
      --output benchmark/results/agrimind_v1_results.json

  # 本地模型 — DDP 辩论协议
  python evaluate.py \\
      --benchmark benchmark/data/agrireason_v1.json \\
      --model-path models/agrimind-v1 \\
      --output benchmark/results/agrimind_v1_ddp.json \\
      --mode ddp

  # API 模型
  python evaluate.py \\
      --benchmark benchmark/data/agrireason_v1.json \\
      --api-key sk-xxx --api-model qwen2.5-vl-72b-instruct \\
      --output benchmark/results/qwen72b_results.json
""",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="benchmark/data/agrireason_v1.json",
        help="AgriReason benchmark JSON 文件路径",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="本地模型路径 (LoRA adapter 或完整模型目录)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DashScope API key (也可通过 DASHSCOPE_API_KEY 环境变量设置)",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="qwen2.5-vl-72b-instruct",
        help="API 模型名 (default: qwen2.5-vl-72b-instruct)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/results/eval_results.json",
        help="评测结果输出 JSON 文件路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "ddp"],
        default="direct",
        help="评测模式: direct=直接推理, ddp=DDP辩论协议 (default: direct)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制评测样本数 (用于测试, 默认: 全部)",
    )
    args = parser.parse_args()

    # 验证 benchmark 文件
    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(f"错误: Benchmark 文件不存在: {bench_path}")
        print("请先运行 generate_agrireason.py 生成 benchmark")
        sys.exit(1)

    # 加载 benchmark
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if args.max_samples is not None:
        benchmark = benchmark[:args.max_samples]

    print(f"加载 {len(benchmark)} 条 benchmark 样本")

    # 初始化模型
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")

    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"错误: 模型路径不存在: {model_path}")
            sys.exit(1)
        model = LocalModel(str(model_path))
        model_name = str(model_path.name)
    elif api_key:
        model = APIModel(api_key=api_key, model=args.api_model)
        model_name = args.api_model
    else:
        print("错误: 请指定 --model-path 或 --api-key")
        sys.exit(1)

    mode_cn = "直接推理" if args.mode == "direct" else "DDP 辩论协议"
    print(f"\n评测模式: {mode_cn}")
    print(f"模型: {model_name}")

    # 运行评测
    start_time = time.time()

    if args.mode == "direct":
        predictions = evaluate_direct(model, benchmark)
    else:
        predictions = evaluate_ddp(model, benchmark)

    elapsed = time.time() - start_time

    # 计算指标
    overall = compute_all_metrics(predictions, benchmark)
    by_type = compute_metrics_by_type(predictions, benchmark)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_json = {
        "metadata": {
            "model": model_name,
            "mode": args.mode,
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
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"\n评测结果已保存到: {output_path}")

    # 打印摘要
    print_summary(overall, by_type, args.mode, model_name, len(benchmark))
    print(f"\n  耗时: {elapsed:.1f}s ({elapsed / len(benchmark):.2f}s/样本)")


if __name__ == "__main__":
    main()
