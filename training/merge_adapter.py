#!/usr/bin/env python3
"""
merge_adapter.py — 将 LoRA adapter 合并回基础模型
====================================================
训练完成后, 将 QLoRA adapter 权重合并进 Qwen2.5-VL 基础模型,
生成可直接用于推理的完整模型 (无需加载 adapter).

用法:
  python training/merge_adapter.py \
      --base-model models/qwen2.5-vl-7b \
      --adapter checkpoints/agrimind-qlora/checkpoint-final \
      --output models/agrimind-v1

注意:
  - 合并使用 full precision (bfloat16) 以保证精度
  - 合并后的模型可以再次量化用于推理部署
  - 合并过程需要约 28GB 内存 (7B 模型的 bf16 权重)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="将 LoRA adapter 合并回 Qwen2.5-VL 基础模型",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="基础模型路径 (例如 models/qwen2.5-vl-7b)",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="LoRA adapter checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="合并后模型的输出路径",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="合并模型的精度 (default: bfloat16)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="4GB",
        help="模型分片最大大小 (default: 4GB)",
    )
    args = parser.parse_args()

    base_model_path = Path(args.base_model).resolve()
    adapter_path = Path(args.adapter).resolve()
    output_path = Path(args.output).resolve()

    # --- Validate paths ---
    if not base_model_path.exists():
        logger.error("基础模型路径不存在: %s", base_model_path)
        sys.exit(1)
    if not adapter_path.exists():
        logger.error("Adapter 路径不存在: %s", adapter_path)
        sys.exit(1)

    # --- Select dtype ---
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    merge_dtype = dtype_map[args.dtype]

    logger.info("=" * 60)
    logger.info("LoRA Adapter 合并")
    logger.info("=" * 60)
    logger.info("  基础模型:  %s", base_model_path)
    logger.info("  Adapter:   %s", adapter_path)
    logger.info("  输出路径:  %s", output_path)
    logger.info("  合并精度:  %s", args.dtype)

    # --- Load base model in full precision ---
    logger.info("加载基础模型 (full precision, %s) ...", args.dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=merge_dtype,
        device_map="cpu",  # Use CPU for merge to avoid VRAM issues
        trust_remote_code=True,
    )
    logger.info("基础模型加载完成")

    # --- Load LoRA adapter ---
    logger.info("加载 LoRA adapter ...")
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        torch_dtype=merge_dtype,
    )
    logger.info("Adapter 加载完成")

    # --- Merge and unload ---
    logger.info("合并 adapter 权重 ...")
    model = model.merge_and_unload()
    logger.info("合并完成")

    # --- Save merged model ---
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("保存合并后的模型至: %s", output_path)
    model.save_pretrained(
        str(output_path),
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )

    # --- Save processor / tokenizer ---
    logger.info("保存 processor / tokenizer ...")
    processor = AutoProcessor.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
    )
    processor.save_pretrained(str(output_path))

    # --- Verify ---
    logger.info("验证合并后的模型 ...")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    test_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(output_path),
        torch_dtype=merge_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    param_count = sum(p.numel() for p in test_model.parameters()) / 1e9
    del test_model

    logger.info("=" * 60)
    logger.info("合并完成")
    logger.info("=" * 60)
    logger.info("  模型参数: %.2f B", param_count)
    logger.info("  输出路径: %s", output_path)
    logger.info("")
    logger.info("推理测试:")
    logger.info("  from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor")
    logger.info("  model = Qwen2_5_VLForConditionalGeneration.from_pretrained('%s')", output_path)
    logger.info("  processor = AutoProcessor.from_pretrained('%s')", output_path)


if __name__ == "__main__":
    main()
