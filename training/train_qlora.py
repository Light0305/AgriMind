#!/usr/bin/env python3
"""
train_qlora.py — QLoRA 微调 Qwen2.5-VL-7B 用于农业病害诊断
============================================================
在 RTX 4090 (24GB) 上使用 4-bit 量化 + LoRA + 梯度检查点进行高效微调。

用法:
  python training/train_qlora.py --config training/configs/qlora_qwen_vl.yaml
  python training/train_qlora.py --config training/configs/qlora_qwen_vl.yaml --resume-from-checkpoint checkpoints/agrimind-qlora/checkpoint-400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AgriDiagnosisDataset(Dataset):
    """Multimodal conversation dataset for Qwen2.5-VL fine-tuning.

    Each JSONL line has the format produced by generate_instructions.py /
    generate_multiturn.py:
        {
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "image", "image": "/path/to/img.jpg"},
                {"type": "text", "text": "..."}
            ]},
            {"role": "assistant", "content": "..."},
            ...
          ]
        }
    """

    def __init__(
        self,
        data_files: list[str],
        processor: AutoProcessor,
        max_seq_length: int = 2048,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.samples: list[dict] = []

        for fpath in data_files:
            fpath = str(Path(fpath).resolve())
            if not os.path.exists(fpath):
                logger.warning("数据文件不存在, 跳过: %s", fpath)
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        if "messages" in sample:
                            self.samples.append(sample)
                    except json.JSONDecodeError:
                        logger.warning("JSON 解析失败: %s 行 %d", fpath, line_num)

        logger.info("加载 %d 条训练样本 (来自 %d 个文件)", len(self.samples), len(data_files))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        messages = sample["messages"]

        # --- Load images referenced in the conversation ---
        images: list[Image.Image] = []
        # Build a clean copy of messages with file:// prefix for processor
        proc_messages = []
        for msg in messages:
            if msg["role"] in ("system", "assistant"):
                proc_messages.append(msg)
                continue
            # User message — may contain image references
            content = msg.get("content", "")
            if isinstance(content, str):
                proc_messages.append(msg)
                continue
            # content is a list of {type, ...} items
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_path = item.get("image", "")
                    # Normalize path: remove file:// prefix if present
                    if img_path.startswith("file://"):
                        img_path = img_path[7:]
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                        # Qwen2.5-VL processor expects {"type": "image", "image": <PIL>}
                        # or file:// path — we use the path form for apply_chat_template
                        # and load images separately
                        new_content.append({"type": "image", "image": f"file://{img_path}"})
                    except Exception as e:
                        logger.debug("无法加载图片 %s: %s", img_path, e)
                        # Skip this image, add placeholder text
                        new_content.append({"type": "text", "text": "[图片加载失败]"})
                else:
                    new_content.append(item)
            proc_messages.append({"role": "user", "content": new_content})

        # --- Build full text with chat template (includes assistant responses) ---
        # For training, we need the full conversation without generation prompt
        full_text = self.processor.apply_chat_template(
            proc_messages, tokenize=False, add_generation_prompt=False
        )

        # --- Build input-only text (to compute label mask) ---
        # We identify assistant token spans by building text without assistant content
        # and comparing lengths. The approach: tokenize full text, then mask non-assistant tokens.

        # --- Process with the Qwen2.5-VL processor ---
        # Use process_vision_info to extract images from messages
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(proc_messages)

        inputs = self.processor(
            text=[full_text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Squeeze batch dim
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # --- Create labels: mask everything except assistant responses ---
        labels = input_ids.clone()

        # Find assistant response boundaries using special tokens
        # Qwen2.5-VL uses: <|im_start|>assistant\n ... <|im_end|>
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Tokenize "assistant" to get its token ids
        assistant_token_ids = self.processor.tokenizer.encode(
            "assistant", add_special_tokens=False
        )
        # Also tokenize newline after "assistant"
        newline_token_ids = self.processor.tokenizer.encode(
            "\n", add_special_tokens=False
        )

        # Strategy: find all <|im_start|>assistant\n ... <|im_end|> spans
        # Only keep labels for the content inside those spans
        ids_list = input_ids.tolist()
        in_assistant = False
        # Default: mask everything (set to -100)
        labels[:] = -100

        i = 0
        while i < len(ids_list):
            if ids_list[i] == im_start_id:
                # Check if followed by "assistant\n"
                # The pattern is: <|im_start|> + "assistant" tokens + "\n" token(s)
                header_len = 1 + len(assistant_token_ids) + len(newline_token_ids)
                if i + header_len <= len(ids_list):
                    candidate = ids_list[i + 1 : i + 1 + len(assistant_token_ids)]
                    if candidate == assistant_token_ids:
                        # Found assistant block start — skip the header
                        i += header_len
                        in_assistant = True
                        continue
                # Not an assistant block — just a user/system block
                i += 1
                continue
            elif ids_list[i] == im_end_id:
                if in_assistant:
                    # Include the <|im_end|> token in loss
                    labels[i] = ids_list[i]
                    in_assistant = False
                i += 1
                continue

            if in_assistant:
                labels[i] = ids_list[i]

            i += 1

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Include pixel_values and image_grid_thw if they exist
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0) \
                if inputs["pixel_values"].dim() > 1 and inputs["pixel_values"].shape[0] == 1 \
                else inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0) \
                if inputs["image_grid_thw"].dim() > 1 and inputs["image_grid_thw"].shape[0] == 1 \
                else inputs["image_grid_thw"]

        return result


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
@dataclass
class MultimodalDataCollator:
    """Collate variable-length multimodal samples into a batch.

    Handles:
    - Padding input_ids, attention_mask, labels to max length in batch
    - Concatenating pixel_values (Qwen2.5-VL uses dynamic resolution, so
      pixel_values can have different shapes per sample)
    - Stacking image_grid_thw
    """

    processor: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # Separate text and vision features
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        # --- Pad text sequences ---
        max_len = max(f["input_ids"].shape[0] for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # Left-pad for causal LM (right-pad is also fine for training)
            # Use right-padding to match HuggingFace convention
            input_ids = torch.nn.functional.pad(
                f["input_ids"], (0, pad_len), value=pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                f["attention_mask"], (0, pad_len), value=0
            )
            labels_padded = torch.nn.functional.pad(
                f["labels"], (0, pad_len), value=-100
            )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels_padded)

        batch = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }

        # --- Handle pixel_values (variable-size image patches) ---
        # Qwen2.5-VL pixel_values shape: (num_patches, C * patch_size * patch_size)
        # Different images have different num_patches, so we concatenate along dim 0
        if any("pixel_values" in f for f in features):
            all_pixel_values = []
            all_grid_thw = []
            for f in features:
                if "pixel_values" in f:
                    pv = f["pixel_values"]
                    if pv.dim() == 1:
                        pv = pv.unsqueeze(0)
                    all_pixel_values.append(pv)
                if "image_grid_thw" in f:
                    gt = f["image_grid_thw"]
                    if gt.dim() == 1:
                        gt = gt.unsqueeze(0)
                    all_grid_thw.append(gt)

            if all_pixel_values:
                batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            if all_grid_thw:
                batch["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)

        return batch


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("配置已加载: %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def load_model_and_processor(config: dict):
    """Load Qwen2.5-VL with 4-bit quantization and apply LoRA."""

    model_cfg = config["model"]
    lora_cfg = config["lora"]

    # --- BitsAndBytes quantization config ---
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(
        model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )

    logger.info("加载基础模型: %s", model_cfg["name_or_path"])
    logger.info("量化配置: 4-bit NF4, double quant=%s, compute=%s",
                model_cfg.get("bnb_4bit_use_double_quant", True), compute_dtype)

    # --- Load model ---
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    # --- Load processor ---
    processor = AutoProcessor.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # --- Prepare model for kbit training ---
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
    )

    # --- LoRA configuration ---
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Report trainable params
    model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
def train_val_split(
    dataset: AgriDiagnosisDataset, val_ratio: float, seed: int = 42
) -> tuple[Dataset, Dataset | None]:
    """Split dataset into train and validation sets."""
    if val_ratio <= 0 or val_ratio >= 1:
        return dataset, None

    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    import torch.utils.data
    return torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QLoRA 微调 Qwen2.5-VL-7B 用于农业病害诊断",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="从指定 checkpoint 恢复训练",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="加载模型和数据, 打印统计信息, 但不启动训练",
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)

    # --- Load model & processor ---
    model, processor = load_model_and_processor(config)

    # --- Enable gradient checkpointing ---
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("梯度检查点已启用")

    # --- Build dataset ---
    train_cfg = config["training"]
    data_cfg = config["data"]

    dataset = AgriDiagnosisDataset(
        data_files=data_cfg["train_files"],
        processor=processor,
        max_seq_length=train_cfg.get("max_seq_length", 2048),
    )

    if len(dataset) == 0:
        logger.error("没有加载到任何训练数据, 请检查数据路径")
        sys.exit(1)

    # --- Train/val split ---
    val_ratio = data_cfg.get("val_split", 0.05)
    train_dataset, val_dataset = train_val_split(dataset, val_ratio)
    logger.info("训练集: %d, 验证集: %d",
                len(train_dataset),
                len(val_dataset) if val_dataset else 0)

    # --- Training arguments ---
    output_dir = Path(train_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        bf16=train_cfg.get("bf16", True),
        fp16=False,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg.get("logging_steps", 10),
        logging_first_step=True,
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        save_strategy="steps",
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=train_cfg.get("save_steps", 200) if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        report_to="none",
        seed=42,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",
    )

    # --- Data collator ---
    data_collator = MultimodalDataCollator(processor=processor)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # --- Train ---
    logger.info("=" * 60)
    logger.info("开始 QLoRA 微调")
    logger.info("=" * 60)
    logger.info("  模型: %s", config["model"]["name_or_path"])
    logger.info("  LoRA rank: %d, alpha: %d", config["lora"]["r"], config["lora"]["lora_alpha"])
    logger.info("  Epochs: %d", training_args.num_train_epochs)
    logger.info("  Batch size: %d x %d (grad accum) = 有效 %d",
                training_args.per_device_train_batch_size,
                training_args.gradient_accumulation_steps,
                training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    logger.info("  Learning rate: %s", training_args.learning_rate)
    logger.info("  Max seq length: %d", train_cfg.get("max_seq_length", 2048))
    logger.info("  Output: %s", output_dir)

    # --- Dry-run mode: print stats and exit ---
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY-RUN 模式 — 跳过训练")
        logger.info("=" * 60)
        # Sample a single batch to verify data pipeline
        try:
            sample = train_dataset[0]
            logger.info("  样本 0 — input_ids shape: %s", sample["input_ids"].shape)
            logger.info("  样本 0 — labels 非 -100 tokens: %d / %d",
                        (sample["labels"] != -100).sum().item(),
                        sample["labels"].shape[0])
            if "pixel_values" in sample:
                logger.info("  样本 0 — pixel_values shape: %s", sample["pixel_values"].shape)
            if "image_grid_thw" in sample:
                logger.info("  样本 0 — image_grid_thw: %s", sample["image_grid_thw"])
        except Exception as e:
            logger.warning("  无法加载样本 0: %s", e)
        # VRAM estimate
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("  总参数量: %.2f B", total_params / 1e9)
        logger.info("  可训练参数: %.2f M (%.2f%%)",
                    trainable_params / 1e6,
                    100.0 * trainable_params / total_params)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info("  GPU 已分配: %.2f GB, 已预留: %.2f GB", allocated, reserved)
        logger.info("Dry-run 完成, 退出.")
        return

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt and not Path(resume_ckpt).exists():
        logger.warning("Checkpoint 路径不存在, 从头开始训练: %s", resume_ckpt)
        resume_ckpt = None

    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- Save final adapter ---
    final_dir = output_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # --- Log metrics ---
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info("=" * 60)
    logger.info("  最终 adapter 保存至: %s", final_dir)
    logger.info("  Training loss: %.4f", metrics.get("train_loss", 0))
    logger.info(
        "  合并模型命令: python training/merge_adapter.py "
        "--base-model %s --adapter %s --output models/agrimind-v1",
        config["model"]["name_or_path"],
        final_dir,
    )


if __name__ == "__main__":
    main()
