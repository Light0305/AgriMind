#!/usr/bin/env python3
"""
ResNet-50 baseline for AgriReason benchmark (B1).
=================================================
Trains a ResNet-50 classifier on PlantVillage, evaluates on AgriReason.

Usage:
    python resnet_baseline.py \
        --data-dir data/raw/PlantVillage \
        --benchmark benchmark/data/agrireason_v1.json \
        --output benchmark/results/B1_resnet50.json

    # 跳过训练, 直接用已有权重评测:
    python resnet_baseline.py \
        --checkpoint benchmark/baselines/resnet50_plantvillage.pth \
        --benchmark benchmark/data/agrireason_v1.json \
        --output benchmark/results/B1_resnet50.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

# 将 benchmark/ 加入路径以便导入 metrics
_BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BENCH_DIR))

from metrics import compute_all_metrics, compute_metrics_by_type  # noqa: E402

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

# PlantVillage 中文标签映射 (目录名 → 中文名)
_LABEL_CN_MAP: dict[str, str] = {
    "Tomato___Late_blight": "番茄晚疫病",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Leaf_Mold": "番茄叶霉病",
    "Tomato___Bacterial_spot": "番茄细菌性斑点病",
    "Tomato___Septoria_leaf_spot": "番茄壳针孢叶斑病",
    "Tomato___Target_Spot": "番茄靶斑病",
    "Tomato___Spider_mites Two-spotted_spider_mite": "番茄二斑叶螨",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "番茄黄化曲叶病毒病",
    "Tomato___Tomato_mosaic_virus": "番茄花叶病毒病",
    "Tomato___healthy": "番茄健康",
    "Potato___Late_blight": "马铃薯晚疫病",
    "Potato___Early_blight": "马铃薯早疫病",
    "Potato___healthy": "马铃薯健康",
    "Apple___Apple_scab": "苹果黑星病",
    "Apple___Black_rot": "苹果黑腐病",
    "Apple___Cedar_apple_rust": "苹果雪松锈病",
    "Apple___healthy": "苹果健康",
    "Grape___Black_rot": "葡萄黑腐病",
    "Grape___Esca_(Black_Measles)": "葡萄黑麻疹病",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "葡萄叶枯病",
    "Grape___healthy": "葡萄健康",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "玉米灰斑病",
    "Corn_(maize)___Northern_Leaf_Blight": "玉米北方叶枯病",
    "Corn_(maize)___Common_rust_": "玉米普通锈病",
    "Corn_(maize)___healthy": "玉米健康",
    "Cherry_(including_sour)___Powdery_mildew": "樱桃白粉病",
    "Cherry_(including_sour)___healthy": "樱桃健康",
    "Peach___Bacterial_spot": "桃细菌性斑点病",
    "Peach___healthy": "桃健康",
    "Pepper,_bell___Bacterial_spot": "辣椒细菌性斑点病",
    "Pepper,_bell___healthy": "辣椒健康",
    "Strawberry___Leaf_scorch": "草莓叶焦病",
    "Strawberry___healthy": "草莓健康",
    "Squash___Powdery_mildew": "南瓜白粉病",
    "Soybean___healthy": "大豆健康",
    "Raspberry___healthy": "覆盆子健康",
    "Blueberry___healthy": "蓝莓健康",
    "Orange___Haunglongbing_(Citrus_greening)": "柑橘黄龙病",
}


class PlantVillageDataset(Dataset):
    """PlantVillage 图片分类数据集 (目录结构: data_dir/class_name/img.jpg)."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.class_to_cn: dict[str, str] = {}
        self._scan()

    def _scan(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        class_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        if not class_dirs:
            raise RuntimeError(f"数据目录为空: {self.data_dir}")

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.class_to_cn = {
            c: _LABEL_CN_MAP.get(c, c) for c in self.classes
        }

        for cls_dir in class_dirs:
            idx = self.class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    self.samples.append((str(img_path), idx))

        print(f"PlantVillage: {len(self.samples)} 张图片, {len(self.classes)} 个类别")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    """构建 ResNet-50 分类器, 替换最后 FC 层."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_resnet(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    checkpoint_path: str | None = None,
) -> tuple[nn.Module, PlantVillageDataset]:
    """训练 ResNet-50 分类器."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = PlantVillageDataset(data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)

    # 80/20 split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # 验证集用 val_transform
    val_set.dataset = PlantVillageDataset(data_dir, transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(
            f"  Epoch {epoch:>2}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = checkpoint_path or "benchmark/baselines/resnet50_plantvillage.pth"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "classes": full_dataset.classes,
                "class_to_cn": full_dataset.class_to_cn,
                "num_classes": num_classes,
                "best_val_acc": best_acc,
            }, save_path)
            print(f"    ✓ 最佳模型已保存 (val_acc={best_acc:.4f})")

    print(f"\n训练完成. 最佳验证准确率: {best_acc:.4f}")
    return model, full_dataset


# ---------------------------------------------------------------------------
# AgriReason 评测
# ---------------------------------------------------------------------------
def evaluate_on_agrireason(
    model: nn.Module,
    classes: list[str],
    class_to_cn: dict[str, str],
    benchmark_path: str,
) -> tuple[list[dict], list[dict]]:
    """在 AgriReason benchmark 上评测 ResNet-50."""
    device = next(model.parameters()).device
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # 构建中文标签到类别索引的反向映射 (用于匹配)
    cn_to_class: dict[str, str] = {}
    for cls_name, cn_name in class_to_cn.items():
        cn_to_class[cn_name] = cls_name

    predictions: list[dict] = []
    for sample in tqdm(benchmark, desc="ResNet-50 评测", unit="sample"):
        image_path = sample.get("image_path", "")
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = eval_transform(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  ⚠ 无法加载图片 [{sample['id']}]: {e}")
            predictions.append({
                "id": sample["id"],
                "diagnosis": "未知",
                "confidence": 0.0,
                "reasoning_chain": [],
                "reasoning": "",
                "raw_response": "",
            })
            continue

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = probs.max(1)

        pred_class = classes[top_idx.item()]
        pred_cn = class_to_cn.get(pred_class, pred_class)
        confidence = top_prob.item()

        # Top-3 用于补充信息
        top3_probs, top3_idxs = probs.topk(min(3, len(classes)), dim=1)
        top3 = [
            (class_to_cn.get(classes[i.item()], classes[i.item()]), p.item())
            for i, p in zip(top3_idxs[0], top3_probs[0])
        ]

        predictions.append({
            "id": sample["id"],
            "diagnosis": pred_cn,
            "confidence": round(confidence, 4),
            "reasoning_chain": [],  # ResNet 无推理链
            "reasoning": "",        # ResNet 无推理文本
            "raw_response": json.dumps(
                {"top3": [(name, round(p, 4)) for name, p in top3]},
                ensure_ascii=False,
            ),
        })

    return predictions, benchmark


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ResNet-50 baseline for AgriReason benchmark (B1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw/PlantVillage",
        help="PlantVillage 数据集目录 (训练用)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="benchmark/data/agrireason_v1.json",
        help="AgriReason benchmark JSON 文件",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark/results/B1_resnet50.json",
        help="评测结果输出路径",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="已有 ResNet-50 权重路径 (跳过训练)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint and Path(args.checkpoint).exists():
        # 从已有权重加载
        print(f"加载已有权重: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        classes = ckpt["classes"]
        class_to_cn = ckpt["class_to_cn"]
        num_classes = ckpt["num_classes"]
        model = build_model(num_classes).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  类别数: {num_classes}, 最佳验证准确率: {ckpt.get('best_val_acc', 'N/A')}")
    else:
        # 训练
        if not Path(args.data_dir).exists():
            print(f"错误: 数据目录不存在: {args.data_dir}")
            print("请下载 PlantVillage 数据集或指定 --checkpoint")
            sys.exit(1)

        ckpt_path = "benchmark/baselines/resnet50_plantvillage.pth"
        model, dataset = train_resnet(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_path=ckpt_path,
        )
        classes = dataset.classes
        class_to_cn = dataset.class_to_cn
        # 重新加载最佳权重
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["state_dict"])

    # 评测
    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(f"错误: Benchmark 文件不存在: {bench_path}")
        sys.exit(1)

    start_time = time.time()
    predictions, benchmark = evaluate_on_agrireason(
        model, classes, class_to_cn, str(bench_path),
    )
    elapsed = time.time() - start_time

    # 计算指标
    overall = compute_all_metrics(predictions, benchmark)
    by_type = compute_metrics_by_type(predictions, benchmark)

    # 保存结果 (与 evaluate.py 输出格式一致)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "experiment_id": "B1",
            "method": "ResNet-50",
            "description": "ResNet-50 image classification baseline (pretrained ImageNet + fine-tuned PlantVillage)",
            "model": "resnet50",
            "mode": "classification",
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
    print(f"  B1 — ResNet-50 Baseline 评测结果")
    print(f"{'═' * 60}")
    print(f"  样本数:       {len(benchmark)}")
    print(f"  Top-1 准确率: {overall['top1_accuracy']:.4f}")
    print(f"  推理质量:     N/A (ResNet 无推理链)")
    print(f"  校准误差:     {overall['calibration_error']:.4f}")
    print(f"  鉴别覆盖率:   N/A (ResNet 无推理文本)")
    print(f"  耗时:         {elapsed:.1f}s")
    print(f"{'═' * 60}")
    print(f"  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
