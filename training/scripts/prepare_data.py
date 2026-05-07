#!/usr/bin/env python3
"""
prepare_data.py — 将下载的数据集统一为 JSONL 格式
====================================================
读取 PlantVillage / PlantDoc / IP102 的目录结构, 生成:
  {"image_path": "...", "label_en": "...", "label_cn": "...", "crop": "...", "source": "..."}

用法:
  python prepare_data.py --input-dir data/raw --output-file data/processed/unified_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 英文 → 中文 标签映射 (覆盖 PlantVillage 全部 38 类 + PlantDoc 常见类)
# ---------------------------------------------------------------------------
LABEL_CN_MAP: dict[str, str] = {
    # ── Tomato (番茄) ──
    "Tomato___Late_blight":                 "番茄晚疫病",
    "Tomato___Early_blight":                "番茄早疫病",
    "Tomato___Bacterial_spot":              "番茄细菌性斑点病",
    "Tomato___Leaf_Mold":                   "番茄叶霉病",
    "Tomato___Septoria_leaf_spot":          "番茄壳针孢叶斑病",
    "Tomato___Spider_mites Two-spotted_spider_mite": "番茄二斑叶螨",
    "Tomato___Target_Spot":                 "番茄靶斑病",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "番茄黄化曲叶病毒病",
    "Tomato___Tomato_mosaic_virus":         "番茄花叶病毒病",
    "Tomato___healthy":                     "番茄健康",
    # ── Apple (苹果) ──
    "Apple___Apple_scab":                   "苹果黑星病",
    "Apple___Black_rot":                    "苹果黑腐病",
    "Apple___Cedar_apple_rust":             "苹果雪松锈病",
    "Apple___healthy":                      "苹果健康",
    # ── Grape (葡萄) ──
    "Grape___Black_rot":                    "葡萄黑腐病",
    "Grape___Esca_(Black_Measles)":         "葡萄黑麻疹病",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "葡萄叶枯病",
    "Grape___healthy":                      "葡萄健康",
    # ── Corn / Maize (玉米) ──
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "玉米灰斑病",
    "Corn_(maize)___Common_rust_":          "玉米普通锈病",
    "Corn_(maize)___Northern_Leaf_Blight":  "玉米北方叶枯病",
    "Corn_(maize)___healthy":               "玉米健康",
    # ── Potato (马铃薯) ──
    "Potato___Early_blight":                "马铃薯早疫病",
    "Potato___Late_blight":                 "马铃薯晚疫病",
    "Potato___healthy":                     "马铃薯健康",
    # ── Strawberry (草莓) ──
    "Strawberry___Leaf_scorch":             "草莓叶焦病",
    "Strawberry___healthy":                 "草莓健康",
    # ── Cherry (樱桃) ──
    "Cherry_(including_sour)___Powdery_mildew": "樱桃白粉病",
    "Cherry_(including_sour)___healthy":    "樱桃健康",
    # ── Peach (桃) ──
    "Peach___Bacterial_spot":               "桃细菌性斑点病",
    "Peach___healthy":                      "桃健康",
    # ── Pepper (辣椒) ──
    "Pepper,_bell___Bacterial_spot":        "辣椒细菌性斑点病",
    "Pepper,_bell___healthy":               "辣椒健康",
    # ── Squash (南瓜) ──
    "Squash___Powdery_mildew":              "南瓜白粉病",
    # ── Soybean (大豆) ──
    "Soybean___healthy":                    "大豆健康",
    # ── Raspberry (覆盆子) ──
    "Raspberry___healthy":                  "覆盆子健康",
    # ── Blueberry (蓝莓) ──
    "Blueberry___healthy":                  "蓝莓健康",
    # ── Orange (柑橘) ──
    "Orange___Haunglongbing_(Citrus_greening)": "柑橘黄龙病",
    # ── PlantDoc 常见补充 ──
    "Tomato leaf late blight":              "番茄叶片晚疫病",
    "Tomato leaf early blight":             "番茄叶片早疫病",
    "Tomato leaf bacterial spot":           "番茄叶片细菌性斑点病",
    "Tomato leaf yellow virus":             "番茄叶片黄化病毒",
    "Tomato leaf mosaic virus":             "番茄叶片花叶病毒",
    "Tomato leaf":                          "番茄叶片健康",
    "Apple leaf":                           "苹果叶片健康",
    "Apple rust leaf":                      "苹果锈病叶片",
    "Apple Scab Leaf":                      "苹果黑星病叶片",
    "Corn leaf blight":                     "玉米叶枯病",
    "Corn rust leaf":                       "玉米锈病叶片",
    "Corn Gray leaf spot":                  "玉米灰斑病",
    "Potato leaf early blight":             "马铃薯早疫病叶片",
    "Potato leaf late blight":              "马铃薯晚疫病叶片",
    "Grape leaf":                           "葡萄叶片健康",
    "Grape leaf black rot":                 "葡萄叶片黑腐病",
    "Cherry leaf":                          "樱桃叶片健康",
    "Peach leaf":                           "桃叶片健康",
    "Raspberry leaf":                       "覆盆子叶片健康",
    "Strawberry leaf":                      "草莓叶片健康",
    "Soybean leaf":                         "大豆叶片健康",
    "Squash Powdery mildew leaf":           "南瓜白粉病叶片",
    "Blueberry leaf":                       "蓝莓叶片健康",
    "Bell pepper leaf":                     "辣椒叶片健康",
    "Bell pepper leaf spot":                "辣椒叶片斑点病",
}

# 从标签推断作物名 (中文)
CROP_CN_MAP: dict[str, str] = {
    "tomato":       "番茄",
    "apple":        "苹果",
    "grape":        "葡萄",
    "corn":         "玉米",
    "maize":        "玉米",
    "potato":       "马铃薯",
    "strawberry":   "草莓",
    "cherry":       "樱桃",
    "peach":        "桃",
    "pepper":       "辣椒",
    "bell pepper":  "辣椒",
    "squash":       "南瓜",
    "soybean":      "大豆",
    "raspberry":    "覆盆子",
    "blueberry":    "蓝莓",
    "orange":       "柑橘",
    "citrus":       "柑橘",
    "rice":         "水稻",
    "wheat":        "小麦",
    "cotton":       "棉花",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _infer_crop(label: str) -> str:
    """从英文标签推断作物中文名."""
    lower = label.lower().replace("_", " ").replace("(", "").replace(")", "")
    for key, cn in CROP_CN_MAP.items():
        if key in lower:
            return cn
    return "未知作物"


def _translate_label(label: str) -> str:
    """英文标签 → 中文标签, 找不到则返回原文."""
    if label in LABEL_CN_MAP:
        return LABEL_CN_MAP[label]
    # Try with spaces replacing underscores
    label_spaces = label.replace("_", " ").strip()
    if label_spaces in LABEL_CN_MAP:
        return LABEL_CN_MAP[label_spaces]
    return label


# ---------------------------------------------------------------------------
# 处理各数据集
# ---------------------------------------------------------------------------
def process_plantvillage(input_dir: Path) -> list[dict]:
    """Process PlantVillage: class directories with images inside."""
    root = input_dir / "PlantVillage"
    records = []
    if not root.exists():
        print("  [PlantVillage] 目录不存在, 跳过")
        return records

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        label_en = class_dir.name
        label_cn = _translate_label(label_en)
        crop = _infer_crop(label_en)
        for img in sorted(class_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                records.append({
                    "image_path": str(img),
                    "label_en": label_en,
                    "label_cn": label_cn,
                    "crop": crop,
                    "source": "PlantVillage",
                })

    print(f"  [PlantVillage] {len(records)} 条记录, {len(class_dirs)} 个类别")
    return records


def process_plantdoc(input_dir: Path) -> list[dict]:
    """Process PlantDoc: look for train/test directories with class subdirectories."""
    root = input_dir / "PlantDoc"
    records = []
    if not root.exists():
        print("  [PlantDoc] 目录不存在, 跳过")
        return records

    # PlantDoc-Dataset typically has train/ and test/ inside
    # Search recursively for class directories containing images
    search_roots = []
    for candidate in ["train", "test", "Train", "Test",
                       "PlantDoc-Dataset/train", "PlantDoc-Dataset/test"]:
        p = root / candidate
        if p.exists():
            search_roots.append(p)
    if not search_roots:
        # Fallback: use root itself
        search_roots = [root]

    seen_paths = set()
    for sr in search_roots:
        for class_dir in sorted(sr.iterdir()):
            if not class_dir.is_dir():
                continue
            label_en = class_dir.name
            label_cn = _translate_label(label_en)
            crop = _infer_crop(label_en)
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTS and str(img) not in seen_paths:
                    seen_paths.add(str(img))
                    records.append({
                        "image_path": str(img),
                        "label_en": label_en,
                        "label_cn": label_cn,
                        "crop": crop,
                        "source": "PlantDoc",
                    })

    print(f"  [PlantDoc] {len(records)} 条记录")
    return records


def process_ip102(input_dir: Path) -> list[dict]:
    """Process IP102: numbered class directories (0–101)."""
    root = input_dir / "IP102"
    records = []
    if not root.exists():
        print("  [IP102] 目录不存在, 跳过")
        return records

    # Try to load class name mapping
    classes_file = root / "classes.txt"
    class_names: dict[str, str] = {}
    if classes_file.exists():
        for line in classes_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                class_names[parts[0]] = parts[1]

    # Scan train/val/test splits
    found_any = False
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        found_any = True
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = class_dir.name
            label_en = class_names.get(class_id, f"pest_class_{class_id}")
            label_cn = _translate_label(label_en)
            crop = _infer_crop(label_en)
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTS:
                    records.append({
                        "image_path": str(img),
                        "label_en": label_en,
                        "label_cn": label_cn,
                        "crop": crop,
                        "source": "IP102",
                    })

    # Fallback: flat numbered directories at root level
    if not found_any:
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            class_id = class_dir.name
            label_en = class_names.get(class_id, f"pest_class_{class_id}")
            label_cn = _translate_label(label_en)
            crop = _infer_crop(label_en)
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTS:
                    records.append({
                        "image_path": str(img),
                        "label_en": label_en,
                        "label_cn": label_cn,
                        "crop": crop,
                        "source": "IP102",
                    })

    print(f"  [IP102] {len(records)} 条记录")
    return records


# ---------------------------------------------------------------------------
# 统计输出
# ---------------------------------------------------------------------------
def print_statistics(records: list[dict]) -> None:
    """Print detailed statistics about the unified dataset."""
    if not records:
        print("\n⚠  没有任何记录, 请先运行 download_datasets.py")
        return

    source_counts = Counter(r["source"] for r in records)
    crop_counts = Counter(r["crop"] for r in records)
    label_counts = Counter(r["label_en"] for r in records)

    print("\n" + "═" * 60)
    print("数据统计")
    print("═" * 60)

    print(f"\n  总图片数: {len(records)}")

    print("\n  ── 按数据集 ──")
    for src, cnt in source_counts.most_common():
        print(f"    {src:<20} {cnt:>8}")

    print("\n  ── 按作物 (前 15) ──")
    for crop, cnt in crop_counts.most_common(15):
        print(f"    {crop:<20} {cnt:>8}")
    if len(crop_counts) > 15:
        print(f"    ... 共 {len(crop_counts)} 种作物")

    print("\n  ── 按类别 (前 20) ──")
    for label, cnt in label_counts.most_common(20):
        print(f"    {label:<50} {cnt:>6}")
    if len(label_counts) > 20:
        print(f"    ... 共 {len(label_counts)} 个类别")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="将农业图像数据集统一为 JSONL 格式",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="原始数据集根目录 (default: data/raw)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/unified_dataset.jsonl",
        help="输出 JSONL 文件路径 (default: data/processed/unified_dataset.jsonl)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_file = Path(args.output_file).resolve()

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        print("请先运行 download_datasets.py")
        sys.exit(1)

    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}\n")

    # Process each dataset
    all_records: list[dict] = []

    print("━" * 60)
    print("处理数据集")
    print("━" * 60)

    all_records.extend(process_plantvillage(input_dir))
    all_records.extend(process_plantdoc(input_dir))
    all_records.extend(process_ip102(input_dir))

    # Write JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n已写入 {len(all_records)} 条记录到: {output_file}")

    # Statistics
    print_statistics(all_records)


if __name__ == "__main__":
    main()
