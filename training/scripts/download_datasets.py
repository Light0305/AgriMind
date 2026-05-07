#!/usr/bin/env python3
"""
download_datasets.py — 下载农业病虫害图像数据集
=================================================
数据集:
  1. PlantVillage  (54k images, 38 crop diseases)  — HuggingFace datasets
  2. PlantDoc      (2.6k images, real field photos) — GitHub clone
  3. IP102         (75k+ images, 102 pest species)  — 手动下载指引

用法:
  python download_datasets.py --output-dir data/raw
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# PlantVillage — via HuggingFace datasets
# ---------------------------------------------------------------------------
def download_plantvillage(output_dir: Path) -> dict:
    """Download PlantVillage dataset from HuggingFace and save as image files."""
    dest = output_dir / "PlantVillage"
    info = {"name": "PlantVillage", "status": "skipped", "images": 0}

    if dest.exists() and any(dest.rglob("*.jpg")) or any(dest.rglob("*.JPG")) if dest.exists() else False:
        count = sum(1 for _ in dest.rglob("*") if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
        info.update(status="already exists", images=count)
        print(f"  [PlantVillage] 已存在 ({count} images), 跳过下载")
        return info

    try:
        from datasets import load_dataset
    except ImportError:
        info["status"] = "failed — pip install datasets"
        print("  [PlantVillage] 缺少 datasets 库, 请 pip install datasets")
        return info

    print("  [PlantVillage] 正在从 HuggingFace 下载 (可能需要几分钟) ...")
    try:
        ds = load_dataset("sartajbhuvaji/PlantDisease", split="train", trust_remote_code=True)
    except Exception as e:
        info["status"] = f"failed — {e}"
        print(f"  [PlantVillage] 下载失败: {e}")
        return info

    dest.mkdir(parents=True, exist_ok=True)

    # label_names from dataset features
    label_names = ds.features["label"].names if hasattr(ds.features["label"], "names") else None

    saved = 0
    for idx, sample in enumerate(tqdm(ds, desc="  [PlantVillage] 保存图片", unit="img")):
        image = sample["image"]
        label_id = sample["label"]
        label_str = label_names[label_id] if label_names else str(label_id)

        class_dir = dest / label_str
        class_dir.mkdir(parents=True, exist_ok=True)

        img_path = class_dir / f"{idx:06d}.jpg"
        if not img_path.exists():
            image.save(img_path)
        saved += 1

    info.update(status="ok", images=saved)
    print(f"  [PlantVillage] 完成, 共 {saved} 张图片")
    return info


# ---------------------------------------------------------------------------
# PlantDoc — clone from GitHub
# ---------------------------------------------------------------------------
def download_plantdoc(output_dir: Path) -> dict:
    """Clone PlantDoc dataset from GitHub."""
    dest = output_dir / "PlantDoc"
    info = {"name": "PlantDoc", "status": "skipped", "images": 0}

    if dest.exists() and any(dest.rglob("*.jpg")) or any(dest.rglob("*.JPG")) if dest.exists() else False:
        count = sum(1 for _ in dest.rglob("*") if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
        info.update(status="already exists", images=count)
        print(f"  [PlantDoc] 已存在 ({count} images), 跳过下载")
        return info

    repo_url = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
    print(f"  [PlantDoc] 正在克隆 {repo_url} ...")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        info["status"] = "failed — git not installed"
        print("  [PlantDoc] 失败: git 未安装")
        return info
    except subprocess.CalledProcessError as e:
        info["status"] = f"failed — {e.stderr.strip()}"
        print(f"  [PlantDoc] 克隆失败: {e.stderr.strip()}")
        return info

    count = sum(1 for _ in dest.rglob("*") if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
    info.update(status="ok", images=count)
    print(f"  [PlantDoc] 完成, 共 {count} 张图片")
    return info


# ---------------------------------------------------------------------------
# IP102 — manual download instructions
# ---------------------------------------------------------------------------
def download_ip102(output_dir: Path) -> dict:
    """Print instructions for manually downloading IP102."""
    dest = output_dir / "IP102"
    info = {"name": "IP102", "status": "manual", "images": 0}

    if dest.exists():
        count = sum(1 for _ in dest.rglob("*") if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if count > 0:
            info.update(status="already exists", images=count)
            print(f"  [IP102] 已存在 ({count} images), 跳过")
            return info

    dest.mkdir(parents=True, exist_ok=True)

    instructions = """
  ╔══════════════════════════════════════════════════════════════════╗
  ║                    IP102 数据集 — 手动下载                       ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  IP102 需要在官网申请后下载:                                       ║
  ║                                                                  ║
  ║  1. 访问: https://github.com/xpwu95/IP102                       ║
  ║  2. 根据 README 中的链接申请数据集                                 ║
  ║  3. 下载后解压到:                                                  ║
  ║     {dest}
  ║                                                                  ║
  ║  目录结构应为:                                                     ║
  ║     IP102/                                                       ║
  ║       ├── train/                                                 ║
  ║       │   ├── 0/  (class 0 images)                               ║
  ║       │   ├── 1/                                                 ║
  ║       │   └── ...                                                ║
  ║       ├── val/                                                   ║
  ║       ├── test/                                                  ║
  ║       └── classes.txt                                            ║
  ║                                                                  ║
  ║  备选: 可从 Kaggle 搜索 "IP102" 获取镜像                          ║
  ╚══════════════════════════════════════════════════════════════════╝
""".format(dest=dest)
    print(instructions)

    # Write a README into the placeholder directory
    readme = dest / "README_DOWNLOAD.txt"
    readme.write_text(
        "IP102 dataset placeholder.\n"
        "See: https://github.com/xpwu95/IP102\n"
        f"Download and extract images into: {dest}\n",
        encoding="utf-8",
    )

    info["status"] = "manual — see instructions above"
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="下载农业病虫害图像数据集 (PlantVillage / PlantDoc / IP102)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="数据集保存根目录 (default: data/raw)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"数据集输出目录: {output_dir}\n")

    results: list[dict] = []

    # 1. PlantVillage
    print("━" * 60)
    print("1/3  PlantVillage")
    print("━" * 60)
    results.append(download_plantvillage(output_dir))

    # 2. PlantDoc
    print("\n" + "━" * 60)
    print("2/3  PlantDoc")
    print("━" * 60)
    results.append(download_plantdoc(output_dir))

    # 3. IP102
    print("\n" + "━" * 60)
    print("3/3  IP102")
    print("━" * 60)
    results.append(download_ip102(output_dir))

    # Summary
    print("\n" + "═" * 60)
    print("下载摘要")
    print("═" * 60)
    print(f"  {'数据集':<16} {'图片数':>8}   状态")
    print("  " + "─" * 50)
    for r in results:
        print(f"  {r['name']:<16} {r['images']:>8}   {r['status']}")
    total = sum(r["images"] for r in results)
    print("  " + "─" * 50)
    print(f"  {'合计':<16} {total:>8}")
    print()


if __name__ == "__main__":
    main()
