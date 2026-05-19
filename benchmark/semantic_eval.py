#!/usr/bin/env python3
"""Compute semantic accuracy using BGE-M3 embedding similarity.

Reads existing result JSON and benchmark files, computes semantic accuracy
where a prediction is correct if cosine similarity between BGE-M3 embeddings
of the predicted diagnosis and ground truth label >= threshold.

Usage:
  python benchmark/semantic_eval.py
  (reads hardcoded paths — edit for your setup)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


def _norm(text: str) -> str:
    """Normalize for exact match (same logic as metrics.py)."""
    text = text.replace("，", ",").replace("：", ":").replace("。", ".")
    text = re.sub(r"[\s\-_,.;:!?()（）【】\[\]\"'\"\"]+", "", text)
    return text.lower()


def load_benchmark_gt(bench_path: str) -> dict[str, str]:
    """Return {sample_id: ground_truth_label}."""
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    return {b["id"]: b["ground_truth"] for b in bench}


def compute_semantic_accuracy(
    pred_file: str,
    bench_path: str,
    threshold: float = 0.75,
) -> dict:
    """Compute exact and semantic accuracy for a result file."""
    from sentence_transformers import SentenceTransformer

    id2gt = load_benchmark_gt(bench_path)

    with open(pred_file, encoding="utf-8") as f:
        data = json.load(f)

    preds = data["predictions"]
    pred_texts = [str(p.get("diagnosis", "")).strip() for p in preds]
    gt_texts = [id2gt.get(p["id"], "") for p in preds]

    model = SentenceTransformer("BAAI/bge-m3")
    p_emb = model.encode(pred_texts, batch_size=64, show_progress_bar=True)
    g_emb = model.encode(gt_texts, batch_size=64, show_progress_bar=True)

    exact, semantic = 0, 0
    for i in range(len(preds)):
        pred = pred_texts[i]
        gt = gt_texts[i]
        if not pred or not gt:
            continue
        if _norm(pred) == _norm(gt) or _norm(gt) in _norm(pred):
            exact += 1
            semantic += 1
        else:
            sim = float(
                np.dot(p_emb[i], g_emb[i])
                / (np.linalg.norm(p_emb[i]) * np.linalg.norm(g_emb[i]) + 1e-8)
            )
            if sim >= threshold:
                semantic += 1

    return {
        "exact_accuracy": round(exact / len(preds), 4),
        "semantic_accuracy": round(semantic / len(preds), 4),
        "n_samples": len(preds),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BENCH = "benchmark/data/agrireason_v1.json"
    RESULTS_DIRS = ["benchmark/results", "benchmark/results_v2"]

    print(f"Semantic Accuracy (BGE-M3)\n")
    print(f"{'File':<45} {'Exact':>10} {'Semantic':>12}")
    print("-" * 70)

    for d in RESULTS_DIRS:
        if not Path(d).exists():
            continue
        for p in sorted(Path(d).glob("*.json")):
            if "report" in p.name or "O2" in p.name:
                continue
            try:
                r = compute_semantic_accuracy(str(p), BENCH)
                print(
                    f"{p.name:<45} {r['exact_accuracy']:>10.4f} {r['semantic_accuracy']:>12.4f}"
                )
            except Exception as e:
                print(f"{p.name:<45} {'error':>10}: {e}")
    print("-" * 70)
