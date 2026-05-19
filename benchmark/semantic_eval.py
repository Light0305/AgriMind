#!/usr/bin/env python3
"""Post-processing: compute semantic accuracy using BGE-M3 embeddings.

Reads existing result JSON files (B2/B3/B4/O1) and computes an additional
"semantic_accuracy" metric that counts a prediction as correct if its
embedding cosine-similarity to the ground truth exceeds a threshold.

Usage:
  python benchmark/semantic_eval.py \
      --results benchmark/results/B4_finetuned_direct.json \
      --results-v2 benchmark/results_v2/B4_finetuned_direct.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_predictions(path: str) -> list[dict]:
    """Load (prediction, ground_truth) pairs from a result file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", [])


def _compute_semantic_accuracy(
    predictions: list[dict],
    embeddings: list[list[float]] | None = None,
    *,
    threshold: float = 0.75,
) -> float:
    """Compute semantic accuracy using BGE-M3 cosine similarity."""
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print(f"  Encoding {len(predictions)} predictions + ground truths ...")

    pred_texts = [p.get("prediction", p.get("diagnosis", "")) for p in predictions]
    gt_texts = [p.get("ground_truth", "") for p in predictions]

    pred_emb = model.encode(pred_texts, batch_size=32)["dense_vecs"]
    gt_emb = model.encode(gt_texts, batch_size=32)["dense_vecs"]

    correct = 0
    for i in range(len(pred_emb)):
        sim = np.dot(pred_emb[i], gt_emb[i]) / (
            np.linalg.norm(pred_emb[i]) * np.linalg.norm(gt_emb[i]) + 1e-8
        )
        if sim >= threshold:
            correct += 1

    return correct / len(pred_emb)


def main():
    parser = argparse.ArgumentParser(
        description="Compute semantic accuracy for ablation result files"
    )
    parser.add_argument(
        "--results", nargs="*", default=[],
        help="Result JSON files to evaluate"
    )
    parser.add_argument(
        "--results-v2", nargs="*", default=[],
        help="V2 result JSON files"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Cosine similarity threshold (default: 0.75)"
    )
    args = parser.parse_args()

    all_files = args.results + getattr(args, "results_v2", [])

    if not all_files:
        # Auto-discover
        for d in ["benchmark/results", "benchmark/results_v2"]:
            if Path(d).exists():
                all_files.extend(
                    str(p) for p in Path(d).glob("*.json")
                    if "report" not in p.name and "O2" not in p.name
                )

    if not all_files:
        print("No result files found. Use --results to specify.")
        sys.exit(1)

    print("Semantic Accuracy Evaluation (BGE-M3, threshold=%.2f)\n" % args.threshold)
    print(f"{'File':<45} {'Exact Acc':>10} {'Semantic Acc':>12}")
    print("-" * 70)

    for fpath in sorted(all_files):
        preds = load_predictions(fpath)
        if not preds:
            print(f"  {Path(fpath).name:<43} {'(no preds)':>25}")
            continue

        # Exact accuracy from predictions
        exact_correct = 0
        for p in preds:
            pred = str(p.get("prediction", p.get("diagnosis", "")))
            gt = str(p.get("ground_truth", ""))
            if pred.strip() == gt.strip():
                exact_correct += 1
            elif gt.strip() and gt.strip() in pred.strip():
                exact_correct += 1
        exact_acc = exact_correct / len(preds)

        sem_acc = _compute_semantic_accuracy(preds, threshold=args.threshold)

        name = Path(fpath).name
        print(f"{name:<45} {exact_acc:>10.4f} {sem_acc:>12.4f}")

    print("-" * 70)


if __name__ == "__main__":
    main()
