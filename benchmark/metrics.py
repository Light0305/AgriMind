#!/usr/bin/env python3
"""
metrics.py — AgriReason Benchmark 评测指标
==========================================
提供 4 个核心指标:
  1. top1_accuracy        — 诊断准确率 (精确匹配)
  2. reasoning_quality    — 推理链质量 (ROUGE-L 文本重叠)
  3. calibration_error    — 置信度校准误差 (ECE)
  4. differential_coverage — 鉴别诊断覆盖率

无外部 embedding 模型依赖, 仅用文本重叠 (ROUGE-L) 评估推理质量。
"""

from __future__ import annotations

import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Top-1 诊断准确率
# ---------------------------------------------------------------------------
def top1_accuracy(
    predictions: list[str],
    ground_truths: list[str],
) -> float:
    """Exact match accuracy on diagnosis.

    对预测和标签做归一化后比较 (去除空格、标点、统一全角半角)。
    返回 [0.0, 1.0]。

    Args:
        predictions: 模型预测的诊断结果列表。
        ground_truths: 真实标签列表。

    Returns:
        准确率。
    """
    if not predictions or not ground_truths:
        return 0.0
    assert len(predictions) == len(ground_truths), (
        f"predictions ({len(predictions)}) 和 ground_truths ({len(ground_truths)}) 长度不一致"
    )

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if _normalize(pred) == _normalize(gt):
            correct += 1
        elif _contains_match(pred, gt):
            correct += 1
    return correct / len(predictions)


def _normalize(text: str) -> str:
    """归一化文本: 去空格、标点, 统一为小写."""
    text = text.strip()
    # 全角 → 半角
    text = text.replace("，", ",").replace("：", ":").replace("。", ".")
    # 去除标点和空格
    text = re.sub(r"[\s\-_.,;:!?()（）【】\[\]\"'""'']+", "", text)
    return text.lower()


def _contains_match(pred: str, gt: str) -> bool:
    """检查预测文本中是否包含 ground_truth (处理模型输出冗长的情况)."""
    norm_pred = _normalize(pred)
    norm_gt = _normalize(gt)
    return norm_gt in norm_pred


# ---------------------------------------------------------------------------
# 2. 推理链质量 (ROUGE-L)
# ---------------------------------------------------------------------------
def reasoning_quality(
    pred_chains: list[list[str]],
    ref_chains: list[list[str]],
) -> float:
    """Semantic similarity between predicted and reference reasoning chains.

    使用 ROUGE-L (基于最长公共子序列) 计算文本重叠度。
    将推理链各步骤拼接成完整文本后计算。
    返回所有样本的平均 ROUGE-L F1 分数 [0.0, 1.0]。

    Args:
        pred_chains: 模型预测的推理链列表, 每条是步骤字符串列表。
        ref_chains: 参考推理链列表。

    Returns:
        平均 ROUGE-L F1 分数。
    """
    if not pred_chains or not ref_chains:
        return 0.0
    assert len(pred_chains) == len(ref_chains), (
        f"pred_chains ({len(pred_chains)}) 和 ref_chains ({len(ref_chains)}) 长度不一致"
    )

    scores = []
    for pred, ref in zip(pred_chains, ref_chains):
        pred_text = "".join(pred)
        ref_text = "".join(ref)
        score = _rouge_l_f1(pred_text, ref_text)
        scores.append(score)

    return sum(scores) / len(scores)


def _lcs_length(x: list, y: list) -> int:
    """最长公共子序列长度 (DP)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # 空间优化: 只保留两行
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    """计算 ROUGE-L F1 分数 (字符级)."""
    if not prediction or not reference:
        return 0.0

    # 按字符 tokenize (中文天然按字, 英文按字符也OK)
    pred_chars = list(prediction.replace(" ", ""))
    ref_chars = list(reference.replace(" ", ""))

    if not pred_chars or not ref_chars:
        return 0.0

    lcs = _lcs_length(pred_chars, ref_chars)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_chars)
    recall = lcs / len(ref_chars)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# 3. 置信度校准误差 (ECE)
# ---------------------------------------------------------------------------
def calibration_error(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    将预测按置信度分箱, 计算每箱 |accuracy - average_confidence| 的加权平均。
    返回 [0.0, 1.0], 越小越好。

    Args:
        confidences: 模型给出的置信度列表, 每项 ∈ [0, 1]。
        correctness: 对应的正确性布尔列表。
        n_bins: 分箱数 (default: 10)。

    Returns:
        ECE 值。
    """
    if not confidences or not correctness:
        return 0.0
    assert len(confidences) == len(correctness), (
        f"confidences ({len(confidences)}) 和 correctness ({len(correctness)}) 长度不一致"
    )

    n = len(confidences)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    # 分箱
    bin_sums: dict[int, float] = defaultdict(float)   # 置信度之和
    bin_correct: dict[int, float] = defaultdict(float) # 正确数
    bin_counts: dict[int, int] = defaultdict(int)      # 样本数

    for conf, corr in zip(confidences, correctness):
        conf = max(0.0, min(1.0, conf))  # clamp
        # 找到所属 bin
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_sums[bin_idx] += conf
        bin_correct[bin_idx] += float(corr)
        bin_counts[bin_idx] += 1

    ece = 0.0
    for b in range(n_bins):
        count = bin_counts[b]
        if count == 0:
            continue
        avg_confidence = bin_sums[b] / count
        avg_accuracy = bin_correct[b] / count
        ece += (count / n) * abs(avg_accuracy - avg_confidence)

    return ece


# ---------------------------------------------------------------------------
# 4. 鉴别诊断覆盖率
# ---------------------------------------------------------------------------
def differential_coverage(
    predictions: list[dict],
    ground_truths: list[dict],
) -> float:
    """For differential diagnosis samples: coverage score.

    评估模型是否:
    1. 正确识别了真实病害 (correct_diagnosis)
    2. 提及并解释了为什么排除混淆病害 (rejection_explained)

    每个样本得分 = 0.6 * correct_diagnosis + 0.4 * rejection_explained
    返回所有样本的平均覆盖率 [0.0, 1.0]。

    Args:
        predictions: 模型预测列表, 每项需包含:
            - "diagnosis": str  — 模型给出的诊断
            - "reasoning": str  — 模型的推理文本
        ground_truths: 真实标签列表, 每项需包含:
            - "ground_truth": str  — 真实诊断
            - "confusable_disease": str  — 需要排除的混淆病害

    Returns:
        平均鉴别诊断覆盖率。
    """
    if not predictions or not ground_truths:
        return 0.0
    assert len(predictions) == len(ground_truths), (
        f"predictions ({len(predictions)}) 和 ground_truths ({len(ground_truths)}) 长度不一致"
    )

    scores = []
    for pred, gt in zip(predictions, ground_truths):
        pred_diagnosis = pred.get("diagnosis", "")
        pred_reasoning = pred.get("reasoning", "")
        true_label = gt.get("ground_truth", "")
        confusable = gt.get("confusable_disease", "")

        # 1. 是否正确诊断
        correct = (
            _normalize(true_label) == _normalize(pred_diagnosis)
            or _contains_match(pred_diagnosis, true_label)
        )

        # 2. 是否提及并排除了混淆病害
        rejection_explained = False
        if confusable:
            norm_confusable = _normalize(confusable)
            norm_reasoning = _normalize(pred_reasoning)
            # 检查推理文本中是否提及混淆病害
            mentioned = norm_confusable in norm_reasoning or confusable in pred_reasoning
            if mentioned:
                # 进一步检查是否有排除性描述
                exclusion_keywords = [
                    "排除", "不是", "非", "区别", "鉴别", "不同于",
                    "不符合", "可以排除", "不太可能", "不像",
                ]
                for kw in exclusion_keywords:
                    if kw in pred_reasoning:
                        rejection_explained = True
                        break

        score = 0.6 * float(correct) + 0.4 * float(rejection_explained)
        scores.append(score)

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 聚合: 一次性计算所有指标
# ---------------------------------------------------------------------------
def compute_all_metrics(
    predictions: list[dict],
    benchmark: list[dict],
) -> dict[str, float]:
    """从预测和 benchmark 数据计算全部 4 个指标.

    Args:
        predictions: 模型预测列表, 每项包含:
            - "id": str
            - "diagnosis": str — 诊断结果
            - "reasoning_chain": list[str] — 推理链
            - "confidence": float — 置信度 [0,1]
            - "reasoning": str — 完整推理文本 (用于鉴别诊断评估)
        benchmark: AgriReason benchmark 样本列表。

    Returns:
        指标字典, 包含:
            - "top1_accuracy": float
            - "reasoning_quality": float
            - "calibration_error": float
            - "differential_coverage": float
    """
    # 按 id 对齐
    bench_map = {s["id"]: s for s in benchmark}
    aligned_preds = []
    aligned_bench = []
    for pred in predictions:
        sid = pred.get("id", "")
        if sid in bench_map:
            aligned_preds.append(pred)
            aligned_bench.append(bench_map[sid])

    if not aligned_preds:
        return {
            "top1_accuracy": 0.0,
            "reasoning_quality": 0.0,
            "calibration_error": 0.0,
            "differential_coverage": 0.0,
        }

    # 1. Top-1 accuracy
    pred_diagnoses = [p.get("diagnosis", "") for p in aligned_preds]
    gt_labels = [b.get("ground_truth", "") for b in aligned_bench]
    acc = top1_accuracy(pred_diagnoses, gt_labels)

    # 2. Reasoning quality
    pred_chains = [p.get("reasoning_chain", []) for p in aligned_preds]
    ref_chains = [b.get("reasoning_chain", []) for b in aligned_bench]
    rq = reasoning_quality(pred_chains, ref_chains)

    # 3. Calibration error
    confs = [p.get("confidence", 0.5) for p in aligned_preds]
    corrs = [
        _normalize(p.get("diagnosis", "")) == _normalize(b.get("ground_truth", ""))
        or _contains_match(p.get("diagnosis", ""), b.get("ground_truth", ""))
        for p, b in zip(aligned_preds, aligned_bench)
    ]
    ce = calibration_error(confs, corrs)

    # 4. Differential coverage (仅 differential 类型)
    diff_preds = []
    diff_gts = []
    for p, b in zip(aligned_preds, aligned_bench):
        if b.get("task_type") == "differential":
            diff_preds.append(p)
            diff_gts.append(b)
    dc = differential_coverage(diff_preds, diff_gts) if diff_preds else 0.0

    return {
        "top1_accuracy": round(acc, 4),
        "reasoning_quality": round(rq, 4),
        "calibration_error": round(ce, 4),
        "differential_coverage": round(dc, 4),
    }


# ---------------------------------------------------------------------------
# 按任务类型分组计算指标
# ---------------------------------------------------------------------------
def compute_metrics_by_type(
    predictions: list[dict],
    benchmark: list[dict],
) -> dict[str, dict[str, float]]:
    """按 task_type 分组计算指标.

    Returns:
        {task_type: {metric_name: value}} 嵌套字典。
    """
    bench_map = {s["id"]: s for s in benchmark}
    grouped: dict[str, tuple[list[dict], list[dict]]] = defaultdict(lambda: ([], []))

    for pred in predictions:
        sid = pred.get("id", "")
        if sid in bench_map:
            b = bench_map[sid]
            task_type = b.get("task_type", "unknown")
            grouped[task_type][0].append(pred)
            grouped[task_type][1].append(b)

    results = {}
    for task_type, (preds, benches) in sorted(grouped.items()):
        results[task_type] = compute_all_metrics(preds, benches)

    return results
