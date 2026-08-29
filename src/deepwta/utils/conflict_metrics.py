from __future__ import annotations

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return 0.0

    order = np.argsort(-probs, kind="stable")
    y = labels[order]
    tp = np.cumsum(y == 1)
    ranks = np.arange(1, len(y) + 1)
    precision = tp / ranks
    return float(np.sum(precision[y == 1]) / positives)


def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(-probs, kind="stable")
    y = labels[order]
    tp = np.cumsum(y == 1) / positives
    fp = np.cumsum(y == 0) / negatives
    tpr = np.concatenate([[0.0], tp, [1.0]])
    fpr = np.concatenate([[0.0], fp, [1.0]])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    bins: int = 15,
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(labels) == 0:
        return 0.0

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probs >= boundaries[index]) & (probs <= boundaries[index + 1])
        else:
            mask = (probs >= boundaries[index]) & (probs < boundaries[index + 1])
        if not np.any(mask):
            continue
        confidence = float(np.mean(probs[mask]))
        accuracy = float(np.mean(labels[mask]))
        error += float(np.mean(mask)) * abs(confidence - accuracy)
    return error


def binary_classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= threshold).astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(labels))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision(labels, probs),
        "roc_auc": roc_auc(labels, probs),
        "brier": float(np.mean((probs - labels) ** 2)),
        "ece": expected_calibration_error(labels, probs),
        "positive_rate": float(np.mean(labels)) if len(labels) else 0.0,
        "predicted_positive_rate": float(np.mean(preds)) if len(preds) else 0.0,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def graph_regression_metrics(
    true_count: np.ndarray,
    pred_count: np.ndarray,
) -> dict[str, float]:
    true_count = np.asarray(true_count, dtype=np.float64)
    pred_count = np.asarray(pred_count, dtype=np.float64)
    error = pred_count - true_count
    return {
        "graph_mae": float(np.mean(np.abs(error))),
        "graph_rmse": float(np.sqrt(np.mean(error ** 2))),
    }
