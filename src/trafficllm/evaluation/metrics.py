"""Metric cho open-set detection (P4).

Định nghĩa đầy đủ nằm ở `docs/protocol.md`, mục "Open-set (P4)".
Quy ước: positive = **unknown**; `is_unknown` là mảng bool.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

EPS = 1e-12


def _check(scores: np.ndarray, is_unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    is_unknown = np.asarray(is_unknown, dtype=bool)
    if scores.shape != is_unknown.shape:
        raise ValueError("scores va is_unknown phai cung shape")
    if is_unknown.all() or not is_unknown.any():
        raise ValueError("can ca hai lop known va unknown de tinh metric")
    return scores, is_unknown


def auroc(scores: np.ndarray, is_unknown: np.ndarray) -> float:
    """Diện tích dưới ROC, positive = unknown. 0,5 = đoán mò."""
    scores, is_unknown = _check(scores, is_unknown)
    return float(roc_auc_score(is_unknown, scores))


def aupr(scores: np.ndarray, is_unknown: np.ndarray) -> float:
    """Average precision, positive = unknown. Luôn đọc kèm prevalence."""
    scores, is_unknown = _check(scores, is_unknown)
    return float(average_precision_score(is_unknown, scores))


def prevalence(is_unknown: np.ndarray) -> float:
    is_unknown = np.asarray(is_unknown, dtype=bool)
    return float(is_unknown.mean())


def fpr_at_tpr(scores: np.ndarray, is_unknown: np.ndarray, target: float = 0.95) -> float:
    """FPR nhỏ nhất đạt `unknown_recall >= target`."""
    scores, is_unknown = _check(scores, is_unknown)
    fpr, tpr, _ = roc_curve(is_unknown, scores)
    reachable = tpr >= target
    if not reachable.any():
        return 1.0
    return float(fpr[reachable].min())


def macro_f1_binary(predicted_unknown: np.ndarray, is_unknown: np.ndarray) -> float:
    """Macro-F1 của bài toán nhị phân known/unknown (trung bình F1 hai lớp).

    Dùng macro thay vì F1 của lớp unknown để kết quả không bị chi phối bởi tỉ lệ
    known/unknown — nhờ vậy so sánh được giữa tỉ lệ tự nhiên và 1:1.
    """
    predicted_unknown = np.asarray(predicted_unknown, dtype=bool)
    is_unknown = np.asarray(is_unknown, dtype=bool)

    scores = []
    for label in (False, True):
        tp = float(np.sum(predicted_unknown & (is_unknown == label)))
        fp = float(np.sum(predicted_unknown & (is_unknown != label)))
        fn = float(np.sum((~predicted_unknown) & (is_unknown == label)))
        denominator = 2.0 * tp + fp + fn
        scores.append(0.0 if denominator == 0.0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def harmonic_mean(first: float, second: float) -> float:
    if first + second <= 0.0:
        return 0.0
    return float(2.0 * first * second / (first + second))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int = 12) -> np.ndarray:
    """Ma trận (num_labels+1) × (num_labels+1); hàng/cột cuối là unknown (12)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = np.zeros((num_labels + 1, num_labels + 1), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_detection(rejected: np.ndarray, original_labels: np.ndarray) -> dict[str, float]:
    """Tỉ lệ bị từ chối theo từng class open (giữ nhãn gốc, AGENTS §6.7)."""
    rejected = np.asarray(rejected, dtype=bool)
    original_labels = np.asarray(original_labels)
    result: dict[str, float] = {}
    for label in np.unique(original_labels):
        mask = original_labels == label
        result[str(int(label))] = float(rejected[mask].mean())
    return result
