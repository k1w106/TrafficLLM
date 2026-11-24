"""Helper dùng chung cho runner P4 và audit P5.

Các hàm ở đây chỉ thao tác trên score/logits/nhãn đã có; không đọc file, không
huấn luyện model. Nhờ vậy `scripts/run_openset.py` và `scripts/run_audit.py`
dùng cùng một định nghĩa metric và cùng một cách tái hiện lỗi upstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trafficllm.evaluation.metrics import (
    aupr,
    auroc,
    confusion_matrix,
    fpr_at_tpr,
    harmonic_mean,
    macro_f1_binary,
    per_class_detection,
    prevalence,
)
from trafficllm.openset.calibration import (
    calibrate_thresholds,
    known_rejection_rate,
    reject_flags,
)
from trafficllm.openset.scoring import klnd_anomaly

PARTITIONS = (
    "closed_train",
    "closed_valid",
    "closed_test",
    "open_valid",
    "open_test",
)


def partition_indices(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Chỉ số dòng (theo sample_id) của từng partition."""
    return {
        partition: np.flatnonzero((manifest["partition"] == partition).to_numpy())
        for partition in PARTITIONS
    }


def compute_metrics(
    *,
    valid_scores: np.ndarray,
    known_scores: np.ndarray,
    open_scores: np.ndarray,
    valid_logits: np.ndarray,
    known_logits: np.ndarray,
    open_logits: np.ndarray,
    valid_labels: np.ndarray,
    known_labels: np.ndarray,
    open_original_labels: np.ndarray,
    num_labels: int,
    unknown_label: int,
    mode: str,
    keep_rate: float,
    magic_empty: bool,
    correct_only_calibration: bool,
    balanced_seed: int,
) -> dict[str, Any]:
    """Hiệu chỉnh ngưỡng trên closed_valid rồi tính toàn bộ metric."""
    valid_pred = valid_logits.argmax(axis=1)
    if correct_only_calibration:
        mask = valid_pred == valid_labels
    else:
        mask = np.ones(len(valid_pred), dtype=bool)

    thresholds = calibrate_thresholds(
        valid_scores[mask],
        valid_pred[mask],
        num_labels,
        mode=mode,
        keep_rate=keep_rate,
        magic_empty=magic_empty,
    )

    known_pred = known_logits.argmax(axis=1)
    known_correct = known_pred == known_labels
    known_rejected = reject_flags(known_scores, thresholds, known_pred)

    open_pred = open_logits.argmax(axis=1)
    open_rejected = reject_flags(open_scores, thresholds, open_pred)

    raw_scores = np.concatenate([known_scores, open_scores])
    is_unknown = np.concatenate(
        [np.zeros(len(known_scores), dtype=bool), np.ones(len(open_scores), dtype=bool)]
    )

    ccr = float(np.mean(~known_rejected & known_correct))
    unknown_recall = float(np.mean(open_rejected))
    false_positive_rate = float(np.mean(known_rejected))
    tnr = 1.0 - false_positive_rate

    known_y_pred = np.where(known_rejected, unknown_label, known_pred)
    open_y_pred = np.where(open_rejected, unknown_label, open_pred)
    y_true = np.concatenate(
        [known_labels, np.full(len(open_scores), unknown_label, dtype=known_labels.dtype)]
    )
    y_pred = np.concatenate([known_y_pred, open_y_pred])
    matrix = confusion_matrix(y_true, y_pred, num_labels=num_labels)

    predicted_unknown = np.concatenate([known_rejected, open_rejected])
    balanced_natural = macro_f1_binary(predicted_unknown, is_unknown)

    rng = np.random.default_rng(balanced_seed)
    n_balanced = min(len(known_rejected), len(open_rejected))
    known_sample = rng.choice(len(known_rejected), size=n_balanced, replace=False)
    open_sample = rng.choice(len(open_rejected), size=n_balanced, replace=False)
    balanced_1to1 = macro_f1_binary(
        np.concatenate([known_rejected[known_sample], open_rejected[open_sample]]),
        np.concatenate(
            [
                np.zeros(n_balanced, dtype=bool),
                np.ones(n_balanced, dtype=bool),
            ]
        ),
    )

    return {
        "calibration": {
            "mode": mode,
            "keep_rate": float(keep_rate),
            "correct_only": bool(correct_only_calibration),
            "n": int(mask.sum()),
            "magic_empty": bool(magic_empty),
        },
        "thresholds": [float(value) for value in thresholds],
        "closed_argmax_accuracy": float(np.mean(known_correct)),
        "CCR": ccr,
        "unknown_recall": unknown_recall,
        "FPR": false_positive_rate,
        "TNR": tnr,
        "AUROC": auroc(raw_scores, is_unknown),
        "AUPR": aupr(raw_scores, is_unknown),
        "prevalence": prevalence(is_unknown),
        "FPR@95TPR": fpr_at_tpr(raw_scores, is_unknown, target=0.95),
        "OSCR": harmonic_mean(ccr, tnr),
        "balanced_F1_natural": balanced_natural,
        "balanced_F1_1to1": balanced_1to1,
        "known_rejection_on_valid": known_rejection_rate(
            valid_scores, thresholds, valid_pred
        ),
        "n_known_test": int(len(known_scores)),
        "n_open_test": int(len(open_scores)),
        "confusion_matrix": matrix.tolist(),
        "open_class_detection": per_class_detection(
            open_rejected, open_original_labels
        ),
    }


def klnd_scores(
    mode: str,
    method: int,
    valid_logits: np.ndarray,
    known_logits: np.ndarray,
    open_logits: np.ndarray,
    means: np.ndarray,
    neighbors: np.ndarray,
    *,
    known_restrict_neighbors: bool | None = None,
    guard_denominator: bool | None = None,
    valid_features: np.ndarray | None = None,
    known_features: np.ndarray | None = None,
    open_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score K-LND; tái hiện bất đối xứng closed/open của upstream ở mode reference.

    `known_restrict_neighbors` cho phép override nhánh của closed test, phục vụ
    chẩn đoán "cùng nhánh" mà không đổi calibration/open.

    `valid_features`/`known_features`/`open_features` cho phép tính khoảng cách
    trên embedding (ablation P6) trong khi lớp dự đoán vẫn lấy từ `*_logits`.
    Mặc định dùng chính logits.

    `guard_denominator=None` suy ra từ mode: `reference` không chặn (giống
    upstream), `corrected` chặn bằng epsilon.
    """
    if guard_denominator is None:
        guard_denominator = mode != "reference"
    if valid_features is None:
        valid_features = valid_logits
    if known_features is None:
        known_features = known_logits
    if open_features is None:
        open_features = open_logits

    valid_pred = np.asarray(valid_logits).argmax(axis=1)
    known_pred = np.asarray(known_logits).argmax(axis=1)
    open_pred = np.asarray(open_logits).argmax(axis=1)

    if method == 1:
        return (
            klnd_anomaly(
                valid_features, means, neighbors, 1, restrict_neighbors=False,
                predicted=valid_pred, guard_denominator=guard_denominator,
            ),
            klnd_anomaly(
                known_features, means, neighbors, 1, restrict_neighbors=False,
                predicted=known_pred, guard_denominator=guard_denominator,
            ),
            klnd_anomaly(
                open_features, means, neighbors, 1, restrict_neighbors=False,
                predicted=open_pred, guard_denominator=guard_denominator,
            ),
        )

    valid = klnd_anomaly(
        valid_features, means, neighbors, method, restrict_neighbors=True,
        predicted=valid_pred, guard_denominator=guard_denominator,
    )
    open_test = klnd_anomaly(
        open_features, means, neighbors, method, restrict_neighbors=True,
        predicted=open_pred, guard_denominator=guard_denominator,
    )
    if known_restrict_neighbors is None:
        known_restrict_neighbors = mode != "reference"
    known_test = klnd_anomaly(
        known_features,
        means,
        neighbors,
        method,
        restrict_neighbors=bool(known_restrict_neighbors),
        predicted=known_pred,
        guard_denominator=guard_denominator,
    )
    return valid, known_test, open_test
