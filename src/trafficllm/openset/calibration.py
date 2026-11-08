"""Hiệu chỉnh ngưỡng open-set từ `closed_valid` (P4).

Ngưỡng **chỉ** được tính từ closed-validation, không bao giờ dùng dữ liệu open
(AGENTS §6.3). Mọi detector quy ước: anomaly score cao hơn = lạ hơn ⇒ reject khi
`score > threshold`.
"""

from __future__ import annotations

import numpy as np

DEFAULT_KEEP_RATE = 0.95
REFERENCE_KEEP_RATE = 0.8


def quantile_threshold(scores: np.ndarray, keep_rate: float, mode: str) -> float:
    """Phân vị của `scores` sao cho giữ lại `keep_rate` mẫu known.

    - `reference`: nearest-rank như upstream — `sorted[int(n · q)]`, clamp vào cuối.
      Đây là phân vị rời rạc, không nội suy.
    - `corrected`: `numpy.quantile` (nội suy tuyến tính).
    """
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.size == 0:
        raise ValueError("khong the tinh nguong tren mang rong")

    if mode == "reference":
        index = int(values.size * keep_rate)
        index = min(index, values.size - 1)
        return float(values[index])
    if mode == "corrected":
        return float(np.quantile(values, keep_rate, method="linear"))
    raise ValueError(f"mode khong hop le: {mode}")


def calibrate_thresholds(
    scores: np.ndarray,
    predicted: np.ndarray,
    num_labels: int,
    mode: str,
    keep_rate: float | None = None,
    magic_empty: bool = False,
) -> np.ndarray:
    """Ngưỡng riêng cho từng lớp dự đoán (giống upstream).

    `magic_empty=True` tái hiện fallback của upstream khi một class không có mẫu
    validation nào được phân loại đúng: class 0 lấy `10`, các class sau lấy ngưỡng
    của class liền trước. `corrected` luôn fallback về phân vị toàn cục.
    """
    scores = np.asarray(scores, dtype=np.float64)
    predicted = np.asarray(predicted)

    if keep_rate is None:
        keep_rate = REFERENCE_KEEP_RATE if mode == "reference" else DEFAULT_KEEP_RATE

    global_threshold = quantile_threshold(scores, keep_rate, mode)
    thresholds = np.full(num_labels, global_threshold, dtype=np.float64)

    for label in range(num_labels):
        subset = np.sort(scores[predicted == label])
        if subset.size == 0:
            if mode == "reference" and magic_empty:
                thresholds[label] = 10.0 if label == 0 else thresholds[label - 1]
            continue
        thresholds[label] = quantile_threshold(subset, keep_rate, mode)

    return thresholds


def reject_flags(scores: np.ndarray, thresholds: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """`True` nếu mẫu bị từ chối, dùng ngưỡng riêng của lớp dự đoán."""
    scores = np.asarray(scores, dtype=np.float64)
    predicted = np.asarray(predicted)
    return scores > np.asarray(thresholds, dtype=np.float64)[predicted]


def known_rejection_rate(
    scores: np.ndarray, thresholds: np.ndarray, predicted: np.ndarray
) -> float:
    """Tỉ lệ mẫu known bị từ chối — chẩn đoán chiều của phân vị."""
    flags = reject_flags(scores, thresholds, predicted)
    return float(flags.mean()) if flags.size else 0.0
