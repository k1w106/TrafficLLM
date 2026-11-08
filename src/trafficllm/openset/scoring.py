"""Anomaly scores cho open-set detection (P4).

Quy ước chung: **mọi hàm trả về anomaly score, cao hơn = lạ hơn**.
Quyết định unknown khi `score > threshold`.

`reference` tái hiện hành vi upstream (kể cả lỗi), `corrected` là bản đã sửa.
Chi tiết từng khác biệt: `docs/protocol.md`, mục "Open-set (P4)".
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax ổn định số học theo nhiệt độ `T`."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def msp_anomaly(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """MSP: 1 − xác suất cao nhất. Model càng kém tự tin càng bị coi là lạ."""
    return 1.0 - softmax(logits, temperature).max(axis=1)


def energy_anomaly(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy: −T·logsumexp(logits/T). Năng lượng thấp ⇒ lạ."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    shift = z.max(axis=1, keepdims=True)
    logsumexp = shift[:, 0] + np.log(np.exp(z - shift).sum(axis=1))
    return -float(temperature) * logsumexp


def pairwise_distances(x: np.ndarray, means: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Khoảng cách Euclid giữa từng hàng của `x` và từng vector trong `means`."""
    x = np.asarray(x, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    out = np.empty((x.shape[0], means.shape[0]), dtype=np.float64)
    for start in range(0, x.shape[0], chunk):
        block = x[start : start + chunk]
        diff = block[:, None, :] - means[None, :, :]
        out[start : start + chunk] = np.sqrt((diff * diff).sum(axis=-1))
    return out


def class_means(
    logits: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
    mode: str = "corrected",
) -> tuple[np.ndarray, np.ndarray]:
    """Vector trung bình mỗi lớp, tính trên mẫu được phân loại **đúng** (như upstream).

    Trả về `(means, counts)`. Ở `corrected`, class rỗng được thay bằng centroid
    toàn cục để tránh chia 0; ở `reference` giữ nguyên NaN như upstream.
    """
    if mode not in {"reference", "corrected"}:
        raise ValueError(f"mode khong hop le: {mode}")

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    correct = logits.argmax(axis=1) == labels

    sums = np.zeros((num_labels, logits.shape[1]), dtype=np.float64)
    counts = np.zeros(num_labels, dtype=np.int64)
    if correct.any():
        np.add.at(sums, labels[correct], logits[correct])
        np.add.at(counts, labels[correct], 1)
        fallback = logits[correct].mean(axis=0)
    else:
        fallback = logits.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts[:, None]

    if mode == "corrected":
        empty = counts == 0
        if empty.any():
            means[empty] = fallback
    return means, counts


def neighbor_sets(
    logits: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    num_labels: int,
    k: int,
    mode: str = "corrected",
) -> np.ndarray:
    """Tập K class "gần" của từng lớp dự đoán, dạng ma trận bool (num_labels × num_labels).

    Upstream xếp hạng bằng tổng `||Mean_k − x|| − ||Mean_d − x||` trên các mẫu
    validation được phân loại đúng, rồi lấy K giá trị nhỏ nhất. Vì `Values[i][i]`
    không bao giờ được cộng nên luôn bằng 0 và **luôn được chọn** — tức class đang
    xét nằm trong tập láng giềng của chính nó. `corrected` loại bỏ lỗi này.
    """
    if mode not in {"reference", "corrected"}:
        raise ValueError(f"mode khong hop le: {mode}")

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    correct = logits.argmax(axis=1) == labels

    values = np.zeros((num_labels, num_labels), dtype=np.float64)
    if correct.any():
        x = logits[correct]
        y = labels[correct]
        dist = pairwise_distances(x, means)
        dist_true = dist[np.arange(len(y)), y]
        np.add.at(values, y, dist - dist_true[:, None])

    selected = np.zeros((num_labels, num_labels), dtype=bool)
    for index in range(num_labels):
        scores = values[index].copy()
        if mode == "corrected":
            scores[index] = np.inf
            limit = min(k, num_labels - 1)
        else:
            limit = min(k, num_labels)
        if limit <= 0:
            continue
        order = np.argsort(scores, kind="stable")[:limit]
        selected[index, order] = True
    return selected


def klnd_anomaly(
    logits: np.ndarray,
    means: np.ndarray,
    neighbors: np.ndarray,
    method: int,
    restrict_neighbors: bool,
) -> np.ndarray:
    """Ba biến thể K-LND của upstream (method 1, 2, 3).

    `restrict_neighbors=True` chỉ cộng trên tập K láng giềng (nhánh **open** của
    upstream); `False` cộng trên **mọi** class khác (nhánh **closed** của upstream).
    Sự bất đối xứng này là lỗi được tái hiện trong chế độ `reference`.
    """
    if method not in (1, 2, 3):
        raise ValueError(f"method khong hop le: {method}")

    logits = np.asarray(logits, dtype=np.float64)
    predicted = logits.argmax(axis=1)
    rows = np.arange(logits.shape[0])

    if method == 1:
        distances = pairwise_distances(logits, means)
        return distances[rows, predicted]

    distances = pairwise_distances(logits, means)
    dist_pred = distances[rows, predicted]

    if restrict_neighbors:
        mask = np.asarray(neighbors, dtype=bool)[predicted].copy()
        mask[rows, predicted] = False
    else:
        mask = np.ones_like(distances, dtype=bool)
        mask[rows, predicted] = False

    total = (distances * mask).sum(axis=1)

    if method == 2:
        # upstream: Tot = Σ (dist_k − dist_d), reject nếu Tot < TH ⇒ nhỏ = lạ
        count = mask.sum(axis=1)
        return -(total - count * dist_pred)

    # method 3: Tot = dist_d / Σ dist_k, reject nếu Tot > TH
    return dist_pred / np.maximum(total, EPS)
