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


def _resolve_predicted(features: np.ndarray, predicted: np.ndarray | None) -> np.ndarray:
    """Nhãn dự đoán dùng để chọn mẫu "đúng" và lớp của ngưỡng.

    Với ablation feature, `features` có thể là embedding trong khi lớp dự đoán
    vẫn lấy từ logits của model. Truyền `predicted` riêng để giữ nguyên lớp dự
    đoán giữa các feature.
    """
    if predicted is None:
        return np.asarray(features).argmax(axis=1)
    return np.asarray(predicted)


def class_means(
    features: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
    mode: str = "corrected",
    predicted: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector trung bình mỗi lớp, tính trên mẫu được phân loại **đúng** (như upstream).

    `features` là không gian dùng để tính mean (logits hoặc embedding). Nếu
    `predicted` được truyền, lớp dự đoán lấy từ đó thay vì `features.argmax`;
    nhờ vậy ablation có thể so sánh logits với embedding mà vẫn dùng cùng một
    lớp dự đoán.

    Trả về `(means, counts)`. Ở `corrected`, class rỗng được thay bằng centroid
    toàn cục để tránh chia 0; ở `reference` giữ nguyên NaN như upstream.
    """
    if mode not in {"reference", "corrected"}:
        raise ValueError(f"mode khong hop le: {mode}")

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    predicted = _resolve_predicted(features, predicted)
    correct = predicted == labels

    sums = np.zeros((num_labels, features.shape[1]), dtype=np.float64)
    counts = np.zeros(num_labels, dtype=np.int64)
    if correct.any():
        np.add.at(sums, labels[correct], features[correct])
        np.add.at(counts, labels[correct], 1)
        fallback = features[correct].mean(axis=0)
    else:
        fallback = features.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts[:, None]

    if mode == "corrected":
        empty = counts == 0
        if empty.any():
            means[empty] = fallback
    return means, counts


def neighbor_sets(
    features: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    num_labels: int,
    k: int,
    mode: str = "corrected",
    predicted: np.ndarray | None = None,
) -> np.ndarray:
    """Tập K class "gần" của từng lớp dự đoán, dạng ma trận bool (num_labels × num_labels).

    `features` là không gian tính khoảng cách (logits hoặc embedding);
    `predicted` cho phép dùng lớp dự đoán từ logits khi feature là embedding.

    Upstream xếp hạng bằng tổng `||Mean_k − x|| − ||Mean_d − x||` trên các mẫu
    validation được phân loại đúng, rồi lấy K giá trị nhỏ nhất. Vì `Values[i][i]`
    không bao giờ được cộng nên luôn bằng 0 và **luôn được chọn** — tức class đang
    xét nằm trong tập láng giềng của chính nó. `corrected` loại bỏ lỗi này.
    """
    if mode not in {"reference", "corrected"}:
        raise ValueError(f"mode khong hop le: {mode}")

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    predicted = _resolve_predicted(features, predicted)
    correct = predicted == labels

    values = np.zeros((num_labels, num_labels), dtype=np.float64)
    if correct.any():
        x = features[correct]
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
    features: np.ndarray,
    means: np.ndarray,
    neighbors: np.ndarray,
    method: int,
    restrict_neighbors: bool,
    predicted: np.ndarray | None = None,
    guard_denominator: bool = True,
) -> np.ndarray:
    """Ba biến thể K-LND của upstream (method 1, 2, 3).

    `features` là không gian tính khoảng cách (logits hoặc embedding);
    `predicted` cho phép dùng lớp dự đoán từ logits khi feature là embedding.

    `restrict_neighbors=True` chỉ cộng trên tập K láng giềng (nhánh **open** của
    upstream); `False` cộng trên **mọi** class khác (nhánh **closed** của upstream).
    Sự bất đối xứng này là lỗi được tái hiện trong chế độ `reference`.

    `guard_denominator=False` tái hiện đúng upstream method 3: không chặn mẫu số
    0 nên có thể trả NaN/inf. Chế độ `corrected` luôn bật guard.
    """
    if method not in (1, 2, 3):
        raise ValueError(f"method khong hop le: {method}")

    features = np.asarray(features, dtype=np.float64)
    predicted = _resolve_predicted(features, predicted)
    rows = np.arange(features.shape[0])

    if method == 1:
        distances = pairwise_distances(features, means)
        return distances[rows, predicted]

    distances = pairwise_distances(features, means)
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
    if guard_denominator:
        return dist_pred / np.maximum(total, EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dist_pred / total
