"""OpenMax cho open-set detection (P4).

Repo gốc (YasodGinige/TrafficLLM) chỉ cài K-LND, không có code OpenMax, nên
module này chỉ phục vụ chế độ ``corrected``.

Thiết kế (ghi rõ trong ``docs/protocol.md``):

- Không gian khoảng cách: ``emb_last`` (hidden state lớp cuối tại token cuối).
- Mean mỗi class: trung bình embedding của các mẫu ``closed_train`` được phân
  loại **đúng**.
- Weibull: fit trên khoảng cách Euclid từ mẫu ``closed_train`` đúng tới mean
  của class đó (``floc=0``).
- Hiệu chỉnh: xếp hạng class theo logit giảm dần, lấy top ``alpha``; class thứ
  hạng ``r`` bị phạt bởi ``log(1 - w)`` với ``w`` là CDF Weibull của khoảng cách.
- Unknown logit: ``logsumexp(logit_top + log(w))`` trên top ``alpha``.
- Score: ``1 - max p_known`` với ``p = softmax([logits_revised, unknown_logit])``;
  lấy max **chỉ trên các lớp known** để xác suất unknown cao cho score cao.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import weibull_min

EPS = 1e-12


@dataclass
class OpenMaxModel:
    """Tham số đã fit: mean embedding từng class + phân bố Weibull."""

    means: np.ndarray
    weibull_params: list[tuple[float, float, float] | None]
    alpha: int
    num_labels: int


def _class_means(
    embeddings: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean embedding từng class trên mẫu phân loại đúng; class rỗng lấy fallback."""
    correct = logits.argmax(axis=1) == labels
    if correct.any():
        fallback = embeddings[correct].mean(axis=0)
    else:
        fallback = embeddings.mean(axis=0)

    means = np.empty((num_labels, embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros(num_labels, dtype=np.int64)
    for label in range(num_labels):
        mask = correct & (labels == label)
        counts[label] = int(mask.sum())
        means[label] = embeddings[mask].mean(axis=0) if counts[label] else fallback
    return means, counts


def _fit_weibull(
    distances: np.ndarray, min_samples: int
) -> tuple[float, float, float] | None:
    """Fit ``weibull_min`` với ``floc=0``; trả ``None`` nếu không đủ mẫu/suy biến."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size < min_samples or np.allclose(distances, distances[0]):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, loc, scale = weibull_min.fit(distances, floc=0.0)
    except Exception:  # fit co the that bai voi du lieu suy bien
        return None
    params = (float(shape), float(loc), float(scale))
    if not np.isfinite(params).all() or params[2] <= 0.0:
        return None
    return params


def fit_openmax(
    embeddings: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
    alpha: int = 3,
    min_class_samples: int = 2,
) -> OpenMaxModel:
    """Fit OpenMax trên ``closed_train`` (không dùng dữ liệu open)."""
    if alpha < 1:
        raise ValueError("alpha phai >= 1")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    means, counts = _class_means(embeddings, logits, labels, num_labels)
    correct = logits.argmax(axis=1) == labels

    params: list[tuple[float, float, float] | None] = []
    for label in range(num_labels):
        if counts[label] == 0:
            params.append(None)
            continue
        mask = correct & (labels == label)
        distances = cdist(
            embeddings[mask], means[label : label + 1], metric="euclidean"
        ).ravel()
        params.append(_fit_weibull(distances, min_class_samples))

    return OpenMaxModel(
        means=means, weibull_params=params, alpha=int(alpha), num_labels=int(num_labels)
    )


def _weibull_cdf(
    distances: np.ndarray, params: tuple[float, float, float] | None
) -> np.ndarray:
    if params is None:
        return np.zeros_like(distances, dtype=np.float64)
    shape, loc, scale = params
    return weibull_min.cdf(distances, shape, loc=loc, scale=scale)


def openmax_anomaly(
    embeddings: np.ndarray, logits: np.ndarray, model: OpenMaxModel
) -> np.ndarray:
    """Anomaly score OpenMax; cao hơn = lạ hơn. Không nhận nhãn thật."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    n_samples = logits.shape[0]
    if n_samples == 0:
        return np.empty(0, dtype=np.float64)

    distances = cdist(embeddings, model.means, metric="euclidean")
    weights = np.zeros_like(distances)
    for label in range(model.num_labels):
        weights[:, label] = _weibull_cdf(distances[:, label], model.weibull_params[label])
    weights = np.clip(weights, 0.0, 1.0)

    alpha = min(int(model.alpha), model.num_labels)
    order = np.argsort(-logits, axis=1, kind="stable")[:, :alpha]
    rows = np.arange(n_samples)[:, None]

    selected_logits = logits[rows, order]
    selected_weights = weights[rows, order]

    revised = logits.copy()
    revised[rows, order] = selected_logits + np.log(
        np.clip(1.0 - selected_weights, EPS, 1.0)
    )
    unknown = np.logaddexp.reduce(
        selected_logits + np.log(np.clip(selected_weights, EPS, 1.0)), axis=1
    )

    combined = np.column_stack([revised, unknown])
    shifted = combined - combined.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    # max chi tren cac lop known: xac suat unknown cao phai cho score cao.
    return 1.0 - probabilities[:, : model.num_labels].max(axis=1)
