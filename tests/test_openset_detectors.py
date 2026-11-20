"""Unit tests detector open-set trên dữ liệu Gaussian giả lập (P4).

Mục tiêu: khoá hướng score, chiều ngưỡng, self-neighbor, fallback class rỗng,
mẫu số 0 và OpenMax. Không cần cache thật.
"""

from __future__ import annotations

import numpy as np
import pytest

from trafficllm.evaluation.metrics import auroc
from trafficllm.openset.calibration import (
    calibrate_thresholds,
    known_rejection_rate,
    quantile_threshold,
    reject_flags,
)
from trafficllm.openset.openmax import fit_openmax, openmax_anomaly
from trafficllm.openset.scoring import (
    class_means,
    energy_anomaly,
    klnd_anomaly,
    msp_anomaly,
    neighbor_sets,
)


def make_gaussian(
    num_labels: int = 3,
    dim: int = 6,
    n_train: int = 150,
    n_test: int = 60,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    centers = np.zeros((num_labels, dim))
    for label in range(num_labels):
        centers[label, label % dim] = 8.0 * (label + 1)

    labels_train = np.repeat(np.arange(num_labels), n_train)
    embeddings_train = np.concatenate(
        [centers[label] + rng.normal(0.0, 0.5, (n_train, dim)) for label in range(num_labels)]
    )
    logits_train = np.full((len(labels_train), num_labels), -2.0)
    logits_train[np.arange(len(labels_train)), labels_train] = 10.0
    logits_train += rng.normal(0.0, 0.05, logits_train.shape)

    labels_test = np.repeat(np.arange(num_labels), n_test)
    embeddings_known = np.concatenate(
        [centers[label] + rng.normal(0.0, 0.5, (n_test, dim)) for label in range(num_labels)]
    )
    logits_known = np.full((len(labels_test), num_labels), -2.0)
    logits_known[np.arange(len(labels_test)), labels_test] = 10.0

    embeddings_unknown = centers.mean(axis=0) + rng.normal(0.0, 0.5, (n_test, dim))
    logits_unknown = np.zeros((n_test, num_labels))

    return {
        "centers": centers,
        "labels_train": labels_train,
        "embeddings_train": embeddings_train,
        "logits_train": logits_train,
        "labels_known": labels_test,
        "embeddings_known": embeddings_known,
        "logits_known": logits_known,
        "embeddings_unknown": embeddings_unknown,
        "logits_unknown": logits_unknown,
    }


def test_msp_and_energy_higher_for_unknown() -> None:
    data = make_gaussian()
    known = data["logits_known"]
    unknown = data["logits_unknown"]

    for score_fn in (msp_anomaly, energy_anomaly):
        known_scores = score_fn(known)
        unknown_scores = score_fn(unknown)
        assert unknown_scores.mean() > known_scores.mean()
        labels = np.array([False] * len(known) + [True] * len(unknown))
        assert auroc(np.concatenate([known_scores, unknown_scores]), labels) > 0.9


def test_reject_flags_direction_and_per_class_threshold() -> None:
    scores = np.array([0.9, 0.1, 0.6])
    thresholds = np.array([0.95, 0.05, 0.5])
    predicted = np.array([0, 1, 2])
    flags = reject_flags(scores, thresholds, predicted)
    assert flags.tolist() == [False, True, True]


def test_quantile_threshold_reference_vs_corrected() -> None:
    scores = np.arange(100, dtype=np.float64)
    reference = quantile_threshold(scores, keep_rate=0.8, mode="reference")
    corrected = quantile_threshold(scores, keep_rate=0.8, mode="corrected")
    assert reference == pytest.approx(80.0)
    assert corrected == pytest.approx(79.2)


def test_calibration_keep_rate_direction() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(5000)
    predicted = np.zeros(5000, dtype=int)

    reference = calibrate_thresholds(scores, predicted, 2, mode="reference", magic_empty=True)
    corrected = calibrate_thresholds(scores, predicted, 2, mode="corrected")
    assert known_rejection_rate(scores, reference, predicted) == pytest.approx(0.20, abs=0.01)
    assert known_rejection_rate(scores, corrected, predicted) == pytest.approx(0.05, abs=0.01)


def test_neighbor_sets_self_inclusion() -> None:
    data = make_gaussian()
    logits = data["logits_train"]
    labels = data["labels_train"]
    means, _ = class_means(logits, labels, 3, mode="corrected")

    reference = neighbor_sets(logits, labels, means, 3, k=2, mode="reference")
    corrected = neighbor_sets(logits, labels, means, 3, k=2, mode="corrected")

    assert np.all(np.diag(reference))
    assert not np.any(np.diag(corrected))
    assert reference.sum(axis=1).tolist() == [2, 2, 2]
    assert corrected.sum(axis=1).tolist() == [2, 2, 2]


def test_class_means_empty_class_fallback() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 3))
    labels = np.repeat([0, 1], 10)
    # Dam bao phan loai dung de co mau cho class 0, 1.
    logits[np.arange(20), labels] = 10.0

    means_corrected, counts = class_means(logits, labels, 3, mode="corrected")
    means_reference, _ = class_means(logits, labels, 3, mode="reference")

    assert counts.tolist() == [10, 10, 0]
    assert np.isfinite(means_corrected[2]).all()
    assert np.isnan(means_reference[2]).all()


def test_klnd_methods_return_finite_scores() -> None:
    data = make_gaussian()
    means, _ = class_means(data["logits_train"], data["labels_train"], 3, mode="corrected")
    neighbors = neighbor_sets(
        data["logits_train"], data["labels_train"], means, 3, k=2, mode="corrected"
    )
    logits = np.concatenate([data["logits_known"], data["logits_unknown"]])

    for method in (1, 2, 3):
        scores = klnd_anomaly(logits, means, neighbors, method, restrict_neighbors=True)
        assert scores.shape == (len(logits),)
        assert np.isfinite(scores).all()


def test_klnd_method3_zero_denominator_is_guarded() -> None:
    means = np.zeros((2, 4))
    logits = np.array([[1.0, 0.0, 0.0, 0.0]])
    neighbors = np.ones((2, 2), dtype=bool)
    scores = klnd_anomaly(logits, means, neighbors, method=3, restrict_neighbors=False)
    assert np.isfinite(scores).all()


def test_openmax_penalizes_far_confident_sample() -> None:
    data = make_gaussian()
    model = fit_openmax(
        data["embeddings_train"],
        data["logits_train"],
        data["labels_train"],
        num_labels=3,
        alpha=2,
    )

    near = data["embeddings_known"][:1]
    far = data["embeddings_unknown"][:1]
    logits = np.array([[10.0, 0.0, 0.0]])

    near_score = openmax_anomaly(near, logits, model)[0]
    far_score = openmax_anomaly(far, logits, model)[0]
    assert far_score > near_score
    assert 0.0 <= near_score <= 1.0
    assert 0.0 <= far_score <= 1.0


def test_openmax_handles_empty_class_and_is_finite() -> None:
    data = make_gaussian()
    labels = data["labels_train"].copy()
    keep = labels < 2
    model = fit_openmax(
        data["embeddings_train"][keep],
        data["logits_train"][keep],
        labels[keep],
        num_labels=3,
        alpha=2,
    )
    scores = openmax_anomaly(data["embeddings_known"], data["logits_known"], model)
    assert scores.shape == (len(data["embeddings_known"]),)
    assert np.isfinite(scores).all()
