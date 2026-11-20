"""Unit tests cho metric open-set (P4).

Định nghĩa metric nằm ở `docs/protocol.md`, mục "Open-set (P4)".
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

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


def test_macro_f1_binary_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        is_unknown = rng.random(200) < 0.4
        predicted_unknown = rng.random(200) < 0.5
        expected = f1_score(is_unknown, predicted_unknown, average="macro")
        assert macro_f1_binary(predicted_unknown, is_unknown) == pytest.approx(expected)


def test_macro_f1_binary_regression_case() -> None:
    # Truong hop tung lam ham cu tra 0.25 thay vi 0.5.
    predicted = np.array([True, False, False, False])
    truth = np.array([True, False, True, True])
    assert macro_f1_binary(predicted, truth) == pytest.approx(0.5)


def test_macro_f1_binary_perfect_prediction() -> None:
    truth = np.array([True, False, True, False])
    assert macro_f1_binary(truth.copy(), truth) == pytest.approx(1.0)


def test_auroc_and_aupr_direction() -> None:
    known = np.array([0.1, 0.2, 0.3, 0.4])
    unknown = np.array([0.6, 0.7, 0.8, 0.9])
    scores = np.concatenate([known, unknown])
    is_unknown = np.array([False] * 4 + [True] * 4)
    assert auroc(scores, is_unknown) == pytest.approx(1.0)
    assert aupr(scores, is_unknown) == pytest.approx(1.0)
    assert prevalence(is_unknown) == pytest.approx(0.5)


def test_fpr_at_tpr_simple() -> None:
    # Dat TPR=1 phai ha nguong xuong 0.4; khi do 1/4 known bi tu choi.
    known = np.array([0.1, 0.2, 0.3, 0.9])
    unknown = np.array([0.4, 0.5, 0.6, 0.7])
    scores = np.concatenate([known, unknown])
    is_unknown = np.array([False] * 4 + [True] * 4)
    assert fpr_at_tpr(scores, is_unknown, target=0.95) == pytest.approx(0.25)


def test_confusion_matrix_shape_and_counts() -> None:
    y_true = np.array([0, 1, 12, 12])
    y_pred = np.array([0, 12, 12, 3])
    matrix = confusion_matrix(y_true, y_pred, num_labels=12)
    assert matrix.shape == (13, 13)
    assert matrix[0, 0] == 1
    assert matrix[1, 12] == 1
    assert matrix[12, 12] == 1
    assert matrix[12, 3] == 1
    assert int(matrix.sum()) == len(y_true)


def test_per_class_detection_keeps_original_labels() -> None:
    rejected = np.array([True, False, True, True, False, True])
    original = np.array([2, 2, 4, 4, 8, 8])
    result = per_class_detection(rejected, original)
    assert result == {"2": 0.5, "4": 1.0, "8": 0.5}


def test_harmonic_mean_zero_safe() -> None:
    assert harmonic_mean(0.0, 0.5) == 0.0
    assert harmonic_mean(1.0, 0.5) == pytest.approx(2.0 / 3.0)
