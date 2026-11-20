"""Chạy các detector open-set trên cache feature USTC (P4).

Không chạy lại GPT-2: mọi detector đọc `outputs/features/ustc_seed42.npz`.
Ngưỡng chỉ hiệu chỉnh từ `closed_valid`; `open_valid` không được dùng.

Usage:
    python3 scripts/run_openset.py --config configs/openset_ustc_seed42.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trafficllm.evaluation.metrics import (  # noqa: E402
    aupr,
    auroc,
    confusion_matrix,
    fpr_at_tpr,
    harmonic_mean,
    macro_f1_binary,
    per_class_detection,
    prevalence,
)
from trafficllm.openset.calibration import (  # noqa: E402
    calibrate_thresholds,
    known_rejection_rate,
    reject_flags,
)
from trafficllm.openset.openmax import fit_openmax, openmax_anomaly  # noqa: E402
from trafficllm.openset.scoring import (  # noqa: E402
    class_means,
    energy_anomaly,
    klnd_anomaly,
    msp_anomaly,
    neighbor_sets,
)
from trafficllm.utils import load_config, set_seed, write_json  # noqa: E402

SCALAR_METRICS = (
    "closed_argmax_accuracy",
    "CCR",
    "unknown_recall",
    "FPR",
    "TNR",
    "AUROC",
    "AUPR",
    "prevalence",
    "FPR@95TPR",
    "OSCR",
    "balanced_F1_natural",
    "balanced_F1_1to1",
    "known_rejection_on_valid",
    "n_known_test",
    "n_open_test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run open-set detectors on USTC cache.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def partition_indices(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        partition: np.flatnonzero((manifest["partition"] == partition).to_numpy())
        for partition in (
            "closed_train",
            "closed_valid",
            "closed_test",
            "open_valid",
            "open_test",
        )
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score K-LND, tái hiện bất đối xứng closed/open của upstream ở mode reference."""
    if method == 1:
        return (
            klnd_anomaly(valid_logits, means, neighbors, 1, restrict_neighbors=False),
            klnd_anomaly(known_logits, means, neighbors, 1, restrict_neighbors=False),
            klnd_anomaly(open_logits, means, neighbors, 1, restrict_neighbors=False),
        )

    valid = klnd_anomaly(valid_logits, means, neighbors, method, restrict_neighbors=True)
    open_test = klnd_anomaly(open_logits, means, neighbors, method, restrict_neighbors=True)
    if mode == "reference":
        # Upstream: closed test cong tren tat ca class, open test chi tren K lang gieng.
        known_test = klnd_anomaly(
            known_logits, means, neighbors, method, restrict_neighbors=False
        )
    else:
        known_test = klnd_anomaly(
            known_logits, means, neighbors, method, restrict_neighbors=True
        )
    return valid, known_test, open_test


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    open_cfg = config["openset"]
    num_labels = int(data_cfg["num_labels"])
    unknown_label = int(data_cfg["unknown_label"])
    k = int(open_cfg["neighbors_k"])
    temperature = float(open_cfg["temperature"])
    balanced_seed = int(open_cfg["balanced_f1_seed"])

    manifest = pd.read_csv(data_cfg["manifest_path"])
    cache = np.load(data_cfg["features_path"], mmap_mode="r")
    if not np.array_equal(cache["sample_id"], manifest["sample_id"].to_numpy()):
        raise SystemExit("sample_id cua cache khong khop manifest")

    indices = partition_indices(manifest)
    logits = cache["logits"]
    embeddings = cache["emb_last"]
    labels = manifest["model_label"].to_numpy()
    original_labels = manifest["original_label"].to_numpy()

    train_idx = indices["closed_train"]
    valid_idx = indices["closed_valid"]
    known_idx = indices["closed_test"]
    open_idx = indices["open_test"]

    valid_logits = np.asarray(logits[valid_idx])
    known_logits = np.asarray(logits[known_idx])
    open_logits = np.asarray(logits[open_idx])
    valid_labels = labels[valid_idx]
    known_labels = labels[known_idx]
    open_original_labels = original_labels[open_idx]

    results: dict[str, Any] = {
        "experiment": config["experiment"],
        "phase": config["phase"],
        "seed": int(config["seed"]),
        "config": config,
        "data": {
            "manifest": data_cfg["manifest_path"],
            "features": data_cfg["features_path"],
            "num_labels": num_labels,
            "unknown_label": unknown_label,
            "partition_sizes": {
                partition: int(len(values)) for partition, values in indices.items()
            },
            "known_classes": sorted(
                int(value) for value in manifest.loc[manifest["is_known"], "original_label"].unique()
            ),
            "open_classes": sorted(
                int(value)
                for value in manifest.loc[~manifest["is_known"], "original_label"].unique()
            ),
        },
        "detectors": {},
        "temperature_sensitivity": {},
        "notes": [
            "reference K-LND: threshold hieu chinh tren closed_valid bang nhanh lang gieng "
            "va chi tren mau phan loai dung; closed_test cong tren tat ca class, open_test "
            "chi tren K lang gieng (bat doi xung cua upstream).",
            "reference MSP/Energy khong co ban upstream; day chi la operating point kieu "
            "upstream (phan vi 0.8), khong phai tai lap code goc.",
            "OpenMax chi bao cao corrected vi repo goc khong cai OpenMax.",
            "open_valid khong duoc dung o P4.",
            "AUROC/AUPR/FPR@95TPR tinh tren anomaly score tho; CCR/unknown_recall/OSCR/"
            "balanced_F1/confusion dung nguong per-class.",
        ],
    }

    # MSP va Energy: dung T co dinh cho ket qua chinh.
    for mode in ("reference", "corrected"):
        keep_rate = (
            float(open_cfg["reference_keep_rate"])
            if mode == "reference"
            else float(open_cfg["corrected_keep_rate"])
        )
        for name, score_fn in (("msp", msp_anomaly), ("energy", energy_anomaly)):
            valid_scores = score_fn(valid_logits, temperature)
            known_scores = score_fn(known_logits, temperature)
            open_scores = score_fn(open_logits, temperature)
            results["detectors"].setdefault(name, {})[mode] = compute_metrics(
                valid_scores=valid_scores,
                known_scores=known_scores,
                open_scores=open_scores,
                valid_logits=valid_logits,
                known_logits=known_logits,
                open_logits=open_logits,
                valid_labels=valid_labels,
                known_labels=known_labels,
                open_original_labels=open_original_labels,
                num_labels=num_labels,
                unknown_label=unknown_label,
                mode=mode,
                keep_rate=keep_rate,
                magic_empty=False,
                correct_only_calibration=False,
                balanced_seed=balanced_seed,
            )

    # K-LND 1/2/3.
    for mode in ("reference", "corrected"):
        keep_rate = (
            float(open_cfg["reference_keep_rate"])
            if mode == "reference"
            else float(open_cfg["corrected_keep_rate"])
        )
        means, counts = class_means(
            np.asarray(logits[train_idx]), labels[train_idx], num_labels, mode=mode
        )
        neighbors = neighbor_sets(
            valid_logits, valid_labels, means, num_labels, k=k, mode=mode
        )
        print(
            f"K-LND {mode}: class counts={counts.tolist()} | "
            f"neighbor sizes={neighbors.sum(axis=1).tolist()}",
            flush=True,
        )
        for method in (1, 2, 3):
            valid_scores, known_scores, open_scores = klnd_scores(
                mode,
                method,
                valid_logits,
                known_logits,
                open_logits,
                means,
                neighbors,
            )
            results["detectors"].setdefault(f"klnd{method}", {})[mode] = compute_metrics(
                valid_scores=valid_scores,
                known_scores=known_scores,
                open_scores=open_scores,
                valid_logits=valid_logits,
                known_logits=known_logits,
                open_logits=open_logits,
                valid_labels=valid_labels,
                known_labels=known_labels,
                open_original_labels=open_original_labels,
                num_labels=num_labels,
                unknown_label=unknown_label,
                mode=mode,
                keep_rate=keep_rate,
                magic_empty=bool(open_cfg["reference_magic_empty"]) if mode == "reference" else False,
                correct_only_calibration=(mode == "reference"),
                balanced_seed=balanced_seed,
            )

    # Do nhay T cua MSP/Energy (chi chan doan, khong dung de chon T/nguong).
    for mode in ("reference", "corrected"):
        keep_rate = (
            float(open_cfg["reference_keep_rate"])
            if mode == "reference"
            else float(open_cfg["corrected_keep_rate"])
        )
        for name, score_fn in (("msp", msp_anomaly), ("energy", energy_anomaly)):
            entries: dict[str, Any] = {}
            for temp in open_cfg["temperature_grid"]:
                temp = float(temp)
                metrics = compute_metrics(
                    valid_scores=score_fn(valid_logits, temp),
                    known_scores=score_fn(known_logits, temp),
                    open_scores=score_fn(open_logits, temp),
                    valid_logits=valid_logits,
                    known_logits=known_logits,
                    open_logits=open_logits,
                    valid_labels=valid_labels,
                    known_labels=known_labels,
                    open_original_labels=open_original_labels,
                    num_labels=num_labels,
                    unknown_label=unknown_label,
                    mode=mode,
                    keep_rate=keep_rate,
                    magic_empty=False,
                    correct_only_calibration=False,
                    balanced_seed=balanced_seed,
                )
                entries[f"{temp:g}"] = {
                    key: metrics[key]
                    for key in ("CCR", "unknown_recall", "FPR", "AUROC", "OSCR", "FPR@95TPR")
                }
            results["temperature_sensitivity"].setdefault(name, {})[mode] = entries

    # OpenMax: chi corrected, fit tren closed_train, nguong tren closed_valid.
    if bool(open_cfg["openmax"]["enabled"]):
        model = fit_openmax(
            np.asarray(embeddings[train_idx]),
            np.asarray(logits[train_idx]),
            labels[train_idx],
            num_labels=num_labels,
            alpha=int(open_cfg["openmax"]["alpha"]),
            min_class_samples=int(open_cfg["openmax"]["min_class_samples"]),
        )
        valid_scores = openmax_anomaly(
            np.asarray(embeddings[valid_idx]), valid_logits, model
        )
        known_scores = openmax_anomaly(
            np.asarray(embeddings[known_idx]), known_logits, model
        )
        open_scores = openmax_anomaly(
            np.asarray(embeddings[open_idx]), open_logits, model
        )
        results["detectors"]["openmax"] = {
            "corrected": compute_metrics(
                valid_scores=valid_scores,
                known_scores=known_scores,
                open_scores=open_scores,
                valid_logits=valid_logits,
                known_logits=known_logits,
                open_logits=open_logits,
                valid_labels=valid_labels,
                known_labels=known_labels,
                open_original_labels=open_original_labels,
                num_labels=num_labels,
                unknown_label=unknown_label,
                mode="corrected",
                keep_rate=float(open_cfg["corrected_keep_rate"]),
                magic_empty=False,
                correct_only_calibration=False,
                balanced_seed=balanced_seed,
            )
        }

    output_json = Path(config["metrics"]["output_json"])
    write_json(output_json, results)

    rows = []
    for detector, modes in results["detectors"].items():
        for mode, metrics in modes.items():
            for key in SCALAR_METRICS:
                rows.append(
                    {
                        "detector": detector,
                        "mode": mode,
                        "metric": key,
                        "value": metrics[key],
                    }
                )
            for class_id, value in metrics["open_class_detection"].items():
                rows.append(
                    {
                        "detector": detector,
                        "mode": mode,
                        "metric": f"open_class_{class_id}",
                        "value": value,
                    }
                )
    output_csv = Path(config["metrics"]["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print("\nKet qua chinh (threshold per-class):", flush=True)
    print(
        f"{'detector':10s} {'mode':10s} {'CCR':>8s} {'UCR':>8s} "
        f"{'AUROC':>8s} {'FPR95':>8s} {'OSCR':>8s}",
        flush=True,
    )
    for detector, modes in results["detectors"].items():
        for mode, metrics in modes.items():
            print(
                f"{detector:10s} {mode:10s} {metrics['CCR']:8.4f} "
                f"{metrics['unknown_recall']:8.4f} {metrics['AUROC']:8.4f} "
                f"{metrics['FPR@95TPR']:8.4f} {metrics['OSCR']:8.4f}",
                flush=True,
            )

    print(f"\nDa luu JSON -> {output_json}", flush=True)
    print(f"Da luu CSV  -> {output_csv}", flush=True)


if __name__ == "__main__":
    main()
