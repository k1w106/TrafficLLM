"""Ablation feature cho K-LND (P6): logits vs emb_last vs emb_mean.

Mọi detector dùng cùng lớp dự đoán lấy từ logits; chỉ không gian tính khoảng
cách (mean/neighbor/score) thay đổi. Nhờ vậy chênh lệch phản ánh chất lượng
feature, không phải chất lượng classifier.

Usage:
    python3 scripts/run_ablation.py --config configs/ablation_ustc_seed42.yaml
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

from trafficllm.openset.pipeline import (  # noqa: E402
    compute_metrics,
    klnd_scores,
    partition_indices,
)
from trafficllm.openset.scoring import class_means, neighbor_sets  # noqa: E402
from trafficllm.utils import load_config, set_seed, write_json  # noqa: E402

SCALAR_METRICS = (
    "closed_argmax_accuracy",
    "CCR",
    "unknown_recall",
    "FPR",
    "TNR",
    "AUROC",
    "AUPR",
    "FPR@95TPR",
    "OSCR",
    "balanced_F1_natural",
    "balanced_F1_1to1",
    "known_rejection_on_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature ablation for K-LND.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    abl_cfg = config["ablation"]
    num_labels = int(data_cfg["num_labels"])
    unknown_label = int(data_cfg["unknown_label"])
    k = int(abl_cfg["neighbors_k"])
    keep_rate = float(abl_cfg["keep_rate"])
    balanced_seed = int(abl_cfg["balanced_f1_seed"])
    mode = str(abl_cfg.get("mode", "corrected"))

    manifest = pd.read_csv(data_cfg["manifest_path"])
    cache = np.load(data_cfg["features_path"], mmap_mode="r")
    if not np.array_equal(cache["sample_id"], manifest["sample_id"].to_numpy()):
        raise SystemExit("sample_id cua cache khong khop manifest")

    indices = partition_indices(manifest)
    labels = manifest["model_label"].to_numpy()
    original_labels = manifest["original_label"].to_numpy()

    train_idx = indices["closed_train"]
    valid_idx = indices["closed_valid"]
    known_idx = indices["closed_test"]
    open_idx = indices["open_test"]

    logits = np.asarray(cache["logits"])
    predicted = logits.argmax(axis=1)

    valid_logits = logits[valid_idx]
    known_logits = logits[known_idx]
    open_logits = logits[open_idx]
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
        },
        "ablation": {},
        "notes": [
            "Lop du doan luon lay tu logits; feature chi thay doi khong gian "
            "mean/neighbor/score.",
            "Ngung hieu chinh tren closed_valid, keep_rate nhu config; open_valid "
            "khong duoc dung.",
            "Day la che do corrected (mot dinh nghia K-LND cho ca closed va open).",
        ],
    }
    rows: list[dict[str, Any]] = []

    for feature_name in abl_cfg["features"]:
        if feature_name not in cache:
            raise SystemExit(f"cache khong co feature: {feature_name}")
        features = np.asarray(cache[feature_name])
        means, counts = class_means(
            features[train_idx],
            labels[train_idx],
            num_labels,
            mode=mode,
            predicted=predicted[train_idx],
        )
        neighbors = neighbor_sets(
            features[valid_idx],
            valid_labels,
            means,
            num_labels,
            k=k,
            mode=mode,
            predicted=predicted[valid_idx],
        )
        print(
            f"{feature_name}: dim={features.shape[1]} | counts={counts.tolist()} | "
            f"neighbor sizes={neighbors.sum(axis=1).tolist()}",
            flush=True,
        )

        for method in abl_cfg["methods"]:
            valid_scores, known_scores, open_scores = klnd_scores(
                mode,
                int(method),
                valid_logits,
                known_logits,
                open_logits,
                means,
                neighbors,
                valid_features=features[valid_idx],
                known_features=features[known_idx],
                open_features=features[open_idx],
            )
            metrics = compute_metrics(
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
            detector = f"klnd{method}"
            results["ablation"].setdefault(feature_name, {})[detector] = metrics
            row: dict[str, Any] = {
                "feature": feature_name,
                "detector": detector,
                "mode": mode,
                "keep_rate": keep_rate,
            }
            row.update({key: metrics[key] for key in SCALAR_METRICS})
            rows.append(row)

    frame = pd.DataFrame(rows)
    output_cfg = config["output"]
    write_json(output_cfg["json_path"], results)
    frame.to_csv(output_cfg["csv_path"], index=False)

    columns = ["feature", "detector", "CCR", "unknown_recall", "AUROC", "FPR@95TPR", "OSCR"]
    print("\nAblation feature (corrected, threshold per-class):")
    print(frame[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nDa luu JSON -> {output_cfg['json_path']}")
    print(f"Da luu CSV  -> {output_cfg['csv_path']}")


if __name__ == "__main__":
    main()
