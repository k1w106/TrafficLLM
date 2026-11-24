"""Tạo manifest DC seed 42 cho audit P5.

DC có 10 class; theo setup 40% closed của paper, chọn 4 class known và 6 class
open. Mặc định chọn known **ngẫu nhiên có seed** (protocol corrected), không
hard-code 4 class đầu như code upstream. Có thể dùng ``--known-mode first-n``
nếu muốn bám sát setup upstream hơn.

Output:
    data/processed/dc_seed42/X.npy
    data/processed/dc_seed42/original_labels.npy
    data/processed/dc_seed42/manifest.csv
    data/processed/dc_seed42/metadata.json

Usage:
    python3 scripts/build_dc_manifest.py \
        --source-dir data/raw/trafficllm_reference/trafficllm_data/DC \
        --output-dir data/processed/dc_seed42 --known-count 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ARRAY_FILES = {
    "train": ("X_train.npy", "y_train.npy"),
    "valid": ("X_valid.npy", "y_valid.npy"),
    "test": ("X_test.npy", "y_test.npy"),
}


def load_arrays(source_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    traces_parts = []
    labels_parts = []
    for split_name, (trace_file, label_file) in ARRAY_FILES.items():
        trace_path = source_dir / trace_file
        label_path = source_dir / label_file
        if not trace_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Khong thay {trace_path} hoac {label_path}")
        traces = np.load(trace_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True).reshape(-1)
        if len(traces) != len(labels):
            raise ValueError(f"{split_name}: X={len(traces)} nhung y={len(labels)}")
        traces_parts.append(traces)
        labels_parts.append(labels.astype(int))
    return np.concatenate(traces_parts), np.concatenate(labels_parts)


def stratified_three_way_split(
    sample_ids: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ids, holdout_ids = train_test_split(
        sample_ids,
        test_size=0.30,
        random_state=seed,
        stratify=labels[sample_ids],
    )
    valid_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=0.50,
        random_state=seed,
        stratify=labels[holdout_ids],
    )
    return train_ids, valid_ids, test_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DC manifest.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-count", type=int, default=4)
    parser.add_argument("--known-mode", choices=["random", "first-n"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values, labels = load_arrays(source_dir)
    all_classes = np.sort(np.unique(labels))
    if args.known_count < 2 or args.known_count >= len(all_classes):
        raise ValueError(
            f"known-count phai trong [2, {len(all_classes) - 1}], nhan {args.known_count}"
        )

    if args.known_mode == "first-n":
        known_classes = all_classes[: args.known_count]
    else:
        rng = np.random.default_rng(args.seed)
        known_classes = np.sort(
            rng.choice(all_classes, size=args.known_count, replace=False)
        )
    open_classes = np.setdiff1d(all_classes, known_classes)
    known_mask = np.isin(labels, known_classes)

    known_ids = np.flatnonzero(known_mask)
    open_ids = np.flatnonzero(~known_mask)
    known_train, known_valid, known_test = stratified_three_way_split(
        known_ids, labels, args.seed
    )
    open_valid, open_test = train_test_split(
        open_ids,
        test_size=0.50,
        random_state=args.seed,
        stratify=labels[open_ids],
    )

    class_to_model_label = {
        int(class_id): index for index, class_id in enumerate(known_classes)
    }
    unknown_model_label = len(known_classes)

    rows = []

    def add_rows(sample_ids: np.ndarray, partition: str) -> None:
        for sample_id in sample_ids:
            original_label = int(labels[sample_id])
            is_known = original_label in class_to_model_label
            model_label = (
                class_to_model_label[original_label]
                if is_known
                else unknown_model_label
            )
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "partition": partition,
                    "original_label": original_label,
                    "model_label": model_label,
                    "is_known": is_known,
                }
            )

    add_rows(known_train, "closed_train")
    add_rows(known_valid, "closed_valid")
    add_rows(known_test, "closed_test")
    add_rows(open_valid, "open_valid")
    add_rows(open_test, "open_test")

    manifest = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    train_original = set(
        manifest.loc[manifest["partition"] == "closed_train", "original_label"].tolist()
    )
    open_original = set(
        manifest.loc[
            manifest["partition"].isin(["open_valid", "open_test"]), "original_label"
        ].tolist()
    )
    assert train_original.isdisjoint(open_original)
    assert len(manifest) == len(labels)
    assert manifest["sample_id"].nunique() == len(labels)
    assert manifest["model_label"].max() <= unknown_model_label
    assert (
        manifest.loc[
            manifest["partition"].isin(["open_valid", "open_test"]), "model_label"
        ]
        == unknown_model_label
    ).all()

    np.save(output_dir / "X.npy", values.astype(np.float32))
    np.save(output_dir / "original_labels.npy", labels)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    metadata = {
        "dataset": "DC",
        "source_dir": str(source_dir),
        "seed": args.seed,
        "known_mode": args.known_mode,
        "known_count": int(args.known_count),
        "known_classes": [int(value) for value in known_classes],
        "open_classes": [int(value) for value in open_classes],
        "unknown_model_label": unknown_model_label,
        "class_to_model_label": class_to_model_label,
        "n_samples": int(len(labels)),
        "n_classes": int(len(all_classes)),
        "feature_shape": [int(values.shape[0]), int(values.shape[1])],
        "split_sizes": manifest["partition"].value_counts().to_dict(),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Da tao:", output_dir)
    print("Tong so trace:", len(labels))
    print("Known classes:", metadata["known_classes"])
    print("Open classes:", metadata["open_classes"])
    print(manifest["partition"].value_counts())
    print("Kiem tra leakage: PASS")


if __name__ == "__main__":
    main()
