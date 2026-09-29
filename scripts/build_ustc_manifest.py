#python scripts/build_ustc_manifest.py --source-dir "data/raw/trafficllm_reference/trafficllm_data/USTC" --output-dir data/processed/ustc_seed42 --known-count 12 --known-mode random --seed 42
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
            raise FileNotFoundError(
                f"Không tìm thấy {trace_path} hoặc {label_path}. "
                "Kiểm tra lại --source-dir."
            )

        traces = np.load(trace_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True).reshape(-1)

        if len(traces) != len(labels):
            raise ValueError(
                f"{split_name}: traces={len(traces)} nhưng labels={len(labels)}"
            )

        traces_parts.append(traces)
        labels_parts.append(labels.astype(int))

    return np.concatenate(traces_parts), np.concatenate(labels_parts)


def stratified_three_way_split(
    sample_ids: np.ndarray,
    all_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ids, holdout_ids = train_test_split(
        sample_ids,
        test_size=0.30,
        random_state=seed,
        stratify=all_labels[sample_ids],
    )

    valid_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=0.50,
        random_state=seed,
        stratify=all_labels[holdout_ids],
    )

    return train_ids, valid_ids, test_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--known-mode",
        choices=["random", "first-n"],
        default="random",
        help="random: protocol đã sửa; first-n: gần với setup tham khảo.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces, labels = load_arrays(source_dir)
    all_classes = np.sort(np.unique(labels))

    if args.known_count < 2 or args.known_count >= len(all_classes):
        raise ValueError(
            f"known-count phải trong [2, {len(all_classes) - 1}], "
            f"nhưng nhận {args.known_count}."
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

    known_train, known_valid, known_test = stratified_three_way_split(known_ids, labels, args.seed)

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

    # Validation bắt buộc: không được rò rỉ class open vào closed train.
    train_original_labels = set(
        manifest.loc[
            manifest["partition"] == "closed_train", "original_label"
        ].tolist()
    )
    open_original_labels = set(
        manifest.loc[
            manifest["partition"].isin(["open_valid", "open_test"]),
            "original_label",
        ].tolist()
    )

    assert train_original_labels.isdisjoint(open_original_labels)
    assert len(manifest) == len(labels)
    assert manifest["sample_id"].nunique() == len(labels)
    assert (
        manifest.loc[
            manifest["partition"] == "closed_train", "model_label"
        ].max()
        < unknown_model_label
    )
    assert (
        manifest.loc[
            manifest["partition"].isin(["open_valid", "open_test"]),
            "model_label",
        ]
        == unknown_model_label
    ).all()

    np.save(output_dir / "traces.npy", traces)
    np.save(output_dir / "original_labels.npy", labels)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    metadata = {
        "dataset": "USTC",
        "seed": args.seed,
        "known_mode": args.known_mode,
        "known_count": int(args.known_count),
        "known_classes": [int(value) for value in known_classes],
        "open_classes": [int(value) for value in open_classes],
        "unknown_model_label": unknown_model_label,
        "class_to_model_label": class_to_model_label,
        "split_sizes": manifest["partition"].value_counts().to_dict(),
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Đã tạo:", output_dir)
    print("Tổng số trace:", len(labels))
    print("Tổng số class:", len(all_classes))
    print("Known classes:", metadata["known_classes"])
    print("Open classes:", metadata["open_classes"])
    print("\nKích thước mỗi partition:")
    print(manifest["partition"].value_counts())
    print("\nKiểm tra leakage: PASS")


if __name__ == "__main__":
    main()