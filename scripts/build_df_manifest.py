"""Build DF manifest and encoded parquet (streaming, crop ngay sau load).

DF upstream setup:
- 95 classes total; first 60 are known, rest (60..94) are open.
- Data loaded via cPickle encoding='latin1'; (76000/9500/9500 × 5000).
- Upstream splits known/open by class then re-splits known with
  train_test_split(shuffle=False) + shuffle; we use stratified split.
- Encoding same as AWF (save_csv_AWF).

Output:
    data/processed/df_seed{seed}/manifest.csv
    data/processed/df_seed{seed}/metadata.json
    data/processed/df_seed{seed}/encoded_gpt2_{mode}.parquet

Usage:
    PYTHONPATH=src python3 scripts/build_df_manifest.py \\
        --source-dir data/raw/trafficllm_reference/trafficllm_data/DF \\
        --output-dir data/processed/df_seed42 --seed 42 \\
        --mode corrected --known-count 60 --known-mode first-n
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _pickle as cPickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trafficllm.encoding.awf import (
    encode_awf_trace_corrected,
    encode_awf_trace_reference,
)


def load_pickle_array(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = cPickle.load(f, encoding="latin1")
    return np.array(data)


def build_manifest_and_data(
    *,
    source_dir: Path,
    seed: int,
    known_count: int,
    known_mode: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Load DF, crop to arrays, build manifest."""
    print("Loading DF labels...")
    y_train = load_pickle_array(source_dir / "y_train_NoDef.pkl").astype(int)
    y_valid = load_pickle_array(source_dir / "y_valid_NoDef.pkl").astype(int)
    y_test = load_pickle_array(source_dir / "y_test_NoDef.pkl").astype(int)
    all_labels = np.concatenate([y_train, y_valid, y_test])
    n_total = len(all_labels)

    all_classes = np.sort(np.unique(all_labels))
    if known_mode == "first-n":
        known_classes = all_classes[:known_count]
    else:
        rng = np.random.default_rng(seed)
        known_classes = np.sort(
            rng.choice(all_classes, size=known_count, replace=False)
        )
    open_classes = np.setdiff1d(all_classes, known_classes)
    n_known_classes = len(known_classes)
    unknown_model_label = n_known_classes
    class_to_model_label = {int(c): i for i, c in enumerate(known_classes)}

    known_mask = np.isin(all_labels, known_classes)
    known_ids = np.flatnonzero(known_mask)
    open_ids = np.flatnonzero(~known_mask)

    known_train, known_holdout = train_test_split(
        known_ids, test_size=0.30, random_state=seed,
        stratify=all_labels[known_ids],
    )
    known_valid, known_test = train_test_split(
        known_holdout, test_size=0.50, random_state=seed,
        stratify=all_labels[known_holdout],
    )
    open_valid, open_test = train_test_split(
        open_ids, test_size=0.50, random_state=seed,
        stratify=all_labels[open_ids],
    )

    rows = []
    for ids, partition in [
        (known_train, "closed_train"),
        (known_valid, "closed_valid"),
        (known_test, "closed_test"),
        (open_valid, "open_valid"),
        (open_test, "open_test"),
    ]:
        for sid in ids:
            orig = int(all_labels[sid])
            rows.append({
                "sample_id": int(sid),
                "partition": partition,
                "original_label": orig,
                "model_label": class_to_model_label.get(orig, unknown_model_label),
                "is_known": orig in class_to_model_label,
            })

    manifest = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    metadata = {
        "dataset": "DF",
        "source_dir": str(source_dir),
        "seed": seed,
        "known_mode": known_mode,
        "known_count": known_count,
        "n_known_classes": n_known_classes,
        "known_classes": [int(c) for c in known_classes],
        "open_classes": [int(c) for c in open_classes],
        "unknown_model_label": unknown_model_label,
        "class_to_model_label": class_to_model_label,
        "n_samples": n_total,
        "n_train_raw": int(len(y_train)),
        "n_valid_raw": int(len(y_valid)),
        "n_test_raw": int(len(y_test)),
        "split_sizes": manifest["partition"].value_counts().to_dict(),
    }

    return all_labels, manifest, metadata


def encode_and_save(
    *,
    source_dir: Path,
    manifest: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
    mode: str,
) -> None:
    encode_fn = (
        encode_awf_trace_reference if mode == "reference"
        else encode_awf_trace_corrected
    )
    n_train_raw = metadata["n_train_raw"]
    n_valid_raw = metadata["n_valid_raw"]

    sample_ids_needed = set(manifest["sample_id"])
    id_to_idx = {sid: idx for idx, sid in enumerate(manifest["sample_id"])}
    all_texts = [""] * len(manifest)

    def encode_split(pkl_name: str, offset: int, label: str):
        print(f"  Loading {pkl_name}...")
        X = load_pickle_array(source_dir / pkl_name)
        count = 0
        for i in range(len(X)):
            sid = offset + i
            if sid not in sample_ids_needed:
                continue
            row = np.array(X[i])
            all_texts[id_to_idx[sid]] = encode_fn(row)
            count += 1
            if count % 10000 == 0:
                print(f"  {label}: {count} encoded")
        del X
        gc.collect()
        print(f"  {label}: done ({count} encoded)")

    print(f"Encoding DF ({mode})...")
    encode_split("X_train_NoDef.pkl", 0, "train")
    encode_split("X_valid_NoDef.pkl", n_train_raw, "valid")
    encode_split("X_test_NoDef.pkl", n_train_raw + n_valid_raw, "test")

    result = manifest.copy()
    result["text"] = all_texts

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    result["token_length"] = result["text"].apply(lambda t: len(tok.encode(t)))

    out_path = output_dir / f"encoded_gpt2_{mode}.parquet"
    result.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(result)} rows)")

    tl = result["token_length"]
    print(f"Token length: min={tl.min()}, p50={tl.median()}, max={tl.max()}, "
          f"mean={tl.mean():.1f}, %>1024={100*(tl>1024).mean():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DF manifest + encoding.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["reference", "corrected"], default="corrected")
    parser.add_argument("--known-count", type=int, default=60)
    parser.add_argument("--known-mode", choices=["random", "first-n"], default="first-n")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_labels, manifest, metadata = build_manifest_and_data(
        source_dir=source_dir, seed=args.seed,
        known_count=args.known_count, known_mode=args.known_mode,
    )

    manifest.to_csv(output_dir / "manifest.csv", index=False)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Manifest: {len(manifest)} rows")
    print(manifest["partition"].value_counts())

    if not args.manifest_only:
        encode_and_save(
            source_dir=source_dir,
            manifest=manifest,
            metadata=metadata,
            output_dir=output_dir,
            mode=args.mode,
        )
    del all_labels
    gc.collect()


if __name__ == "__main__":
    main()
