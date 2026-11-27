"""Build AWF manifest and encoded parquet (streaming, mmap).

AWF upstream setup:
- 200 known classes in train/valid/test (100k/40k/40k × 5000)
- X_open.npz[:100000] = open set, all assigned label 200
- Encoding: same for reference and corrected (label-free),
  except reference drops last pair (upstream bug).

Output:
    data/processed/awf_seed{seed}/manifest.csv
    data/processed/awf_seed{seed}/metadata.json
    data/processed/awf_seed{seed}/encoded_gpt2_{mode}.parquet

Usage:
    PYTHONPATH=src python3 scripts/build_awf_manifest.py \\
        --source-dir data/raw/trafficllm_reference/trafficllm_data/AWF \\
        --output-dir data/processed/awf_seed42 --seed 42 \\
        --mode corrected --open-count 100000
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trafficllm.encoding.awf import (
    encode_awf_trace_corrected,
    encode_awf_trace_reference,
)

CHUNK_SIZE = 5000


def build_manifest(
    *,
    source_dir: Path,
    seed: int,
    open_count: int,
) -> tuple[pd.DataFrame, dict]:
    y_train = np.load(source_dir / "y_train.npy").astype(int)
    y_valid = np.load(source_dir / "y_valid.npy").astype(int)
    y_test = np.load(source_dir / "y_test.npy").astype(int)
    n_train = len(y_train)
    n_valid = len(y_valid)
    n_test = len(y_test)

    all_classes = np.sort(np.unique(np.concatenate([y_train, y_valid, y_test])))
    n_known_classes = len(all_classes)
    unknown_model_label = n_known_classes
    class_to_model_label = {int(c): i for i, c in enumerate(all_classes)}

    xo = np.load(source_dir / "X_open.npz", allow_pickle=True)
    n_open_available = xo["data"].shape[0]
    n_open = min(open_count, n_open_available)
    del xo

    rows = []
    offset = 0

    def add_split(labels, partition, start_id):
        for i, lab in enumerate(labels):
            rows.append({
                "sample_id": start_id + i,
                "partition": partition,
                "original_label": int(lab),
                "model_label": class_to_model_label[int(lab)],
                "is_known": True,
            })
        return start_id + len(labels)

    offset = add_split(y_train, "closed_train", 0)
    offset = add_split(y_valid, "closed_valid", offset)
    offset = add_split(y_test, "closed_test", offset)

    rng = np.random.default_rng(seed)
    open_perm = rng.permutation(n_open)
    n_open_valid = n_open // 2
    open_valid_idx = np.sort(open_perm[:n_open_valid])
    open_test_idx = np.sort(open_perm[n_open_valid:])

    for local_i in open_valid_idx:
        rows.append({
            "sample_id": offset + int(local_i),
            "partition": "open_valid",
            "original_label": 200,
            "model_label": unknown_model_label,
            "is_known": False,
        })
    for local_i in open_test_idx:
        rows.append({
            "sample_id": offset + int(local_i),
            "partition": "open_test",
            "original_label": 200,
            "model_label": unknown_model_label,
            "is_known": False,
        })

    manifest = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    metadata = {
        "dataset": "AWF",
        "source_dir": str(source_dir),
        "seed": seed,
        "n_known_classes": n_known_classes,
        "known_classes": [int(c) for c in all_classes],
        "open_label": 200,
        "unknown_model_label": unknown_model_label,
        "class_to_model_label": class_to_model_label,
        "n_samples": len(manifest),
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "n_open": n_open,
        "open_count_requested": open_count,
        "split_sizes": manifest["partition"].value_counts().to_dict(),
    }

    return manifest, metadata


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
    n_train = metadata["n_train"]
    n_valid = metadata["n_valid"]
    n_test = metadata["n_test"]
    n_open = metadata["n_open"]

    X_train = np.load(source_dir / "X_train.npy", mmap_mode="r")
    X_valid = np.load(source_dir / "X_valid.npy", mmap_mode="r")
    X_test = np.load(source_dir / "X_test.npy", mmap_mode="r")
    xo = np.load(source_dir / "X_open.npz", allow_pickle=True)
    X_open = xo["data"][:n_open]

    all_texts = [""] * len(manifest)
    id_to_idx = {sid: idx for idx, sid in enumerate(manifest["sample_id"])}

    def encode_array(X, start_id, label):
        for i in range(len(X)):
            sid = start_id + i
            if sid not in id_to_idx:
                continue
            row = np.array(X[i])
            all_texts[id_to_idx[sid]] = encode_fn(row)
            if (i + 1) % 10000 == 0:
                print(f"  {label}: {i+1}/{len(X)}")
        print(f"  {label}: done ({len(X)})")

    print(f"Encoding AWF ({mode})...")
    encode_array(X_train, 0, "train")
    encode_array(X_valid, n_train, "valid")
    encode_array(X_test, n_train + n_valid, "test")
    encode_array(X_open, n_train + n_valid + n_test, "open")
    del X_train, X_valid, X_test, X_open, xo
    gc.collect()

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
    parser = argparse.ArgumentParser(description="Build AWF manifest + encoding.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["reference", "corrected"], default="corrected")
    parser.add_argument("--open-count", type=int, default=100000)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, metadata = build_manifest(
        source_dir=source_dir, seed=args.seed, open_count=args.open_count,
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


if __name__ == "__main__":
    main()
