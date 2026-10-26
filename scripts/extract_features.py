"""Trích và cache feature cho toàn bộ 97.115 mẫu USTC (P3).

Chạy model **một lần**, lưu logits + embedding để P4 chạy detector trên cache.

Usage:
    python3 scripts/extract_features.py --config configs/ustc_gpt2_seed42.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trafficllm.models.gpt2 import (  # noqa: E402
    DynamicPadCollator,
    TraceDataset,
    build_model,
    build_tokenizer,
    extract_features,
    load_checkpoint,
)
from trafficllm.utils import load_config, resolve_device, set_seed, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache GPT-2 features for USTC.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def partition_accuracy(
    frame: pd.DataFrame, logits: np.ndarray, label_column: str, num_labels: int
) -> dict[str, dict[str, float | int | str | None]]:
    """Accuracy theo từng partition, chỉ tính cho partition closed-set.

    Partition chứa nhãn ``num_labels`` (unknown) là open-set: ``argmax`` trên
    ``num_labels`` lớp không bao giờ trả về nhãn unknown nên accuracy luôn bằng 0
    và không có ý nghĩa. P4 sẽ đánh giá open-set bằng unknown recall / AUROC.
    """
    result: dict[str, dict[str, float | int | str | None]] = {}
    predicted = logits.argmax(axis=1)
    for partition, group in frame.groupby("partition", sort=False):
        indices = group.index.to_numpy()
        labels = group[label_column].to_numpy()
        entry: dict[str, float | int | str | None] = {"n": int(len(group))}
        if bool((labels < num_labels).all()):
            entry["accuracy"] = round(float((predicted[indices] == labels).mean()), 6)
        else:
            entry["accuracy"] = None
            entry["note"] = f"open-set: nhan {num_labels} = unknown, accuracy khong co nghia"
        result[str(partition)] = entry
    return result


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    device = resolve_device()
    set_seed(int(config["seed"]), bool(config["train"]["deterministic"]))

    checkpoint_path = Path(
        args.checkpoint
        or Path(config["train"]["checkpoint_dir"]) / config["train"]["checkpoint_name"]
    )
    output_path = Path(args.output or config["features"]["output_path"])

    frame = pd.read_parquet(config["data"]["encoded_path"])
    tokenizer = build_tokenizer(config["model"]["name"])
    dataset = TraceDataset(frame, config["data"]["text_column"], config["data"]["label_column"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["features"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=DynamicPadCollator(tokenizer, int(config["model"]["max_len"])),
    )

    model = build_model(
        config["model"]["name"],
        int(config["model"]["num_labels"]),
        device,
        str(config["model"].get("attn_implementation", "eager")),
    )
    payload = load_checkpoint(model, checkpoint_path, device)
    print(
        f"checkpoint: {checkpoint_path} (epoch {payload['best_epoch']}, "
        f"valid_acc {payload['best_valid_accuracy']:.4f})",
        flush=True,
    )

    started = time.time()
    sample_ids, logits, emb_last, emb_mean = extract_features(
        model, loader, device, amp=str(config["train"]["amp"])
    )
    elapsed = time.time() - started
    print(f"da trich {len(sample_ids)} mau trong {elapsed / 60:.1f} phut", flush=True)

    expected_ids = frame["sample_id"].to_numpy()
    if not np.array_equal(sample_ids, expected_ids):
        raise SystemExit("sample_id khong khop thu tu manifest - dung lai")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sample_id=sample_ids.astype(np.int64),
        logits=logits.astype(np.float32),
        emb_last=emb_last.astype(np.float32),
        emb_mean=emb_mean.astype(np.float32),
    )
    print(f"da luu feature -> {output_path} ({output_path.stat().st_size / 1e6:.0f} MB)", flush=True)

    metrics = {
        "experiment": config["experiment"],
        "phase": "P3-features",
        "seed": config["seed"],
        "checkpoint": str(checkpoint_path),
        "n_samples": int(len(sample_ids)),
        "n_labels": int(config["model"]["num_labels"]),
        "feature_shapes": {
            "logits": list(logits.shape),
            "emb_last": list(emb_last.shape),
            "emb_mean": list(emb_mean.shape),
        },
        "accuracy_by_partition": partition_accuracy(
            frame,
            logits,
            config["data"]["label_column"],
            int(config["model"]["num_labels"]),
        ),
        "seconds": round(elapsed, 1),
        "output": str(output_path),
    }
    write_json(config["metrics"]["output_path"], metrics)
    print(f"da luu metrics -> {config['metrics']['output_path']}", flush=True)
    for partition, values in metrics["accuracy_by_partition"].items():
        if values["accuracy"] is None:
            print(f"  {partition:12s} n={values['n']:6d} (open-set: khong tinh accuracy)")
        else:
            print(f"  {partition:12s} n={values['n']:6d} accuracy={values['accuracy']:.4f}")


if __name__ == "__main__":
    main()
