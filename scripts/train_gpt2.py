"""Huấn luyện GPT-2 trên closed-set USTC (P3).

Usage:
    python3 scripts/train_gpt2.py --config configs/ustc_gpt2_seed42.yaml
    python3 scripts/train_gpt2.py --config configs/ustc_gpt2_seed42.yaml --max-steps 20
    python3 scripts/train_gpt2.py --config configs/ustc_gpt2_seed42.yaml --determinism-check --max-steps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trafficllm.models.gpt2 import (  # noqa: E402
    build_loaders,
    build_model,
    build_tokenizer,
    fit,
)
from trafficllm.utils import (  # noqa: E402
    count_parameters,
    load_config,
    resolve_device,
    save_config,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on USTC closed-set.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="giới hạn step mỗi epoch")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def load_frames(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_cfg = config["data"]
    frame = pd.read_parquet(data_cfg["encoded_path"])
    frames = {}
    for split, key in (
        ("train", "train_partition"),
        ("valid", "valid_partition"),
        ("test", "test_partition"),
    ):
        partition = data_cfg[key]
        frames[split] = frame[frame["partition"] == partition].reset_index(drop=True)
        print(f"{split:5s} ({partition}): {len(frames[split])} mau", flush=True)
    return frames


def run(
    config: dict[str, Any], args: argparse.Namespace, save_checkpoint: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    device = resolve_device()
    set_seed(int(config["seed"]), bool(config["train"]["deterministic"]))

    tokenizer = build_tokenizer(config["model"]["name"])
    frames = load_frames(config)
    loaders = build_loaders(config, tokenizer, frames, int(config["seed"]))

    model = build_model(
        config["model"]["name"],
        int(config["model"]["num_labels"]),
        device,
        str(config["model"].get("attn_implementation", "eager")),
    )
    print(f"device={device} | trainable params={count_parameters(model) / 1e6:.1f}M", flush=True)

    checkpoint_path = None
    if save_checkpoint:
        checkpoint_path = (
            Path(config["train"]["checkpoint_dir"]) / config["train"]["checkpoint_name"]
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best, history = fit(
        config,
        model,
        loaders,
        device,
        epochs=args.epochs,
        max_steps=args.max_steps,
        checkpoint_path=checkpoint_path,
    )
    return best, history, {"device": str(device), "tokenizer_len": len(tokenizer)}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"config: {args.config} | seed={config['seed']} | amp={config['train']['amp']}", flush=True)

    if args.determinism_check:
        args.epochs = 1  # đủ để kết luận tất định, khỏi chạy 3 epoch
        signatures = []
        for run_index in (1, 2):
            best, history, _ = run(config, args)
            checksum = float(
                sum(float(value.double().sum()) for value in best["state"].values())
            )
            signatures.append(
                (history[-1]["train_loss"], history[-1]["train_accuracy"], checksum)
            )
            print(f"run {run_index}: {signatures[-1]}", flush=True)

        identical = all(
            abs(left - right) < 1e-9
            for left, right in zip(signatures[0], signatures[1], strict=True)
        )
        print(
            "DETERMINISM: " + ("PASS - hai lan chay giong het" if identical else "FAIL - ket qua khac nhau"),
            flush=True,
        )
        raise SystemExit(0 if identical else 1)

    best, history, runtime = run(config, args, save_checkpoint=not args.no_save)

    if args.no_save:
        print("--no-save: khong ghi checkpoint", flush=True)
        return

    checkpoint_dir = Path(config["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config["train"]["checkpoint_name"]
    torch.save(
        {
            "model_state": best["state"],
            "config": config,
            "seed": int(config["seed"]),
            "best_epoch": best["epoch"],
            "best_valid_accuracy": best["accuracy"],
            "history": history,
        },
        checkpoint_path,
    )
    print(f"da luu checkpoint -> {checkpoint_path}", flush=True)

    save_config(config, checkpoint_path.with_suffix(".config.yaml"))

    write_json(
        Path(config["metrics"]["output_path"]),
        {
            "experiment": config["experiment"],
            "phase": "P3",
            "seed": config["seed"],
            "amp": config["train"]["amp"],
            "history": history,
            "best_epoch": best["epoch"],
            "best_valid_accuracy": best["accuracy"],
            "checkpoint": str(checkpoint_path),
            **runtime,
        },
    )
    print(f"da luu metrics -> {config['metrics']['output_path']}", flush=True)


if __name__ == "__main__":
    main()
