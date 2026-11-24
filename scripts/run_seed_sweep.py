"""Sweep nhiều seed cho USTC: train -> extract -> openset -> ablation -> aggregate (P6).

Config sinh tự động vào ``outputs/generated_configs/`` để không sửa config gốc.
Mặc định chạy GPT-2 pretrained; ``--random-init`` dùng đối chứng random-init.

Usage:
    python3 scripts/run_seed_sweep.py --seeds 42 43 44 --dry-run
    python3 scripts/run_seed_sweep.py --seeds 43 44
    python3 scripts/run_seed_sweep.py --seeds 42 43 44 --random-init --dry-run
    python3 scripts/run_seed_sweep.py --seeds 42 43 44 --stages aggregate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = REPO_ROOT / "outputs/generated_configs"
STAGES = ("train", "extract", "openset", "ablation", "audit", "aggregate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed sweep for USTC.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["train", "extract", "openset", "ablation", "aggregate"],
        choices=STAGES,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def _replace_seed(value: Any, seed: int) -> Any:
    if isinstance(value, dict):
        return {
            key: (seed if key in {"seed", "balanced_f1_seed"} else _replace_seed(item, seed))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_seed(item, seed) for item in value]
    if isinstance(value, str):
        return value.replace("seed42", f"seed{seed}")
    return value


def build_configs(seed: int, random_init: bool) -> dict[str, Path]:
    suffix = "_random_init" if random_init else ""
    base_train = REPO_ROOT / (
        "configs/ustc_gpt2_random_init_seed42.yaml"
        if random_init
        else "configs/ustc_gpt2_seed42.yaml"
    )
    base_openset = REPO_ROOT / "configs/openset_ustc_seed42.yaml"
    base_ablation = REPO_ROOT / "configs/ablation_ustc_seed42.yaml"

    train = _replace_seed(_load(base_train), seed)
    train["seed"] = seed
    train["experiment"] = f"ustc_gpt2{suffix}_seed{seed}"
    train["model"]["pretrained"] = not random_init
    train["train"]["checkpoint_name"] = f"ustc_gpt2{suffix}_seed{seed}.pt"
    train["features"]["output_path"] = f"outputs/features/ustc{suffix}_seed{seed}.npz"
    train["metrics"]["output_path"] = f"outputs/metrics/ustc_gpt2{suffix}_seed{seed}.json"

    openset = _replace_seed(_load(base_openset), seed)
    openset["seed"] = seed
    openset["experiment"] = f"openset_ustc{suffix}_seed{seed}"
    openset["data"]["features_path"] = train["features"]["output_path"]
    openset["metrics"]["output_json"] = f"outputs/metrics/openset_ustc{suffix}_seed{seed}.json"
    openset["metrics"]["output_csv"] = f"outputs/metrics/openset_ustc{suffix}_seed{seed}.csv"
    openset["audit"]["output_json"] = f"outputs/metrics/audit_ustc{suffix}_seed{seed}.json"
    openset["audit"]["output_csv"] = f"outputs/metrics/audit_ustc{suffix}_seed{seed}.csv"

    ablation = _replace_seed(_load(base_ablation), seed)
    ablation["seed"] = seed
    ablation["experiment"] = f"ablation_features_ustc{suffix}_seed{seed}"
    ablation["data"]["features_path"] = train["features"]["output_path"]
    ablation["output"]["json_path"] = f"outputs/metrics/ablation_ustc{suffix}_seed{seed}.json"
    ablation["output"]["csv_path"] = f"outputs/metrics/ablation_ustc{suffix}_seed{seed}.csv"

    paths = {
        "train": GENERATED_DIR / f"ustc_gpt2{suffix}_seed{seed}.yaml",
        "openset": GENERATED_DIR / f"openset_ustc{suffix}_seed{seed}.yaml",
        "ablation": GENERATED_DIR / f"ablation_ustc{suffix}_seed{seed}.yaml",
    }
    _write(train, paths["train"])
    _write(openset, paths["openset"])
    _write(ablation, paths["ablation"])
    return paths


def commands_for(seed: int, paths: dict[str, Path], stages: list[str]) -> list[list[str]]:
    python = sys.executable
    commands: list[list[str]] = []
    if "train" in stages:
        commands.append([python, "scripts/train_gpt2.py", "--config", str(paths["train"])])
    if "extract" in stages:
        commands.append([python, "scripts/extract_features.py", "--config", str(paths["train"])])
    if "openset" in stages:
        commands.append([python, "scripts/run_openset.py", "--config", str(paths["openset"])])
    if "ablation" in stages:
        commands.append([python, "scripts/run_ablation.py", "--config", str(paths["ablation"])])
    if "audit" in stages:
        commands.append([python, "scripts/run_audit.py", "--config", str(paths["openset"])])
    return commands


def main() -> None:
    args = parse_args()
    seeds = sorted(set(args.seeds))
    stages = list(dict.fromkeys(args.stages))

    plan: list[tuple[int, list[list[str]]]] = []
    for seed in seeds:
        paths = build_configs(seed, args.random_init)
        plan.append((seed, commands_for(seed, paths, stages)))

    aggregate_command = [
        sys.executable,
        "scripts/aggregate_seeds.py",
        "--seeds",
        *[str(seed) for seed in seeds],
    ]
    if args.random_init:
        aggregate_command.append("--random-init")

    print(f"Seeds: {seeds} | random_init={args.random_init} | stages={stages}")
    print(f"Generated configs: {GENERATED_DIR.relative_to(REPO_ROOT)}")
    for seed, commands in plan:
        print(f"\n[seed {seed}]")
        for command in commands:
            print("  " + " ".join(command))
    if "aggregate" in stages:
        print("\n[aggregate]")
        print("  " + " ".join(aggregate_command))

    if args.dry_run:
        return

    for seed, commands in plan:
        for command in commands:
            print(f"\n>>> seed {seed}: {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)

    if "aggregate" in stages:
        print(f"\n>>> aggregate: {' '.join(aggregate_command)}", flush=True)
        subprocess.run(aggregate_command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
