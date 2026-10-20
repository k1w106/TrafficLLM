"""Phân tích phân bố độ dài token của USTC đã mã hoá cho GPT-2.

Usage:
    python3 scripts/analyze_token_length.py \
        --parquet data/processed/ustc_seed42/encoded_gpt2.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARTITION_ORDER = [
    "closed_train",
    "closed_valid",
    "closed_test",
    "open_valid",
    "open_test",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token length report for USTC.")
    parser.add_argument(
        "--parquet", default="data/processed/ustc_seed42/encoded_gpt2.parquet"
    )
    parser.add_argument("--report", default="outputs/metrics/token_length_report.csv")
    parser.add_argument("--figure", default="outputs/figures/token_length.png")
    parser.add_argument("--threshold", type=int, default=1024)
    return parser.parse_args()


def summarize(name: str, lengths: np.ndarray, threshold: int) -> dict:
    return {
        "partition": name,
        "n": int(len(lengths)),
        "min": int(lengths.min()),
        "p50": int(np.percentile(lengths, 50)),
        "p90": int(np.percentile(lengths, 90)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
        "pct_over_threshold": round(100.0 * float((lengths > threshold).mean()), 4),
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.parquet)

    rows = []
    for name in PARTITION_ORDER:
        if name not in set(frame["partition"]):
            continue
        lengths = frame.loc[frame["partition"] == name, "token_length"].to_numpy()
        rows.append(summarize(name, lengths, args.threshold))
    rows.append(summarize("ALL", frame["token_length"].to_numpy(), args.threshold))

    report = pd.DataFrame(rows)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print(report.to_string(index=False))
    print(f"\nĐã ghi báo cáo -> {report_path}")

    figure_path = Path(args.figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(frame["token_length"], bins=50, color="#4C78A8")
    axes[0].axvline(
        args.threshold, color="#E45756", linestyle="--", label=f"max_len = {args.threshold}"
    )
    axes[0].set_xlabel("token length")
    axes[0].set_ylabel("số trace")
    axes[0].set_title("Độ dài token (toàn bộ USTC)")
    axes[0].legend()

    for name in PARTITION_ORDER:
        if name not in set(frame["partition"]):
            continue
        subset = frame.loc[frame["partition"] == name, "token_length"]
        axes[1].hist(subset, bins=50, histtype="step", label=name)
    axes[1].set_xlabel("token length")
    axes[1].set_ylabel("số trace")
    axes[1].set_title("Theo partition")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    print(f"Đã ghi hình -> {figure_path}")


if __name__ == "__main__":
    main()
