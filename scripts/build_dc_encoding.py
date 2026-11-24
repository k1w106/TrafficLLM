"""Build encoded DC dataset cho audit P5.

``--mode reference`` tái hiện encoding có nhãn của upstream (label leakage).
``--mode corrected`` dùng encoding label-free, pattern cố định.

Output mặc định:
    data/processed/dc_seed42/encoded_reference_gpt2.parquet
    data/processed/dc_seed42/encoded_corrected_gpt2.parquet
    data/processed/dc_seed42/token_length_reference.csv
    data/processed/dc_seed42/token_length_corrected.csv

Usage:
    python3 scripts/build_dc_encoding.py --mode reference
    python3 scripts/build_dc_encoding.py --mode corrected --divisor 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trafficllm.encoding.dc import (  # noqa: E402
    DEFAULT_DIVISOR,
    encode_dc_trace_corrected,
    encode_dc_trace_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode DC for GPT-2.")
    parser.add_argument("--data-dir", default="data/processed/dc_seed42")
    parser.add_argument("--mode", choices=["reference", "corrected"], required=True)
    parser.add_argument("--divisor", type=int, default=DEFAULT_DIVISOR)
    parser.add_argument("--pattern", default="left")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def token_length_report(
    frame: pd.DataFrame, max_len: int
) -> pd.DataFrame:
    rows = []
    for partition, group in frame.groupby("partition", sort=False):
        lengths = group["token_length"].to_numpy()
        rows.append(
            {
                "partition": partition,
                "n": int(len(lengths)),
                "min": int(lengths.min()),
                "p50": float(np.quantile(lengths, 0.50)),
                "p95": float(np.quantile(lengths, 0.95)),
                "p99": float(np.quantile(lengths, 0.99)),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
                "pct_over_max_len": float(100.0 * np.mean(lengths > max_len)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output = Path(
        args.output
        or data_dir
        / f"encoded_{args.mode}_gpt2.parquet"
    )
    report = Path(
        args.report
        or data_dir
        / f"token_length_{args.mode}.csv"
    )

    values = np.load(data_dir / "X.npy")
    manifest = pd.read_csv(data_dir / "manifest.csv")
    if len(values) != len(manifest):
        raise SystemExit(f"X={len(values)} khac manifest rows={len(manifest)}")

    order = manifest["sample_id"].to_numpy()
    values = values[order]
    labels = manifest["original_label"].to_numpy()

    started = time.time()
    if args.mode == "reference":
        print(
            "CANH BAO: mode=reference dung nhan goc trong encoding (label leakage), "
            "chi de audit.",
            flush=True,
        )
        texts = [
            encode_dc_trace_reference(row, int(label), divisor=args.divisor)
            for row, label in zip(values, labels, strict=True)
        ]
    else:
        texts = [
            encode_dc_trace_corrected(row, divisor=args.divisor, pattern=args.pattern)
            for row in values
        ]
    print(f"Da ma hoa {len(texts)} trace trong {time.time() - started:.1f}s", flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    lengths = np.empty(len(texts), dtype=np.int32)
    started = time.time()
    for offset in range(0, len(texts), args.batch_size):
        chunk = texts[offset : offset + args.batch_size]
        encoded = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        lengths[offset : offset + len(chunk)] = [len(ids) for ids in encoded]
    print(f"Da tokenize trong {time.time() - started:.1f}s", flush=True)

    frame = manifest.copy()
    frame["text"] = texts
    frame["token_length"] = lengths
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)

    report_frame = token_length_report(frame, args.max_len)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(report, index=False)

    print(f"Da ghi {len(frame)} dong -> {output}", flush=True)
    print(f"Da ghi token length report -> {report}", flush=True)
    print(
        "token length: min={0} p50={1:.0f} p95={2:.0f} max={3}".format(
            int(lengths.min()),
            float(np.quantile(lengths, 0.50)),
            float(np.quantile(lengths, 0.95)),
            int(lengths.max()),
        ),
        flush=True,
    )
    print(
        f"% vuot {args.max_len} token: {100.0 * np.mean(lengths > args.max_len):.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
