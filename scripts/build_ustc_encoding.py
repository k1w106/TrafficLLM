"""Build the encoded USTC dataset (GPT-2 character-level) for the audit.

Input : data/processed/ustc_seed42/{traces.npy, manifest.csv}
Output: data/processed/ustc_seed42/encoded_gpt2.parquet  (97.115 rows)

Usage:
    python3 scripts/build_ustc_encoding.py --data-dir data/processed/ustc_seed42
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

from trafficllm.encoding.ustc import encode_trace_gpt2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode USTC traces for GPT-2.")
    parser.add_argument("--data-dir", default="data/processed/ustc_seed42")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.output) if args.output else data_dir / "encoded_gpt2.parquet"

    traces = np.load(data_dir / "traces.npy", allow_pickle=True)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    if len(traces) != len(manifest):
        raise SystemExit(f"traces={len(traces)} khác manifest rows={len(manifest)}")

    start = time.time()
    texts = [encode_trace_gpt2(str(trace)) for trace in traces]
    print(f"Đã mã hoá {len(texts)} trace trong {time.time() - start:.1f}s")

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    lengths = np.empty(len(texts), dtype=np.int32)
    start = time.time()
    for offset in range(0, len(texts), args.batch_size):
        chunk = texts[offset : offset + args.batch_size]
        encoded = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        lengths[offset : offset + len(chunk)] = [len(ids) for ids in encoded]
    print(f"Đã tokenize trong {time.time() - start:.1f}s")

    encoded_df = manifest.copy()
    encoded_df["text"] = texts
    encoded_df["token_length"] = lengths

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_parquet(out_path, index=False)

    print(f"Đã ghi {len(encoded_df)} dòng -> {out_path}")
    print(
        "token length: min={0} max={1} mean={2:.1f}".format(
            lengths.min(), lengths.max(), lengths.mean()
        )
    )
    print(f"% vượt 1024 token: {100.0 * (lengths > 1024).mean():.4f}%")


if __name__ == "__main__":
    main()
