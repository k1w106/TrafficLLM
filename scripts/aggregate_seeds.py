from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trafficllm.utils import write_json  # noqa: E402

SCALAR_METRICS = (
    "closed_argmax_accuracy",
    "CCR",
    "unknown_recall",
    "FPR",
    "TNR",
    "AUROC",
    "AUPR",
    "FPR@95TPR",
    "OSCR",
    "balanced_F1_natural",
    "balanced_F1_1to1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate seed results (mean +/- std).")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument(
        "--output-json", default="outputs/metrics/seed_sweep_ustc.json"
    )
    parser.add_argument(
        "--output-csv", default="outputs/metrics/seed_sweep_ustc.csv"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="bo qua seed thieu file thay vi dung",
    )
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "n": int(array.size)}


def _collect(
    per_seed: list[dict[str, Any]],
    path: tuple[str, ...],
    metric: str,
) -> dict[str, float | int]:
    values: list[float] = []
    for payload in per_seed:
        node: Any = payload
        for key in path:
            node = node[key]
        values.append(float(node[metric]))
    return _stats(values)


def main() -> None:
    args = parse_args()
    suffix = "_random_init" if args.random_init else ""
    seeds = sorted(set(args.seeds))

    openset_payloads: list[dict[str, Any]] = []
    ablation_payloads: list[dict[str, Any]] = []
    extract_payloads: list[dict[str, Any]] = []
    used_seeds: list[int] = []
    missing: list[str] = []

    for seed in seeds:
        paths = {
            "openset": REPO_ROOT / f"outputs/metrics/openset_ustc{suffix}_seed{seed}.json",
            "ablation": REPO_ROOT / f"outputs/metrics/ablation_ustc{suffix}_seed{seed}.json",
            "extract": REPO_ROOT / f"outputs/metrics/ustc_gpt2{suffix}_seed{seed}.json",
        }
        absent = [str(path) for path in paths.values() if not path.exists()]
        if absent:
            missing.extend(absent)
            if args.allow_missing:
                continue
            raise SystemExit("Thieu file:\n  " + "\n  ".join(absent))
        openset_payloads.append(json.loads(paths["openset"].read_text(encoding="utf-8")))
        ablation_payloads.append(json.loads(paths["ablation"].read_text(encoding="utf-8")))
        extract_payloads.append(json.loads(paths["extract"].read_text(encoding="utf-8")))
        used_seeds.append(seed)

    if not used_seeds:
        raise SystemExit("Khong co seed nao du file de aggregate")

    results: dict[str, Any] = {
        "experiment": f"seed_sweep_ustc{suffix}",
        "phase": "P6",
        "random_init": bool(args.random_init),
        "seeds": used_seeds,
        "n_seeds": len(used_seeds),
        "missing_files": missing,
        "std_ddof": "sample (ddof=1); 0 khi n=1",
        "openset": {},
        "ablation": {},
        "closed_accuracy": {},
    }
    rows: list[dict[str, Any]] = []

    detectors = sorted(openset_payloads[0]["detectors"])
    for detector in detectors:
        modes = sorted(openset_payloads[0]["detectors"][detector])
        for mode in modes:
            entry: dict[str, Any] = {}
            for metric in SCALAR_METRICS:
                entry[metric] = _collect(
                    openset_payloads, ("detectors", detector, mode), metric
                )
                rows.append(
                    {
                        "section": "openset",
                        "group": detector,
                        "mode": mode,
                        "metric": metric,
                        **entry[metric],
                    }
                )
            results["openset"].setdefault(detector, {})[mode] = entry

    features = sorted(ablation_payloads[0]["ablation"])
    for feature in features:
        detectors_for_feature = sorted(ablation_payloads[0]["ablation"][feature])
        for detector in detectors_for_feature:
            entry = {}
            for metric in SCALAR_METRICS:
                entry[metric] = _collect(
                    ablation_payloads, ("ablation", feature, detector), metric
                )
                rows.append(
                    {
                        "section": "ablation",
                        "group": feature,
                        "mode": detector,
                        "metric": metric,
                        **entry[metric],
                    }
                )
            results["ablation"].setdefault(feature, {})[detector] = entry

    for partition in ("closed_valid", "closed_test"):
        values = [
            float(payload["accuracy_by_partition"][partition]["accuracy"])
            for payload in extract_payloads
        ]
        results["closed_accuracy"][partition] = _stats(values)
        rows.append(
            {
                "section": "closed_accuracy",
                "group": partition,
                "mode": "argmax",
                "metric": "accuracy",
                **results["closed_accuracy"][partition],
            }
        )

    frame = pd.DataFrame(rows)
    write_json(args.output_json, results)
    frame.to_csv(args.output_csv, index=False)

    print(f"Seeds: {used_seeds} (n={len(used_seeds)})")
    print("\nOpen-set corrected (mean +/- std):")
    for detector in ("klnd1", "klnd2", "klnd3"):
        if detector not in results["openset"]:
            continue
        entry = results["openset"][detector]["corrected"]
        print(
            f"  {detector:6s} CCR={entry['CCR']['mean']:.4f}+/-{entry['CCR']['std']:.4f}  "
            f"UCR={entry['unknown_recall']['mean']:.4f}+/-{entry['unknown_recall']['std']:.4f}  "
            f"AUROC={entry['AUROC']['mean']:.4f}+/-{entry['AUROC']['std']:.4f}"
        )
    print("\nAblation (mean +/- std):")
    for feature in features:
        for detector in ("klnd1", "klnd2", "klnd3"):
            if detector not in results["ablation"][feature]:
                continue
            entry = results["ablation"][feature][detector]
            print(
                f"  {feature:9s} {detector:6s} "
                f"UCR={entry['unknown_recall']['mean']:.4f}+/-{entry['unknown_recall']['std']:.4f}  "
                f"AUROC={entry['AUROC']['mean']:.4f}+/-{entry['AUROC']['std']:.4f}"
            )
    print(f"\nDa luu JSON -> {args.output_json}")
    print(f"Da luu CSV  -> {args.output_csv}")


if __name__ == "__main__":
    main()
