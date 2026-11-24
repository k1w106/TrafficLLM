"""Audit P5 (run phu): encoding DC reference (co nhan) vs corrected (label-free).

Hai cache feature được train riêng vì encoding khác nhau. Script so sánh:

- ``reference_upstream``: cache reference + protocol reference (keep 0,9,
  calibration correct-only, K-LND bat doi xung nhu upstream).
- ``reference_corrected_protocol``: **cùng cache reference** nhưng chạy protocol
  corrected (keep 0,95, calibration toàn bộ valid, K-LND cùng nhánh).
- ``corrected``: cache corrected + protocol corrected.

Nhờ vậy ``corrected - reference_corrected_protocol`` đo riêng tác động của
label leakage trong encoding, không lẫn với sửa protocol.

Usage (sau khi user đã train + extract feature):
    python3 scripts/run_dc_audit.py --config configs/audit_dc_seed42.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trafficllm.openset.openmax import fit_openmax, openmax_anomaly  # noqa: E402
from trafficllm.openset.pipeline import (  # noqa: E402
    compute_metrics,
    klnd_scores,
    partition_indices,
)
from trafficllm.openset.scoring import (  # noqa: E402
    class_means,
    energy_anomaly,
    msp_anomaly,
    neighbor_sets,
)
from trafficllm.utils import load_config, set_seed, write_json  # noqa: E402

SCALAR_METRICS = (
    "closed_argmax_accuracy",
    "CCR",
    "unknown_recall",
    "FPR",
    "TNR",
    "AUROC",
    "AUPR",
    "prevalence",
    "FPR@95TPR",
    "OSCR",
    "balanced_F1_natural",
    "balanced_F1_1to1",
    "known_rejection_on_valid",
)

DELTA_METRICS = (
    "closed_argmax_accuracy",
    "CCR",
    "unknown_recall",
    "FPR",
    "AUROC",
    "FPR@95TPR",
    "OSCR",
    "balanced_F1_1to1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DC encoding.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def metric_kwargs(
    *,
    valid_logits: np.ndarray,
    known_logits: np.ndarray,
    open_logits: np.ndarray,
    valid_labels: np.ndarray,
    known_labels: np.ndarray,
    open_original_labels: np.ndarray,
    num_labels: int,
    unknown_label: int,
    mode: str,
    keep_rate: float,
    magic_empty: bool,
    correct_only: bool,
    balanced_seed: int,
    valid_scores: np.ndarray,
    known_scores: np.ndarray,
    open_scores: np.ndarray,
) -> dict[str, Any]:
    return compute_metrics(
        valid_scores=valid_scores,
        known_scores=known_scores,
        open_scores=open_scores,
        valid_logits=valid_logits,
        known_logits=known_logits,
        open_logits=open_logits,
        valid_labels=valid_labels,
        known_labels=known_labels,
        open_original_labels=open_original_labels,
        num_labels=num_labels,
        unknown_label=unknown_label,
        mode=mode,
        keep_rate=keep_rate,
        magic_empty=magic_empty,
        correct_only_calibration=correct_only,
        balanced_seed=balanced_seed,
    )


def evaluate_cache(
    *,
    cache_path: Path,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    keep_rate: float,
    magic_empty: bool,
    correct_only: bool,
    include_openmax: bool,
) -> dict[str, Any]:
    data_cfg = config["data"]
    open_cfg = config["openset"]
    num_labels = int(data_cfg["num_labels"])
    unknown_label = int(data_cfg["unknown_label"])
    k = int(open_cfg["neighbors_k"])
    temperature = float(open_cfg["temperature"])
    balanced_seed = int(open_cfg["balanced_f1_seed"])

    cache = np.load(cache_path, mmap_mode="r")
    if not np.array_equal(cache["sample_id"], manifest["sample_id"].to_numpy()):
        raise SystemExit(f"sample_id cua {cache_path} khong khop manifest")

    indices = partition_indices(manifest)
    logits = np.asarray(cache["logits"])
    embeddings = np.asarray(cache["emb_last"])
    labels = manifest["model_label"].to_numpy()
    original_labels = manifest["original_label"].to_numpy()

    train_idx = indices["closed_train"]
    valid_idx = indices["closed_valid"]
    known_idx = indices["closed_test"]
    open_idx = indices["open_test"]

    valid_logits = np.asarray(logits[valid_idx])
    known_logits = np.asarray(logits[known_idx])
    open_logits = np.asarray(logits[open_idx])
    valid_labels = labels[valid_idx]
    known_labels = labels[known_idx]
    open_original_labels = original_labels[open_idx]

    def build(
        valid_scores: np.ndarray,
        known_scores: np.ndarray,
        open_scores: np.ndarray,
        *,
        klnd: bool,
    ) -> dict[str, Any]:
        return metric_kwargs(
            valid_scores=valid_scores,
            known_scores=known_scores,
            open_scores=open_scores,
            valid_logits=valid_logits,
            known_logits=known_logits,
            open_logits=open_logits,
            valid_labels=valid_labels,
            known_labels=known_labels,
            open_original_labels=open_original_labels,
            num_labels=num_labels,
            unknown_label=unknown_label,
            mode=mode,
            keep_rate=keep_rate,
            magic_empty=magic_empty if klnd else False,
            correct_only=correct_only if klnd else False,
            balanced_seed=balanced_seed,
        )

    detectors: dict[str, dict[str, Any]] = {}
    for name, score_fn in (("msp", msp_anomaly), ("energy", energy_anomaly)):
        detectors[name] = build(
            score_fn(valid_logits, temperature),
            score_fn(known_logits, temperature),
            score_fn(open_logits, temperature),
            klnd=False,
        )

    means, counts = class_means(
        np.asarray(logits[train_idx]), labels[train_idx], num_labels, mode=mode
    )
    neighbors = neighbor_sets(
        valid_logits, valid_labels, means, num_labels, k=k, mode=mode
    )
    for method in (1, 2, 3):
        valid_scores, known_scores, open_scores = klnd_scores(
            mode,
            method,
            valid_logits,
            known_logits,
            open_logits,
            means,
            neighbors,
        )
        detectors[f"klnd{method}"] = build(
            valid_scores, known_scores, open_scores, klnd=True
        )

    if include_openmax:
        model = fit_openmax(
            np.asarray(embeddings[train_idx]),
            np.asarray(logits[train_idx]),
            labels[train_idx],
            num_labels=num_labels,
            alpha=int(open_cfg["openmax"]["alpha"]),
            min_class_samples=int(open_cfg["openmax"]["min_class_samples"]),
        )
        detectors["openmax"] = build(
            openmax_anomaly(np.asarray(embeddings[valid_idx]), valid_logits, model),
            openmax_anomaly(np.asarray(embeddings[known_idx]), known_logits, model),
            openmax_anomaly(np.asarray(embeddings[open_idx]), open_logits, model),
            klnd=False,
        )

    return {
        "cache": str(cache_path),
        "protocol": {
            "mode": mode,
            "keep_rate": float(keep_rate),
            "magic_empty": bool(magic_empty),
            "correct_only_calibration": bool(correct_only),
            "neighbors_k": k,
        },
        "class_counts": counts.tolist(),
        "neighbor_sizes": neighbors.sum(axis=1).tolist(),
        "detectors": detectors,
    }


def subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in DELTA_METRICS}


def delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {key: float(right[key]) - float(left[key]) for key in DELTA_METRICS}


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    set_seed(int(config["seed"]))
    data_cfg = config["data"]
    open_cfg = config["openset"]

    manifest = pd.read_csv(data_cfg["manifest_path"])
    reference_features = Path(data_cfg["reference_features_path"])
    corrected_features = Path(data_cfg["corrected_features_path"])

    evaluations: dict[str, Any] = {}
    evaluations["reference_upstream"] = evaluate_cache(
        cache_path=reference_features,
        manifest=manifest,
        config=config,
        mode="reference",
        keep_rate=float(open_cfg["reference_keep_rate"]),
        magic_empty=bool(open_cfg.get("reference_magic_empty", True)),
        correct_only=True,
        include_openmax=False,
    )
    evaluations["reference_corrected_protocol"] = evaluate_cache(
        cache_path=reference_features,
        manifest=manifest,
        config=config,
        mode="corrected",
        keep_rate=float(open_cfg["corrected_keep_rate"]),
        magic_empty=False,
        correct_only=False,
        include_openmax=True,
    )
    evaluations["corrected"] = evaluate_cache(
        cache_path=corrected_features,
        manifest=manifest,
        config=config,
        mode="corrected",
        keep_rate=float(open_cfg["corrected_keep_rate"]),
        magic_empty=False,
        correct_only=False,
        include_openmax=True,
    )

    detectors = sorted(
        set(evaluations["corrected"]["detectors"])
        | set(evaluations["reference_corrected_protocol"]["detectors"])
    )
    summary: dict[str, Any] = {}
    for detector in detectors:
        entry: dict[str, Any] = {}
        if detector in evaluations["reference_upstream"]["detectors"]:
            entry["reference_upstream"] = subset(
                evaluations["reference_upstream"]["detectors"][detector]
            )
        if detector in evaluations["reference_corrected_protocol"]["detectors"]:
            entry["reference_corrected_protocol"] = subset(
                evaluations["reference_corrected_protocol"]["detectors"][detector]
            )
        if detector in evaluations["corrected"]["detectors"]:
            entry["corrected"] = subset(
                evaluations["corrected"]["detectors"][detector]
            )
        if (
            "reference_corrected_protocol" in entry
            and "corrected" in entry
        ):
            entry["delta_encoding_isolated"] = delta(
                evaluations["reference_corrected_protocol"]["detectors"][detector],
                evaluations["corrected"]["detectors"][detector],
            )
        if "reference_upstream" in entry and "corrected" in entry:
            entry["delta_upstream_vs_corrected"] = delta(
                evaluations["reference_upstream"]["detectors"][detector],
                evaluations["corrected"]["detectors"][detector],
            )
        summary[detector] = entry

    token_length: dict[str, Any] = {}
    for name in ("reference", "corrected"):
        path = Path(data_cfg["manifest_path"]).parent / f"token_length_{name}.csv"
        if path.exists():
            token_length[name] = pd.read_csv(path).to_dict(orient="records")

    results = {
        "experiment": config["experiment"],
        "phase": config.get("phase", "P5"),
        "seed": int(config["seed"]),
        "config_path": args.config,
        "data": {
            "manifest": data_cfg["manifest_path"],
            "num_labels": int(data_cfg["num_labels"]),
            "unknown_label": int(data_cfg["unknown_label"]),
            "partition_sizes": {
                partition: int((manifest["partition"] == partition).sum())
                for partition in (
                    "closed_train",
                    "closed_valid",
                    "closed_test",
                    "open_valid",
                    "open_test",
                )
            },
        },
        "evaluations": evaluations,
        "summary": summary,
        "token_length": token_length,
        "notes": [
            "DC la run phu, chi de audit label leakage trong encoding.",
            "reference_upstream tai hien encoding co nhan + protocol upstream.",
            "reference_corrected_protocol dung cung cache reference nhung protocol "
            "corrected, de tach tac dong encoding khoi tac dong protocol.",
            "delta_encoding_isolated = corrected - reference_corrected_protocol.",
            "OpenMax khong co trong upstream; chi xuat hien o hai evaluation "
            "corrected protocol.",
            "open_valid khong duoc dung.",
        ],
    }

    audit_cfg = config.get("audit", {})
    output_json = Path(
        audit_cfg.get("output_json", "outputs/metrics/audit_dc_seed42.json")
    )
    output_csv = Path(
        audit_cfg.get("output_csv", "outputs/metrics/audit_dc_seed42.csv")
    )
    write_json(output_json, results)

    rows: list[dict[str, Any]] = []
    for evaluation, payload in evaluations.items():
        for detector, metrics in payload["detectors"].items():
            for key in SCALAR_METRICS:
                rows.append(
                    {
                        "evaluation": evaluation,
                        "detector": detector,
                        "metric": key,
                        "value": metrics[key],
                    }
                )
            for class_id, value in metrics["open_class_detection"].items():
                rows.append(
                    {
                        "evaluation": evaluation,
                        "detector": detector,
                        "metric": f"open_class_{class_id}",
                        "value": value,
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print("\nDC audit (CCR / UCR / AUROC / FPR95 / OSCR):", flush=True)
    print(
        f"{'evaluation':30s} {'detector':10s} {'CCR':>8s} {'UCR':>8s} "
        f"{'AUROC':>8s} {'FPR95':>8s} {'OSCR':>8s}",
        flush=True,
    )
    for evaluation, payload in evaluations.items():
        for detector, metrics in payload["detectors"].items():
            print(
                f"{evaluation:30s} {detector:10s} {metrics['CCR']:8.4f} "
                f"{metrics['unknown_recall']:8.4f} {metrics['AUROC']:8.4f} "
                f"{metrics['FPR@95TPR']:8.4f} {metrics['OSCR']:8.4f}",
                flush=True,
            )

    print("\nDelta encoding (corrected - reference_corrected_protocol):", flush=True)
    for detector, entry in summary.items():
        if "delta_encoding_isolated" in entry:
            values = entry["delta_encoding_isolated"]
            print(
                f"  {detector:10s} dCCR={values['CCR']:+.4f} "
                f"dUCR={values['unknown_recall']:+.4f} "
                f"dAUROC={values['AUROC']:+.4f} "
                f"dFPR95={values['FPR@95TPR']:+.4f} "
                f"dOSCR={values['OSCR']:+.4f}",
                flush=True,
            )
    print(f"\nDa luu JSON -> {output_json}", flush=True)
    print(f"Da luu CSV  -> {output_csv}", flush=True)


if __name__ == "__main__":
    main()
