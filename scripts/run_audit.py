"""Audit P5: so sánh reference vs corrected trên cùng cache feature.

Bốn biến thể operating point:

- ``reference_08``  : lỗi upstream + keep_rate 0,8 (mặc định của paper/repo).
- ``reference_095`` : lỗi upstream + keep_rate 0,95.
- ``corrected_08``  : đã sửa + keep_rate 0,8.
- ``corrected_095`` : đã sửa + keep_rate 0,95.

Nhờ vậy mỗi chênh lệch được tách thành: sửa lỗi, đổi operating point, hoặc
tương tác giữa hai yếu tố. Script cũng chạy các chẩn đoán:

- ``same_branch``: K-LND 2/3 reference nhưng closed_test dùng cùng nhánh K
  láng giềng với open_test, để lộ phần AUROC/FPR "ảo" do bất đối xứng.
- ``self_neighbor``: cùng một định nghĩa score, chỉ đổi tập láng giềng có/không
  có chính class đang xét.
- ``calibration_correct_only``: chỉ đổi calibration correct-only (upstream) sang
  toàn bộ closed_valid.
- ``msp_saturation``: kiểm tra MSP có bão hoà ở 0 trên closed_valid không.

Usage:
    python3 scripts/run_audit.py --config configs/openset_ustc_seed42.yaml
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

from trafficllm.openset.calibration import calibrate_thresholds  # noqa: E402
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
    softmax,
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
    parser = argparse.ArgumentParser(description="Audit reference vs corrected.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def variant_specs(open_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reference_keep = float(open_cfg["reference_keep_rate"])
    corrected_keep = float(open_cfg["corrected_keep_rate"])
    magic_empty = bool(open_cfg.get("reference_magic_empty", True))
    return {
        "reference_08": {
            "mode": "reference",
            "keep_rate": reference_keep,
            "magic_empty": magic_empty,
            "correct_only": True,
        },
        "reference_095": {
            "mode": "reference",
            "keep_rate": 0.95,
            "magic_empty": magic_empty,
            "correct_only": True,
        },
        "corrected_08": {
            "mode": "corrected",
            "keep_rate": reference_keep,
            "magic_empty": False,
            "correct_only": False,
        },
        "corrected_095": {
            "mode": "corrected",
            "keep_rate": corrected_keep,
            "magic_empty": False,
            "correct_only": False,
        },
    }


def subset(metrics: dict[str, Any], keys: tuple[str, ...] = DELTA_METRICS) -> dict[str, float]:
    return {key: float(metrics[key]) for key in keys}


def delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {key: float(right[key]) - float(left[key]) for key in DELTA_METRICS}


def build_metrics(
    *,
    valid_scores: np.ndarray,
    known_scores: np.ndarray,
    open_scores: np.ndarray,
    valid_logits: np.ndarray,
    known_logits: np.ndarray,
    open_logits: np.ndarray,
    valid_labels: np.ndarray,
    known_labels: np.ndarray,
    open_original_labels: np.ndarray,
    num_labels: int,
    unknown_label: int,
    spec: dict[str, Any],
    balanced_seed: int,
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
        mode=str(spec["mode"]),
        keep_rate=float(spec["keep_rate"]),
        magic_empty=bool(spec["magic_empty"]),
        correct_only_calibration=bool(spec["correct_only"]),
        balanced_seed=balanced_seed,
    )


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    open_cfg = config["openset"]
    num_labels = int(data_cfg["num_labels"])
    unknown_label = int(data_cfg["unknown_label"])
    k = int(open_cfg["neighbors_k"])
    temperature = float(open_cfg["temperature"])
    balanced_seed = int(open_cfg["balanced_f1_seed"])

    manifest = pd.read_csv(data_cfg["manifest_path"])
    cache = np.load(data_cfg["features_path"], mmap_mode="r")
    if not np.array_equal(cache["sample_id"], manifest["sample_id"].to_numpy()):
        raise SystemExit("sample_id cua cache khong khop manifest")

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

    specs = variant_specs(open_cfg)
    results: dict[str, Any] = {
        "experiment": f"audit_{config['experiment']}",
        "phase": "P5",
        "seed": int(config["seed"]),
        "config_path": args.config,
        "data": {
            "manifest": data_cfg["manifest_path"],
            "features": data_cfg["features_path"],
            "num_labels": num_labels,
            "unknown_label": unknown_label,
            "neighbors_k": k,
            "partition_sizes": {
                partition: int(len(values)) for partition, values in indices.items()
            },
        },
        "variants": {
            name: {
                "mode": spec["mode"],
                "keep_rate": float(spec["keep_rate"]),
                "magic_empty": bool(spec["magic_empty"]),
                "correct_only_calibration": bool(spec["correct_only"]),
            }
            for name, spec in specs.items()
        },
        "detectors": {},
        "diagnostics": {
            "same_branch": {},
            "self_neighbor": {},
            "calibration_correct_only": {},
            "msp_saturation": {},
        },
        "summary": {},
        "notes": [
            "reference tai hien loi upstream: K-LND closed_test cong tren tat ca class, "
            "open_test chi tren K lang gieng; calibration chi tren mau phan loai dung; "
            "class rong fallback 10/previous; phan vi nearest-rank.",
            "corrected dung cung nhanh K lang gieng cho closed/open, calibration tren "
            "toan bo closed_valid, class rong fallback centroid, phan vi noi suy.",
            "bon bien the tach tac dong cua sua loi va doi operating point.",
            "OpenMax chi co corrected vi repo goc khong cai OpenMax.",
            "open_valid khong duoc dung trong audit.",
        ],
    }

    # MSP / Energy.
    for name, score_fn in (("msp", msp_anomaly), ("energy", energy_anomaly)):
        valid_scores = score_fn(valid_logits, temperature)
        known_scores = score_fn(known_logits, temperature)
        open_scores = score_fn(open_logits, temperature)
        for variant, spec in specs.items():
            results["detectors"].setdefault(name, {})[variant] = build_metrics(
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
                spec=spec,
                balanced_seed=balanced_seed,
            )

    # K-LND 1/2/3.
    means: dict[str, np.ndarray] = {}
    neighbors: dict[str, np.ndarray] = {}
    for mode in ("reference", "corrected"):
        means[mode], counts = class_means(
            np.asarray(logits[train_idx]), labels[train_idx], num_labels, mode=mode
        )
        neighbors[mode] = neighbor_sets(
            valid_logits, valid_labels, means[mode], num_labels, k=k, mode=mode
        )
        results["data"].setdefault("class_counts", {})[mode] = counts.tolist()
        results["data"].setdefault("neighbor_sizes", {})[mode] = (
            neighbors[mode].sum(axis=1).tolist()
        )
        print(
            f"K-LND {mode}: class counts={counts.tolist()} | "
            f"neighbor sizes={neighbors[mode].sum(axis=1).tolist()}",
            flush=True,
        )

    for mode in ("reference", "corrected"):
        for method in (1, 2, 3):
            valid_scores, known_scores, open_scores = klnd_scores(
                mode,
                method,
                valid_logits,
                known_logits,
                open_logits,
                means[mode],
                neighbors[mode],
            )
            for variant, spec in specs.items():
                if spec["mode"] != mode:
                    continue
                results["detectors"].setdefault(f"klnd{method}", {})[variant] = build_metrics(
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
                    spec=spec,
                    balanced_seed=balanced_seed,
                )

    # OpenMax: chi corrected.
    if bool(open_cfg["openmax"]["enabled"]):
        model = fit_openmax(
            np.asarray(embeddings[train_idx]),
            np.asarray(logits[train_idx]),
            labels[train_idx],
            num_labels=num_labels,
            alpha=int(open_cfg["openmax"]["alpha"]),
            min_class_samples=int(open_cfg["openmax"]["min_class_samples"]),
        )
        valid_scores = openmax_anomaly(np.asarray(embeddings[valid_idx]), valid_logits, model)
        known_scores = openmax_anomaly(np.asarray(embeddings[known_idx]), known_logits, model)
        open_scores = openmax_anomaly(np.asarray(embeddings[open_idx]), open_logits, model)
        for variant in ("corrected_08", "corrected_095"):
            results["detectors"].setdefault("openmax", {})[variant] = build_metrics(
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
                spec=specs[variant],
                balanced_seed=balanced_seed,
            )

    # Chan doan 1: cung nhanh cho K-LND 2/3 reference.
    for method in (2, 3):
        valid_scores, known_scores, open_scores = klnd_scores(
            "reference",
            method,
            valid_logits,
            known_logits,
            open_logits,
            means["reference"],
            neighbors["reference"],
            known_restrict_neighbors=True,
        )
        results["diagnostics"]["same_branch"][f"klnd{method}"] = build_metrics(
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
            spec=specs["reference_08"],
            balanced_seed=balanced_seed,
        )

    # Chan doan 2: self-neighbor, giu co dinh moi thu tru tap lang gieng.
    for method in (1, 2, 3):
        entries: dict[str, Any] = {}
        for name, neighbor_set in (
            ("with_self", neighbors["reference"]),
            ("without_self", neighbors["corrected"]),
        ):
            valid_scores, known_scores, open_scores = klnd_scores(
                "corrected",
                method,
                valid_logits,
                known_logits,
                open_logits,
                means["corrected"],
                neighbor_set,
                known_restrict_neighbors=True,
            )
            entries[name] = build_metrics(
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
                spec=specs["corrected_095"],
                balanced_seed=balanced_seed,
            )
        entries["delta_without_minus_with"] = delta(
            entries["with_self"], entries["without_self"]
        )
        results["diagnostics"]["self_neighbor"][f"klnd{method}"] = entries

    # Chan doan 3: calibration correct-only vs toan bo closed_valid (reference).
    for method in (1, 2, 3):
        valid_scores, known_scores, open_scores = klnd_scores(
            "reference",
            method,
            valid_logits,
            known_logits,
            open_logits,
            means["reference"],
            neighbors["reference"],
        )
        entries = {}
        for name, spec in (
            ("correct_only", specs["reference_08"]),
            (
                "all_valid",
                {
                    **specs["reference_08"],
                    "correct_only": False,
                },
            ),
        ):
            entries[name] = build_metrics(
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
                spec=spec,
                balanced_seed=balanced_seed,
            )
        entries["delta_all_minus_correct"] = delta(
            entries["correct_only"], entries["all_valid"]
        )
        results["diagnostics"]["calibration_correct_only"][f"klnd{method}"] = entries

    # Chan doan 4: MSP co bao hoa o 0 tren closed_valid khong.
    valid_probs = softmax(valid_logits, temperature)
    top_prob = valid_probs.max(axis=1)
    msp_scores = 1.0 - top_prob
    sorted_logits = np.sort(valid_logits, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    valid_pred = valid_logits.argmax(axis=1)
    reference_thresholds = calibrate_thresholds(
        msp_scores,
        valid_pred,
        num_labels,
        mode="reference",
        keep_rate=float(open_cfg["reference_keep_rate"]),
        magic_empty=bool(open_cfg.get("reference_magic_empty", True)),
    )
    corrected_thresholds = calibrate_thresholds(
        msp_scores,
        valid_pred,
        num_labels,
        mode="corrected",
        keep_rate=float(open_cfg["corrected_keep_rate"]),
    )
    results["diagnostics"]["msp_saturation"] = {
        "valid_top_prob_median": float(np.median(top_prob)),
        "valid_top_prob_mean": float(np.mean(top_prob)),
        "valid_top_prob_min": float(np.min(top_prob)),
        "valid_top_prob_max": float(np.max(top_prob)),
        "valid_msp_median": float(np.median(msp_scores)),
        "valid_msp_fraction_zero": float(np.mean(msp_scores == 0.0)),
        "valid_msp_fraction_below_1e_6": float(np.mean(msp_scores < 1e-6)),
        "valid_max_logit_median": float(np.median(sorted_logits[:, -1])),
        "valid_margin_median": float(np.median(margin)),
        "reference_threshold_median": float(np.median(reference_thresholds)),
        "reference_threshold_max": float(np.max(reference_thresholds)),
        "corrected_threshold_median": float(np.median(corrected_thresholds)),
        "corrected_threshold_max": float(np.max(corrected_thresholds)),
    }

    # Summary: tach tac dong sua loi / operating point.
    for detector, variants in results["detectors"].items():
        entry: dict[str, Any] = {}
        for name in ("reference_08", "reference_095", "corrected_08", "corrected_095"):
            if name in variants:
                entry[name] = subset(variants[name])
        if "reference_08" in variants and "corrected_08" in variants:
            entry["delta_bug_fix"] = delta(variants["reference_08"], variants["corrected_08"])
        if "reference_08" in variants and "corrected_095" in variants:
            entry["delta_combined"] = delta(
                variants["reference_08"], variants["corrected_095"]
            )
        if "reference_08" in variants and "reference_095" in variants:
            entry["delta_operating_reference"] = delta(
                variants["reference_08"], variants["reference_095"]
            )
        if "corrected_08" in variants and "corrected_095" in variants:
            entry["delta_operating_corrected"] = delta(
                variants["corrected_08"], variants["corrected_095"]
            )
        results["summary"][detector] = entry

    audit_cfg = config.get("audit", {})
    output_json = Path(
        audit_cfg.get(
            "output_json",
            Path(config["metrics"]["output_json"]).with_name(
                f"audit_{config['experiment']}.json"
            ),
        )
    )
    output_csv = Path(
        audit_cfg.get(
            "output_csv",
            Path(config["metrics"]["output_csv"]).with_name(
                f"audit_{config['experiment']}.csv"
            ),
        )
    )
    write_json(output_json, results)

    rows: list[dict[str, Any]] = []
    for detector, variants in results["detectors"].items():
        for variant, metrics in variants.items():
            for key in SCALAR_METRICS:
                rows.append(
                    {
                        "detector": detector,
                        "variant": variant,
                        "metric": key,
                        "value": metrics[key],
                    }
                )
            for class_id, value in metrics["open_class_detection"].items():
                rows.append(
                    {
                        "detector": detector,
                        "variant": variant,
                        "metric": f"open_class_{class_id}",
                        "value": value,
                    }
                )
    for method, metrics in results["diagnostics"]["same_branch"].items():
        for key in SCALAR_METRICS:
            rows.append(
                {
                    "detector": method,
                    "variant": "diag_same_branch",
                    "metric": key,
                    "value": metrics[key],
                }
            )
    for method, entries in results["diagnostics"]["self_neighbor"].items():
        for name in ("with_self", "without_self"):
            for key in SCALAR_METRICS:
                rows.append(
                    {
                        "detector": method,
                        "variant": f"diag_self_neighbor_{name}",
                        "metric": key,
                        "value": entries[name][key],
                    }
                )
    for method, entries in results["diagnostics"]["calibration_correct_only"].items():
        for name in ("correct_only", "all_valid"):
            for key in SCALAR_METRICS:
                rows.append(
                    {
                        "detector": method,
                        "variant": f"diag_calibration_{name}",
                        "metric": key,
                        "value": entries[name][key],
                    }
                )
    for key, value in results["diagnostics"]["msp_saturation"].items():
        rows.append(
            {
                "detector": "msp",
                "variant": "diag_msp_saturation",
                "metric": key,
                "value": value,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print("\nAudit (CCR / UCR / AUROC / FPR95 / OSCR):", flush=True)
    print(
        f"{'detector':10s} {'variant':16s} {'CCR':>8s} {'UCR':>8s} "
        f"{'AUROC':>8s} {'FPR95':>8s} {'OSCR':>8s}",
        flush=True,
    )
    for detector, variants in results["detectors"].items():
        for variant, metrics in variants.items():
            print(
                f"{detector:10s} {variant:16s} {metrics['CCR']:8.4f} "
                f"{metrics['unknown_recall']:8.4f} {metrics['AUROC']:8.4f} "
                f"{metrics['FPR@95TPR']:8.4f} {metrics['OSCR']:8.4f}",
                flush=True,
            )

    print("\nChan doan:", flush=True)
    for method, metrics in results["diagnostics"]["same_branch"].items():
        print(
            f"  {method} reference same-branch: AUROC={metrics['AUROC']:.4f} "
            f"FPR95={metrics['FPR@95TPR']:.4f} CCR={metrics['CCR']:.4f} "
            f"UCR={metrics['unknown_recall']:.4f}",
            flush=True,
        )
    for method, entries in results["diagnostics"]["self_neighbor"].items():
        values = entries["delta_without_minus_with"]
        print(
            f"  {method} self-neighbor delta (without - with): "
            f"AUROC={values['AUROC']:+.4f} FPR95={values['FPR@95TPR']:+.4f} "
            f"CCR={values['CCR']:+.4f} UCR={values['unknown_recall']:+.4f}",
            flush=True,
        )
    for method, entries in results["diagnostics"]["calibration_correct_only"].items():
        values = entries["delta_all_minus_correct"]
        print(
            f"  {method} calibration delta (all - correct-only): "
            f"AUROC={values['AUROC']:+.4f} CCR={values['CCR']:+.4f} "
            f"UCR={values['unknown_recall']:+.4f}",
            flush=True,
        )
    saturation = results["diagnostics"]["msp_saturation"]
    print(
        "  MSP valid: median top-prob={:.4f}, fraction score 0={:.4f}, "
        "fraction < 1e-6={:.4f}, threshold median ref/corr={:.4f}/{:.4f}".format(
            saturation["valid_top_prob_median"],
            saturation["valid_msp_fraction_zero"],
            saturation["valid_msp_fraction_below_1e_6"],
            saturation["reference_threshold_median"],
            saturation["corrected_threshold_median"],
        ),
        flush=True,
    )
    print(f"\nDa luu JSON -> {output_json}", flush=True)
    print(f"Da luu CSV  -> {output_csv}", flush=True)


if __name__ == "__main__":
    main()
