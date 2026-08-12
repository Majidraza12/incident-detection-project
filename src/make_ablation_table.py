"""
Build the temporal-ablation table from the runs produced by train_ablation.py.

Reads outputs/metrics/ablation/runs_summary.csv, evaluates every best-by-val
checkpoint on the SAME held-out test set, then aggregates mean +/- std across
seeds per architecture. Writes:
  - outputs/metrics/ablation/ablation.csv      (machine-readable, per-metric)
  - outputs/metrics/ablation/ablation_table.md (drop straight into the README)

Usage:
    python src/make_ablation_table.py
"""

import os
import csv
from collections import defaultdict

import numpy as np

from model import CNNLSTM, CNNMeanPool
from evaluate import build_test_loader, evaluate_checkpoint

METRICS_DIR = "outputs/metrics/ablation"
SUMMARY_CSV = os.path.join(METRICS_DIR, "runs_summary.csv")

# run_name -> (display label, model factory)
MODELS = {
    "cnn_lstm": ("CNN-LSTM (ResNet-18 + LSTM)", lambda: CNNLSTM()),
    "frame_mean": ("Frame-only (ResNet-18 + mean-pool)", lambda: CNNMeanPool()),
}


def load_summary():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found. Run: python src/train_ablation.py first."
        )
    with open(SUMMARY_CSV, newline="") as f:
        return list(csv.DictReader(f))


def fmt(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"


def main():
    rows = load_summary()
    test_loader = build_test_loader()

    # Collect per-architecture metric lists across seeds.
    agg = defaultdict(lambda: {
        "test_acc": [], "violent_f1": [], "best_val_acc": [], "num_params": None,
    })

    per_run_out = []

    for row in rows:
        run_name = row["run_name"]
        if run_name not in MODELS:
            print(f"Skipping unknown run_name: {run_name}")
            continue

        _, factory = MODELS[run_name]
        ckpt = row["checkpoint_path"]
        print(f"Evaluating {run_name} seed {row['seed']} -> {ckpt}")

        results = evaluate_checkpoint(factory(), ckpt, test_loader)

        agg[run_name]["test_acc"].append(results["accuracy"])
        agg[run_name]["violent_f1"].append(results["violent_f1"])
        agg[run_name]["best_val_acc"].append(float(row["best_val_acc"]))
        agg[run_name]["num_params"] = int(row["num_params"])

        per_run_out.append({
            "run_name": run_name,
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "best_val_acc": float(row["best_val_acc"]),
            "test_acc": results["accuracy"],
            "violent_f1": results["violent_f1"],
        })

    # Write per-metric aggregate CSV.
    os.makedirs(METRICS_DIR, exist_ok=True)
    agg_csv = os.path.join(METRICS_DIR, "ablation.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "num_params", "n_seeds",
            "val_acc_mean", "val_acc_std",
            "test_acc_mean", "test_acc_std",
            "violent_f1_mean", "violent_f1_std",
        ])
        for run_name, (label, _) in MODELS.items():
            a = agg[run_name]
            if not a["test_acc"]:
                continue
            writer.writerow([
                label, a["num_params"], len(a["test_acc"]),
                np.mean(a["best_val_acc"]), np.std(a["best_val_acc"]),
                np.mean(a["test_acc"]), np.std(a["test_acc"]),
                np.mean(a["violent_f1"]), np.std(a["violent_f1"]),
            ])

    # Write a README-ready markdown table.
    md_path = os.path.join(METRICS_DIR, "ablation_table.md")
    with open(md_path, "w") as f:
        f.write("| Model | Params | Val acc | Test acc | Violent F1 |\n")
        f.write("|-------|-------:|:-------:|:--------:|:----------:|\n")
        for run_name, (label, _) in MODELS.items():
            a = agg[run_name]
            if not a["test_acc"]:
                continue
            f.write(
                f"| {label} | {a['num_params']:,} | "
                f"{fmt(np.mean(a['best_val_acc']), np.std(a['best_val_acc']))} | "
                f"{fmt(np.mean(a['test_acc']), np.std(a['test_acc']))} | "
                f"{fmt(np.mean(a['violent_f1']), np.std(a['violent_f1']))} |\n"
            )
        f.write(
            f"\n_Mean +/- std over {len(SEEDS_USED(rows))} seeds; "
            f"best-by-val checkpoints; test set = 53 clips._\n"
        )

    print("\n=== Ablation summary ===")
    for run_name, (label, _) in MODELS.items():
        a = agg[run_name]
        if not a["test_acc"]:
            continue
        print(
            f"{label}: test_acc {fmt(np.mean(a['test_acc']), np.std(a['test_acc']))}, "
            f"violent_f1 {fmt(np.mean(a['violent_f1']), np.std(a['violent_f1']))}"
        )
    print(f"\nWrote {agg_csv}\nWrote {md_path}")


def SEEDS_USED(rows):
    return sorted({r["seed"] for r in rows})


if __name__ == "__main__":
    main()
