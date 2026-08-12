"""
Temporal ablation runner: CNN-LSTM vs frame-only mean-pool.

Runs each architecture across multiple seeds through the SAME training harness
(train_utils.train_run), reusing the SAME data/splits, so the only variable is
the temporal aggregation. Every run selects its best checkpoint by validation
loss and writes a per-run history CSV. make_ablation_table.py then evaluates all
resulting checkpoints on the held-out test set.

Usage:
    python src/train_ablation.py
"""

import os
import csv

import torch
from torch.utils.data import DataLoader

from dataset import ViolenceVideoDataset
from model import CNNLSTM, CNNMeanPool
from train_utils import set_seed, make_generator, compute_class_weights, train_run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5, not 10: in both prior 10-epoch runs the best validation loss landed at
# epoch 1 and never improved after (see outputs/metrics/training_metrics*.csv),
# so later epochs produced no checkpoint that was ever kept. Seeds are kept at 3
# because the LSTM-vs-mean-pool gap is small relative to seed noise on a
# 53-clip test set.
EPOCHS = 5
BATCH_SIZE = 2
LR = 1e-4
SEQ_LEN = 16
SEEDS = [0, 1, 2]

# Model factories. Keyed by run_name (note: NOT "baseline" — that name already
# refers to the unweighted checkpoint in this repo).
MODEL_FACTORIES = {
    "cnn_lstm": lambda: CNNLSTM(),
    "frame_mean": lambda: CNNMeanPool(),
}

METRICS_DIR = "outputs/metrics/ablation"
CHECKPOINT_DIR = "outputs/checkpoints/ablation"


def build_loaders(seed):
    """Fresh loaders with a seeded shuffle generator. Splits are fixed on disk."""
    train_ds = ViolenceVideoDataset("data/splits/train", seq_len=SEQ_LEN)
    val_ds = ViolenceVideoDataset("data/splits/val", seq_len=SEQ_LEN)

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_ds, train_loader, val_loader


def save_history_csv(result):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(
        METRICS_DIR, f"history_{result['run_name']}_seed{result['seed']}.csv"
    )
    h = result["history"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(h["epoch"])):
            writer.writerow([
                h["epoch"][i],
                h["train_loss"][i],
                h["val_loss"][i],
                h["train_acc"][i],
                h["val_acc"][i],
            ])
    return path


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Models: {list(MODEL_FACTORIES)} | Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}\n")

    summary_rows = []

    for run_name, factory in MODEL_FACTORIES.items():
        for seed in SEEDS:
            print(f"\n===== {run_name} | seed {seed} =====")

            # Seed BEFORE building loaders and model so shuffle + weight init
            # are reproducible.
            set_seed(seed)
            train_ds, train_loader, val_loader = build_loaders(seed)
            class_weights, class_counts = compute_class_weights(
                train_ds.video_paths, DEVICE
            )
            print(f"Train class counts [non-violent, violent]: {class_counts}")

            model = factory().to(DEVICE)
            result = train_run(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                run_name=run_name,
                seed=seed,
                device=DEVICE,
                epochs=EPOCHS,
                lr=LR,
                checkpoint_dir=CHECKPOINT_DIR,
            )

            hist_path = save_history_csv(result)
            n_params = sum(p.numel() for p in model.parameters())

            summary_rows.append({
                "run_name": run_name,
                "seed": seed,
                "checkpoint_path": result["checkpoint_path"],
                "history_csv": hist_path,
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "best_val_acc": result["best_val_acc"],
                "num_params": n_params,
            })

    # A single index of every run, consumed by make_ablation_table.py.
    summary_path = os.path.join(METRICS_DIR, "runs_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nAll runs complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
