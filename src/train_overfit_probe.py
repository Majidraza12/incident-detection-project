"""
Overfitting diagnostic: does augmentation + a frozen backbone shrink the
train-val gap?

The base CNN-LSTM overfits hard (train_acc -> 1.0 while val_acc plateaus). This
script runs two configurations through the same harness and the same seeds and
reports the train-val accuracy gap for each:

  - "original": no augmentation, backbone fine-tuned end-to-end (the status quo)
  - "aug_frozen": train-only augmentation + ImageNet normalize, ResNet backbone
    frozen (only LSTM + head train)

The point is not a higher score — it is evidence the overfitting was diagnosed
and addressed, not just noticed. Output: outputs/metrics/overfit_probe.csv +
a printed gap comparison.

Usage:
    python src/train_overfit_probe.py
"""

import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ViolenceVideoDataset
from model import CNNLSTM
from train_utils import set_seed, make_generator, compute_class_weights, train_run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5, not 10: matches train_ablation.py. The train-val gap this script measures
# is already plainly open by epoch 5 (0.955 train vs 0.808 val in the prior run).
EPOCHS = 5
BATCH_SIZE = 2
LR = 1e-4
SEQ_LEN = 16
SEEDS = [0, 1, 2]

METRICS_DIR = "outputs/metrics/overfit_probe"
CHECKPOINT_DIR = "outputs/checkpoints/overfit_probe"

CONFIGS = {
    "original": {"augment": False, "normalize": False, "freeze_backbone": False},
    "aug_frozen": {"augment": True, "normalize": True, "freeze_backbone": True},
}


def build_loaders(seed, augment, normalize):
    # Augmentation/normalize apply to TRAIN only; val stays deterministic
    # (but must share normalization so the backbone sees consistent inputs).
    train_ds = ViolenceVideoDataset(
        "data/splits/train", seq_len=SEQ_LEN, augment=augment, normalize=normalize
    )
    val_ds = ViolenceVideoDataset(
        "data/splits/val", seq_len=SEQ_LEN, augment=False, normalize=normalize
    )
    generator = make_generator(seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_ds, train_loader, val_loader


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}\n")
    rows = []

    for cfg_name, cfg in CONFIGS.items():
        for seed in SEEDS:
            print(f"\n===== {cfg_name} | seed {seed} =====")
            set_seed(seed)
            train_ds, train_loader, val_loader = build_loaders(
                seed, cfg["augment"], cfg["normalize"]
            )
            class_weights, _ = compute_class_weights(train_ds.video_paths, DEVICE)

            model = CNNLSTM().to(DEVICE)

            trainable_params = None
            if cfg["freeze_backbone"]:
                for p in model.cnn.parameters():
                    p.requires_grad = False
                trainable_params = [p for p in model.parameters() if p.requires_grad]

            result = train_run(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                run_name=f"{cfg_name}",
                seed=seed,
                device=DEVICE,
                epochs=EPOCHS,
                lr=LR,
                checkpoint_dir=CHECKPOINT_DIR,
                trainable_params=trainable_params,
            )

            # Gap at the best-by-val epoch (index = best_epoch - 1).
            h = result["history"]
            bi = result["best_epoch"] - 1
            train_acc = h["train_acc"][bi]
            val_acc = h["val_acc"][bi]
            rows.append({
                "config": cfg_name,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_val_gap": train_acc - val_acc,
            })

    csv_path = os.path.join(METRICS_DIR, "overfit_probe.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Train-val gap (mean +/- std over seeds, at best-by-val epoch) ===")
    for cfg_name in CONFIGS:
        gaps = [r["train_val_gap"] for r in rows if r["config"] == cfg_name]
        print(f"{cfg_name}: gap {np.mean(gaps):.3f} +/- {np.std(gaps):.3f}")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
