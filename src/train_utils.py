"""
Shared training utilities for reproducible, fair experiments.

Everything that must be held constant across the CNN-LSTM and frame-only
ablation arms lives here, so the only thing that changes between runs is the
model itself. Two pieces:

- set_seed / make_generator: make a run reproducible (weight init, DataLoader
  shuffling, dropout) so an A/B comparison is meaningful.
- train_run: a single training loop that every experiment calls. It selects the
  best checkpoint by lowest validation loss (NOT the final epoch, which is the
  most overfit), and returns the full per-epoch history.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    """Seed every RNG that affects a run and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed):
    """A seeded torch.Generator for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def compute_class_weights(video_paths, device):
    """Inverse-frequency weights over [non-violent, violent], from folder names.

    Matches the original train.py logic so the weighting arm is unchanged.
    """
    class_counts = [0, 0]
    for video_path in video_paths:
        label_name = video_path.parent.name
        if label_name == "non-violent":
            class_counts[0] += 1
        elif label_name == "violent":
            class_counts[1] += 1

    total = sum(class_counts)
    weights = [
        total / (2 * class_counts[0]) if class_counts[0] > 0 else 0.0,
        total / (2 * class_counts[1]) if class_counts[1] > 0 else 0.0,
    ]
    return torch.tensor(weights, dtype=torch.float32).to(device), class_counts


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_run(
    model,
    train_loader,
    val_loader,
    class_weights,
    run_name,
    seed,
    device,
    epochs=10,
    lr=1e-4,
    checkpoint_dir="outputs/checkpoints",
    trainable_params=None,
):
    """Train one model, saving the best-by-val-loss checkpoint.

    Returns a dict with per-epoch history plus the epoch/metrics of the best
    (lowest val loss) checkpoint. `trainable_params` lets a frozen-backbone run
    optimize a subset; defaults to all model parameters.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    params = trainable_params if trainable_params is not None else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_seed{seed}.pth")
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            marker = "  <- best (saved)"

        print(
            f"[{run_name} seed{seed}] "
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{marker}"
        )

    print(
        f"[{run_name} seed{seed}] Best checkpoint: epoch {best_epoch} "
        f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}) -> {ckpt_path}"
    )

    return {
        "run_name": run_name,
        "seed": seed,
        "checkpoint_path": ckpt_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history,
    }
