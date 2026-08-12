"""
Test-set evaluation.

Exposes evaluate_checkpoint(...) so both this script and the ablation table
runner score models the exact same way on the same held-out test set. Running
this file directly reproduces the original single-model report + confusion
matrix for the weighted CNN-LSTM checkpoint.
"""

import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from dataset import ViolenceVideoDataset
from model import CNNLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_NAMES = ["non-violent", "violent"]
VIOLENT_IDX = 1


def build_test_loader(seq_len=16, batch_size=2):
    test_ds = ViolenceVideoDataset("data/splits/test", seq_len=seq_len)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def evaluate_checkpoint(model, ckpt_path, test_loader, device=DEVICE):
    """Load weights into `model`, run inference, return metrics.

    Returns a dict: accuracy, violent_f1, macro_f1, predictions, labels,
    confusion_matrix, and the sklearn text report.
    """
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    # Pass explicit labels so metrics stay well-defined even if a run predicts
    # only one class (both classes are present in the real 53-clip test set).
    labels = [0, 1]
    acc = accuracy_score(all_labels, all_preds)
    violent_f1 = f1_score(
        all_labels, all_preds, pos_label=VIOLENT_IDX, average="binary", zero_division=0
    )
    macro_f1 = f1_score(all_labels, all_preds, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    report = classification_report(
        all_labels, all_preds, labels=labels, target_names=TARGET_NAMES, zero_division=0
    )

    return {
        "accuracy": acc,
        "violent_f1": violent_f1,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
        "confusion_matrix": cm,
        "report": report,
    }


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    model_path = "outputs/checkpoints/cnn_lstm_weighted.pth"
    test_loader = build_test_loader()

    results = evaluate_checkpoint(CNNLSTM(), model_path, test_loader)

    print("Classification Report:\n")
    print(results["report"])
    with open("outputs/metrics/classification_report.txt", "w") as f:
        f.write(results["report"])

    cm = results["confusion_matrix"]
    print("Confusion Matrix:\n")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Violent", "Violent"],
        yticklabels=["Non-Violent", "Violent"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
