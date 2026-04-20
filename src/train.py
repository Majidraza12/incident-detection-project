import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import ViolenceVideoDataset
from model import CNNLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-4

# -----------------------------
# Create output directories
# -----------------------------
os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
train_ds = ViolenceVideoDataset("data/splits/train", seq_len=16)
val_ds = ViolenceVideoDataset("data/splits/val", seq_len=16)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Compute class weights from training dataset
# Label mapping:
# 0 = non-violent
# 1 = violent
# -----------------------------
class_counts = [0, 0]

for video_path in train_ds.video_paths:
    label_name = video_path.parent.name
    if label_name == "non-violent":
        class_counts[0] += 1
    elif label_name == "violent":
        class_counts[1] += 1

total_samples = sum(class_counts)

# Inverse-frequency class weighting
class_weights = [
    total_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 0.0,
    total_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 0.0,
]

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print(f"Training class counts: {class_counts}")
print(f"Using class weights: {class_weights.tolist()}")

# -----------------------------
# Initialize model, loss, optimizer
# -----------------------------
model = CNNLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Track metrics across epochs
# -----------------------------
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # Save metrics in lists
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# -----------------------------
# Save weighted-loss model
# -----------------------------
model_path = "outputs/checkpoints/cnn_lstm_weighted.pth"
torch.save(model.state_dict(), model_path)
print(f"Weighted-loss model saved to {model_path}")

# -----------------------------
# Save metrics to CSV
# -----------------------------
csv_path = "outputs/metrics/training_metrics_weighted.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
    for i in range(EPOCHS):
        writer.writerow([
            i + 1,
            train_losses[i],
            val_losses[i],
            train_accs[i],
            val_accs[i]
        ])

print(f"Training metrics saved to {csv_path}")

# -----------------------------
# Plot weighted loss curve
# -----------------------------
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Training Loss")
plt.plot(range(1, EPOCHS + 1), val_losses, marker="o", label="Validation Loss")
plt.title("Weighted Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
loss_plot_path = "outputs/plots/loss_curve_weighted.png"
plt.savefig(loss_plot_path, bbox_inches="tight")
plt.close()
print(f"Loss curve saved to {loss_plot_path}")

# -----------------------------
# Plot weighted accuracy curve
# -----------------------------
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_accs, marker="o", label="Training Accuracy")
plt.plot(range(1, EPOCHS + 1), val_accs, marker="o", label="Validation Accuracy")
plt.title("Weighted Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
acc_plot_path = "outputs/plots/accuracy_curve_weighted.png"
plt.savefig(acc_plot_path, bbox_inches="tight")
plt.close()
print(f"Accuracy curve saved to {acc_plot_path}")