import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import ViolenceVideoDataset
from model import CNNLSTM

# -----------------------------
# Setup
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "outputs/checkpoints/cnn_lstm_baseline.pth"
MODEL_PATH = "outputs/checkpoints/cnn_lstm_weighted.pth" #Evaluating the updated model

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# -----------------------------
# Load test dataset
# -----------------------------
test_ds = ViolenceVideoDataset("data/splits/test", seq_len=16)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

# -----------------------------
# Load trained model
# -----------------------------
model = CNNLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# Run inference
# -----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)

        outputs = model(x)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(y.numpy())

# -----------------------------
# Classification report
# -----------------------------
report = classification_report(
    all_labels,
    all_preds,
    target_names=["non-violent", "violent"]
)

print("Classification Report:\n")
print(report)

with open("outputs/metrics/classification_report.txt", "w") as f:
    f.write(report)

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:\n")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Violent", "Violent"],
    yticklabels=["Non-Violent", "Violent"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("outputs/plots/confusion_matrix.png")
plt.show()