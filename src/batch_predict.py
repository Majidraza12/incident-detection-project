from pathlib import Path
import csv
import torch
import torch.nn.functional as F

from dataset import ViolenceVideoDataset
from model import CNNLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "non-violent": 0,
    "violent": 1,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def predict_single_video(model, video_path, seq_len=16):
    temp_dataset = ViolenceVideoDataset(root_dir="data/splits/test", seq_len=seq_len)
    frames = temp_dataset.sample_frames(Path(video_path)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(frames)
        probs = F.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return IDX_TO_LABEL[pred_idx], confidence


def infer_actual_label(video_path: Path):
    parts = [p.lower() for p in video_path.parts]
    if "violent" in parts:
        return "violent"
    if "non-violent" in parts:
        return "non-violent"
    return "unknown"


def main():
    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(
        torch.load("outputs/checkpoints/cnn_lstm_baseline.pth", map_location=DEVICE)
    )
    model.eval()

    video_list = [
        # ----- violent examples -----
        "data/splits/test/violent/cam1_5.mp4",
        "data/splits/test/violent/cam1_9.mp4",
        "data/splits/test/violent/cam1_36.mp4",
        "data/splits/test/violent/cam1_112.mp4",
        "data/splits/test/violent/cam1_115.mp4",
        "data/splits/test/violent/cam2_11.mp4",
        "data/splits/test/violent/cam2_21.mp4",
        "data/splits/test/violent/cam2_23.mp4",
        "data/splits/test/violent/cam2_42.mp4",
        "data/splits/test/violent/cam2_59.mp4",

        # ----- non-violent examples -----
        "data/splits/test/non-violent/cam1_1.mp4",
        "data/splits/test/non-violent/cam1_2.mp4",
        "data/splits/test/non-violent/cam1_5.mp4",
        "data/splits/test/non-violent/cam1_31.mp4",
        "data/splits/test/non-violent/cam1_46.mp4",
        "data/splits/test/non-violent/cam2_5.mp4",
        "data/splits/test/non-violent/cam2_26.mp4",
        "data/splits/test/non-violent/cam2_52.mp4",
        "data/splits/test/non-violent/cam2_54.mp4",
        "data/splits/test/non-violent/cam2_55.mp4",
    ]

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "batch_predictions.csv"

    rows = []

    for video in video_list:
        video_path = Path(video)

        if not video_path.exists():
            print(f"Skipping missing file: {video}")
            continue

        predicted_label, confidence = predict_single_video(model, video_path)
        actual_label = infer_actual_label(video_path)
        correct = predicted_label == actual_label

        row = {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "correct": correct,
        }
        rows.append(row)

        print(
            f"Video: {video_path.name} | "
            f"Actual: {actual_label} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: {confidence:.4f} | "
            f"Correct: {correct}"
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "video_path",
                "actual_label",
                "predicted_label",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved predictions to: {output_csv}")


if __name__ == "__main__":
    main()