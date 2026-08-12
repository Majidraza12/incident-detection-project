import sys
from functools import lru_cache
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import CNNLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/checkpoints/cnn_lstm_weighted.pth"
SEQ_LEN = 16

CLASS_NAMES = {
    0: "Non-Violent",
    1: "Violent"
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def sample_frames(video_path, seq_len=16):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError(f"Could not read frames from video: {video_path}")

    indices = torch.linspace(0, total_frames - 1, steps=seq_len).long().tolist()
    frames = []

    current_idx = 0
    target_set = set(indices)
    grabbed_frames = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            grabbed_frames[current_idx] = transform(frame)

        current_idx += 1

    cap.release()

    for idx in indices:
        if idx in grabbed_frames:
            frames.append(grabbed_frames[idx])
        else:
            frames.append(torch.zeros(3, 224, 224))

    return torch.stack(frames)  # [T, C, H, W]

@lru_cache(maxsize=1)
def load_model():
    # Cached so repeated predict_video() calls in one process load weights once.
    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def predict_video(video_path):
    model = load_model()
    frames = sample_frames(video_path, seq_len=SEQ_LEN)
    frames = frames.unsqueeze(0).to(DEVICE)  # [1, T, C, H, W] - T:: number of frames i.e 16

    with torch.no_grad():  #Disables gradient tracking
        outputs = model(frames)
        probs = F.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return CLASS_NAMES[pred_idx], confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/video.mp4")
        return

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"Error: file not found -> {video_path}")
        return

    label, confidence = predict_video(video_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()