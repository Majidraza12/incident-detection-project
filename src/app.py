import tempfile
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
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

@st.cache_resource
def load_model():
    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def sample_frames(video_path, seq_len=16):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError("Could not read frames from uploaded video.")

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

    return torch.stack(frames)

def predict_video(video_path):
    model = load_model()
    frames = sample_frames(video_path, seq_len=SEQ_LEN)
    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(frames)
        probs = F.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return CLASS_NAMES[pred_idx], confidence

st.title("Video-Based Incident Detection")
st.write("Upload a surveillance-style video clip to classify it as violent or non-violent.")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    if st.button("Run Prediction"):
        with st.spinner("Analyzing video..."):
            label, confidence = predict_video(temp_video_path)

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.4f}")