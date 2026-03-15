from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {
    "non-violent": 0,
    "violent": 1,
}

class ViolenceVideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=16):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.video_paths = list(self.root_dir.rglob("*.mp4"))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.video_paths)

    def sample_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.seq_len

        indices = torch.linspace(0, max(total_frames - 1, 0), steps=self.seq_len).long().tolist()

        current_idx = 0
        target_set = set(indices)
        grabbed_frames = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in target_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                grabbed_frames[current_idx] = self.transform(frame)

            current_idx += 1

        cap.release()

        for idx in indices:
            if idx in grabbed_frames:
                frames.append(grabbed_frames[idx])
            else:
                frames.append(torch.zeros(3, 224, 224))

        return torch.stack(frames)  # [seq, C, H, W]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label_name = video_path.parent.name
        label = LABEL_MAP[label_name]
        frames = self.sample_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)
