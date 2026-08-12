from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {
    "non-violent": 0,
    "violent": 1,
}

# ImageNet statistics — the pretrained ResNet-18 backbone was trained on inputs
# normalized this way. Applied to train and eval identically (no serve skew).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ViolenceVideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=16, augment=False, normalize=False):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.video_paths = list(self.root_dir.rglob("*.mp4"))

        # `augment` turns on train-only spatial/color jitter to fight overfitting.
        # It must never be enabled for val/test (that would make metrics
        # non-deterministic and optimistic). `normalize` applies ImageNet stats.
        self.transform = self._build_transform(augment=augment, normalize=normalize)

    def _build_transform(self, augment, normalize):
        steps = [transforms.ToPILImage()]
        if augment:
            steps += [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        else:
            steps += [transforms.Resize((224, 224))]
        steps += [transforms.ToTensor()]
        if normalize:
            steps += [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(steps)

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
