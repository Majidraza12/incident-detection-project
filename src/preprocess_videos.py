import os
import cv2
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TARGET_SIZE = (224, 224)
TARGET_FPS = 15

def collect_videos():
    video_files = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        video_files.extend(RAW_DIR.rglob(ext))
    return sorted(video_files)

def process_video(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Skipping unreadable file: {input_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, TARGET_FPS, TARGET_SIZE)

    frame_idx = 0
    sample_interval = max(int(round(original_fps / TARGET_FPS)), 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            frame = cv2.resize(frame, TARGET_SIZE)
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

def main():
    videos = collect_videos()
    print(f"Processing {len(videos)} videos...")

    #Debugging paths
    print("Current working directory:", Path.cwd())
    print("RAW_DIR exists:", RAW_DIR.exists())
    print("RAW_DIR path:", RAW_DIR.resolve())

    for video in videos:
        rel_path = video.relative_to(RAW_DIR)
        out_path = PROCESSED_DIR / rel_path.with_suffix(".mp4")
        process_video(video, out_path)
    

    print("Done.")

if __name__ == "__main__":
    main()
