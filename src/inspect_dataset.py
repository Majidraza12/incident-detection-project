import os
import cv2
from pathlib import Path

RAW_DIR = Path("data/raw")

# LABEL_MAP = {
#     "non-violent": 0,
#     "violent": 1,
# }

def inspect_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "path": str(video_path),
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }

def collect_videos():
    video_files = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        video_files.extend(RAW_DIR.rglob(ext))
    return sorted(video_files)

def main():
    videos = collect_videos()
    print(f"Found {len(videos)} video files.\n")

    bad_files = []
    for video in videos[:20]:
        info = inspect_video(video)
        if info is None:
            bad_files.append(video)
        else:
            print(info)

    if bad_files:
        print("\nUnreadable files:")
        for bf in bad_files:
            print(bf)

if __name__ == "__main__":
    main()
