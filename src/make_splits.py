from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

PROCESSED_DIR = Path("data/processed/violence-detection-dataset")
SPLITS_DIR = Path("data/splits")

def gather_files():
    violent = list((PROCESSED_DIR / "violent").rglob("*.mp4"))
    nonviolent = list((PROCESSED_DIR / "non-violent").rglob("*.mp4"))
    return violent, nonviolent

def split_class(files):
    train, temp = train_test_split(files, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)
    return {"train": train, "val": val, "test": test}

def copy_files(split_dict, class_name):
    for split_name, files in split_dict.items():
        out_dir = SPLITS_DIR / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            camera_name = f.parent.name
            new_name = f"{camera_name}_{f.name}"
            shutil.copy2(f, out_dir / new_name)

def main():
    violent, nonviolent = gather_files()

    print(f"Found violent videos: {len(violent)}")
    print(f"Found non-violent videos: {len(nonviolent)}")

    if len(violent) == 0 or len(nonviolent) == 0:
        print("Error: could not find videos. Check your processed dataset path.")
        return

    violent_splits = split_class(violent)
    nonviolent_splits = split_class(nonviolent)

    copy_files(violent_splits, "violent")
    copy_files(nonviolent_splits, "non-violent")

    print("Dataset split completed.")

if __name__ == "__main__":
    main()