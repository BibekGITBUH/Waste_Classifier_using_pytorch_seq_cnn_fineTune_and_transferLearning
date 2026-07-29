"""
Splits the raw dataset (datasets/final/<class>/*.jpg) into
train/val/test folders under datasets/split/, using the ratios and seed
defined in config.py. This is the script config.py's docstring refers to
as "the download script" output being split -- run this once after your
raw class-labeled images are in place.

Usage:
    python prepare_data.py
"""

import random
import shutil
from pathlib import Path

import config


def prepare_data():
    random.seed(config.SPLIT_SEED)

    raw_dir = Path(config.RAW_DATA_DIR)
    split_dir = Path(config.SPLIT_DATA_DIR)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found at {raw_dir}. "
            f"Expected structure: {raw_dir}/<class_name>/*.jpg"
        )

    valid_extensions = {".jpg", ".jpeg", ".png"}

    for split in ("train", "val", "test"):
        for cls in config.CLASS_NAMES:
            (split_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"Splitting data: train={config.TRAIN_RATIO}, "
          f"val={config.VAL_RATIO}, test={config.TEST_RATIO} "
          f"(seed={config.SPLIT_SEED})\n")

    total_counts = {"train": 0, "val": 0, "test": 0}

    for cls in config.CLASS_NAMES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"⚠️  Warning: class folder '{cls}' not found in {raw_dir}, skipping.")
            continue

        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in valid_extensions]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)
        # remainder goes to test, so ratios always sum to the full set
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            for f in files:
                shutil.copy2(f, split_dir / split_name / cls / f.name)
            total_counts[split_name] += len(files)

        print(f"{cls:<10} total={n:<5} train={len(splits['train']):<5} "
              f"val={len(splits['val']):<5} test={len(splits['test']):<5}")

    print(f"\nDone. Totals -> train={total_counts['train']}, "
          f"val={total_counts['val']}, test={total_counts['test']}")
    print(f"Split data written to: {split_dir}/")


if __name__ == "__main__":
    prepare_data()
