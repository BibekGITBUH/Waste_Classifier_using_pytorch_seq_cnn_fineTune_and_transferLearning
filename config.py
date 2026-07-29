"""
Central configuration for the Smart Waste Image Classifier project.
Edit paths/hyperparameters here rather than hunting through scripts.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This should point at the folder created by the download script, with
# structure: RAW_DATA_DIR/<class_name>/*.jpg
RAW_DATA_DIR = "datasets/final"

# After running prepare_data.py, split data lives here:
# SPLIT_DATA_DIR/train/<class>/*.jpg
# SPLIT_DATA_DIR/val/<class>/*.jpg
# SPLIT_DATA_DIR/test/<class>/*.jpg
SPLIT_DATA_DIR = "datasets/split"

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"  # metrics, plots, confusion matrices go here

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic"]  # alphabetical
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Data split ratios
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Image / dataloader settings
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224  # works for custom CNN and MobileNetV2/ResNet50 alike
BATCH_SIZE = 32
NUM_WORKERS = 4

# ImageNet normalization stats (needed later for pretrained models;
# harmless to use now too, keeps preprocessing consistent across experiments)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

RANDOM_SEED = 42
