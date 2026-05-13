# -*- coding: utf-8 -*-
"""
Configuration file for the SegFormer-SFG segmentation experiment.

This public configuration provides a general example of how to organize
the training, validation, testing, and inference settings. Dataset paths
and some experiment-specific parameters should be modified according to
the user's local environment.
"""

import os


# ============================================================
# 1. Basic project settings
# ============================================================

PROJECT_NAME = "SegFormer-SFG"
TASK_NAME = "Grape stem and fruit segmentation"

# Root directory of this repository
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Output directory
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "segformer_sfg")

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Visualization result directory
VIS_DIR = os.path.join(OUTPUT_DIR, "visualization")


# ============================================================
# 2. Dataset settings
# ============================================================

# Public example data path.
# Please replace this path with your local dataset path when running.
DATA_ROOT = os.path.join(ROOT_DIR, "data", "mask_aug")

TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "masks")

VAL_IMAGE_DIR = os.path.join(DATA_ROOT, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_ROOT, "val", "masks")

TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "test", "masks")

# Example inference image folder
INFER_IMAGE_DIR = os.path.join(ROOT_DIR, "examples", "images")
INFER_SAVE_DIR = os.path.join(ROOT_DIR, "examples", "outputs", "segmentation_vis")


# ============================================================
# 3. Class settings
# ============================================================

NUM_CLASSES = 3

ID2LABEL = {
    0: "background",
    1: "stem",
    2: "fruit",
}

LABEL2ID = {
    "background": 0,
    "stem": 1,
    "fruit": 2,
}

# Classes used for foreground evaluation
EVAL_CLASS_IDS = [1, 2]


# ============================================================
# 4. Model settings
# ============================================================

# SegFormer backbone.
# Users can replace this with other SegFormer variants if needed.
BACKBONE = "nvidia/segformer-b2-finetuned-ade-512-512"

# Input image size
IMG_SIZE = 512

# Whether to use FSA-RG style foreground structure attention gate
USE_FSA_GATE = True

# Whether to use FSGM module
USE_FSGM = True

# Whether to use depthwise convolution inside FSAGate
USE_DWCONV = True


# ============================================================
# 5. Training settings
# ============================================================

EPOCHS = 80
# Enable automatic mixed precision training
USE_AMP = False
# Force CPU training if GPU is not available
USE_CPU = False


# ============================================================
# 6. Loss settings
# ============================================================

# ============================================================
# 7. Evaluation settings
# ============================================================

# Main evaluation metrics:
# mIoU, mDice, stem_IoU, stem_Dice, stem_Precision, stem_Recall,
# fruit_IoU, fruit_Dice, fruit_Precision, fruit_Recall

BEST_MODEL_NAME = "segformer_sfg_best.pt"

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)

SAVE_VIS = True


# ============================================================
# 8. Inference settings
# ============================================================

# Output types:
# 1. predicted label mask
# 2. colorized segmentation mask
# 3. overlay visualization image

SAVE_PRED_MASK = True
SAVE_COLOR_MASK = True
SAVE_OVERLAY = True


# ============================================================
# 9. Color map for visualization
# ============================================================

# RGB color definition for visualization
COLOR_MAP = {
    0: (0, 0, 0),        # background: black
    1: (255, 0, 0),      # stem: red
    2: (0, 255, 0),      # fruit: green
}


# ============================================================
# 10. Utility function
# ============================================================

def print_config():
    """Print key configuration information."""
    print("=" * 60)
    print(f"Project: {PROJECT_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Backbone: {BACKBONE}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Seed: {SEED}")
    print(f"Use AMP: {USE_AMP}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
