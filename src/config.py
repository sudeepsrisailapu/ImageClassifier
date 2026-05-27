import os
import torch

DATA_DIR = "data/chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

EPOCHS = 20
BATCH_SIZE = 32
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
EARLY_STOP = 5

IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


