import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/heart_dataset")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MSK_DIR = os.path.join(DATA_ROOT, "train", "masks")
TEST_IMG_DIR  = os.path.join(DATA_ROOT, "test", "images")
TEST_MSK_DIR  = os.path.join(DATA_ROOT, "test", "masks")

# -------------------------------
# TRAINING / PREPROCESSING CONFIG
# -------------------------------
TARGET_SPACING = (1.5, 1.5, 1.5)     
PATCH_SIZE = (96, 96, 96)
PATCHES_PER_VOLUME = 16
FG_SAMPLE_PROB = 0.9
BATCH_SIZE = 2
EPOCHS = 30

# -------------------
# OUTPUT / LOGGING
# -------------------
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
AUG_DIR = os.path.join(RESULTS_DIR, "augmented_images")

for d in [RESULTS_DIR, LOG_DIR, PRED_DIR, CKPT_DIR,PLOTS_DIR,AUG_DIR]:
    os.makedirs(d, exist_ok=True)
