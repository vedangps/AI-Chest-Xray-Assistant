from pathlib import Path

# ==========================
# Project Directories
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"

# ==========================
# Model Configuration
# ==========================

MODEL_PATH = MODEL_DIR / "custom_cnn" / "best_model.pth"
DENSENET_MODEL_PATH = MODEL_DIR / "densenet121" / "best_model.pth"
DENSENET_TUNED_METRICS_PATH = (
    MODEL_DIR / "densenet121" / "tuned_metrics.json"
)

DEFAULT_DECISION_THRESHOLD = 0.50
THRESHOLD_SWEEP_MIN = 0.40
THRESHOLD_SWEEP_MAX = 0.95
THRESHOLD_SWEEP_STEPS = 111

# ==========================
# Grad-CAM Configuration
# ==========================

# Standard deviation (in output pixels) for Gaussian smoothing of the
# heatmap. Removes the "hazy"/speckled activations produced by upsampling
# a coarse 7x7 feature map to full resolution.
GRADCAM_SMOOTHING_SIGMA = 8.0

# Activations below this normalized intensity are treated as background
# noise or anatomical edges (ribs/heart borders) and suppressed so the
# overlay highlights only high-confidence lung pathology.
GRADCAM_SUPPRESSION_THRESHOLD = 0.35

# Peak opacity of the heatmap where activation is strongest. Blending is
# modulated per-pixel by activation strength so cold regions keep the
# original radiograph instead of a uniform color wash.
GRADCAM_OVERLAY_ALPHA = 0.5

# ==========================
# Data Configuration
# ==========================

IMAGE_SIZE = (224, 224)

RANDOM_SEED = 42

VALIDATION_SPLIT = 0.2

CLASS_NAMES = (
    "NORMAL",
    "PNEUMONIA",
)

# ==========================
# Training Configuration
# ==========================

BATCH_SIZE = 16

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

NUM_EPOCHS = 30

NUM_WORKERS = 0

LR_SCHEDULER_FACTOR = 0.5

LR_SCHEDULER_PATIENCE = 2

LR_SCHEDULER_MIN_LR = 0.000001

EARLY_STOPPING_PATIENCE = 5

EARLY_STOPPING_MIN_DELTA = 0.0

# ==========================
# Test Configuration
# ==========================

if __name__ == "__main__":

    print("=" * 50)
    print("AI Chest X-ray Assistant")
    print("=" * 50)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Data         : {DATA_DIR}")
    print(f"Models       : {MODEL_DIR}")
    print(f"Model Path   : {MODEL_PATH}")
    print(f"DenseNet Path: {DENSENET_MODEL_PATH}")
    print(f"Assets       : {ASSETS_DIR}")
    print(f"Docs         : {DOCS_DIR}")
    print(f"Image Size   : {IMAGE_SIZE}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")
