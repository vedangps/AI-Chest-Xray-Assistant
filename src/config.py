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

MODEL_PATH = MODEL_DIR / "best_model.pth"

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

BATCH_SIZE = 32

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
    print(f"Assets       : {ASSETS_DIR}")
    print(f"Docs         : {DOCS_DIR}")
    print(f"Image Size   : {IMAGE_SIZE}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")
