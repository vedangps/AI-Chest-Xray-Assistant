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
# Data Configuration
# ==========================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = (
    "NORMAL",
    "PNEUMONIA",
)

# ==========================
# Training Configuration
# ==========================

BATCH_SIZE = 32

LEARNING_RATE = 0.001

NUM_EPOCHS = 10

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
    print(f"Assets       : {ASSETS_DIR}")
    print(f"Docs         : {DOCS_DIR}")
    print(f"Image Size   : {IMAGE_SIZE}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")