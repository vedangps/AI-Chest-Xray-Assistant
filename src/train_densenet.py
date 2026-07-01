"""
Training script for the DenseNet121 transfer learning model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import (
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_MIN_LR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
)
from dataloader import create_dataloader
from dataset import get_training_class_weights
from densenet_model import ChestXrayDenseNet121
from train import train_one_epoch, validate_one_epoch

# Define unique isolated checkpoint tracking path for DenseNet121
DENSENET_MODEL_PATH = MODEL_DIR / "densenet121" / "best_model.pth"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader("train")
    validation_loader = create_dataloader("val")

    model = ChestXrayDenseNet121(pretrained=True).to(device)

    weights = get_training_class_weights()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    DENSENET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    epochs_no_improve = 0

    print("=" * 50)
    print("DenseNet121 Transfer Learning Training Active")
    print("=" * 50)
    print(f"Device              : {device}")
    print(f"Target Output Path  : {DENSENET_MODEL_PATH}")
    print(f"Training Batches    : {len(train_loader)}")
    print(f"Validation Batches  : {len(validation_loader)}")
    print(f"Optimizer           : {optimizer.__class__.__name__}")

    for epoch in range(NUM_EPOCHS):
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss = validate_one_epoch(model, validation_loader, criterion, device)

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print("\n" + "=" * 50)
        print(f"DenseNet121 Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("=" * 50)
        print(f"Learning Rate   : {current_lr:.6f}")
        print(f"Training Loss   : {train_loss:.4f}")
        print(f"Validation Loss : {validation_loss:.4f}")

        if validation_loss < (best_validation_loss - EARLY_STOPPING_MIN_DELTA):
            best_validation_loss = validation_loss
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), DENSENET_MODEL_PATH)
            print(f"✓ New DenseNet121 best model saved! (Val Loss: {validation_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹ Early stopping triggered after {epoch + 1} epochs.")
            break


if __name__ == "__main__":
    main()
    