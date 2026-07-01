"""
Training script for the Chest X-ray CNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE,
    MODEL_PATH,
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
from model import ChestXrayCNN


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
) -> float:
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        # Move data to the selected device
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate batch loss
        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    return average_loss


def validate_one_epoch(
    model,
    validation_loader,
    criterion,
    device,
) -> float:
    """
    Evaluate the model on the validation dataset.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in validation_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute validation loss
            loss = criterion(outputs, labels)

            # Accumulate batch loss
            running_loss += loss.item()

    average_loss = running_loss / len(validation_loader)
    return average_loss


def main() -> None:
    """
    Initialize all training components.
    """
    # Select CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoaders
    train_loader = create_dataloader("train")
    validation_loader = create_dataloader("val")

    # Create model
    model = ChestXrayCNN().to(device)

    # Retrieve and format class weights to handle dataset imbalance
    weights = get_training_class_weights()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer with Weight Decay added for L2 Regularization
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    # Ensure the model directory exists
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # Track best metrics and early stopping
    best_validation_loss = float("inf")
    epochs_no_improve = 0

    print("=" * 50)
    print("Training Setup Complete")
    print("=" * 50)
    print(f"Device              : {device}")
    print(f"Training Batches    : {len(train_loader)}")
    print(f"Validation Batches  : {len(validation_loader)}")
    print(f"Class Weights       : {weights}")
    print(f"Loss Function       : {criterion}")
    print(f"Optimizer           : {optimizer.__class__.__name__}")
    print(f"Weight Decay        : {WEIGHT_DECAY}")

    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        # Step the scheduler based on validation loss
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print("\n" + "=" * 50)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("=" * 50)
        print(f"Learning Rate   : {current_lr:.6f}")
        print(f"Training Loss   : {train_loss:.4f}")
        print(f"Validation Loss : {validation_loss:.4f}")

        # Check for improvement and apply Early Stopping
        if validation_loss < (best_validation_loss - EARLY_STOPPING_MIN_DELTA):
            best_validation_loss = validation_loss
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✓ New best model saved! (Validation Loss: {validation_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹ Early stopping triggered after {epoch + 1} epochs.")
            break


if __name__ == "__main__":
    main()