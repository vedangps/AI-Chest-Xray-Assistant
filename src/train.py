"""
Training script for the Chest X-ray CNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
)
from dataloader import create_dataloader
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
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create DataLoaders
    train_loader = create_dataloader("train")
    validation_loader = create_dataloader("val")

    # Create model
    model = ChestXrayCNN().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Ensure the model directory exists
    MODEL_DIR.mkdir(exist_ok=True)

    # Path to save the best model
    best_model_path = MODEL_DIR / "best_model.pth"

    # Track the best validation loss
    best_validation_loss = float("inf")

    print("=" * 50)
    print("Training Setup Complete")
    print("=" * 50)

    print(f"Device              : {device}")
    print(f"Training Batches    : {len(train_loader)}")
    print(f"Validation Batches  : {len(validation_loader)}")
    print(f"Loss Function       : {criterion}")
    print(f"Optimizer           : {optimizer.__class__.__name__}")

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

        print("\n" + "=" * 50)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("=" * 50)
        print(f"Training Loss   : {train_loss:.4f}")
        print(f"Validation Loss : {validation_loss:.4f}")

        # Save model if validation loss improves
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(
                f"✓ New best model saved! "
                f"(Validation Loss: {validation_loss:.4f})"
            )


if __name__ == "__main__":
    main()