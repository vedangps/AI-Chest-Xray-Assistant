"""
Training script for the Chest X-ray CNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import LEARNING_RATE, NUM_EPOCHS
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

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    return average_loss


def main() -> None:
    """
    Initialize all training components.
    """

    # Select CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader = create_dataloader("train")

    # Move model to device
    model = ChestXrayCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("=" * 50)
    print("Training Setup Complete")
    print("=" * 50)

    print(f"Device           : {device}")
    print(f"Training Batches : {len(train_loader)}")
    print(f"Loss Function    : {criterion}")
    print(f"Optimizer        : {optimizer.__class__.__name__}")

    for epoch in range(NUM_EPOCHS):

        average_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )


if __name__ == "__main__":
    main()