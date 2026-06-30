"""
Evaluate the trained Chest X-ray CNN on the test dataset.
"""

import torch
import torch.nn as nn

from config import MODEL_DIR
from dataloader import create_dataloader
from model import ChestXrayCNN


def evaluate(
    model,
    test_loader,
    criterion,
    device,
) -> float:
    """
    Evaluate the model on the test dataset.
    """

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            running_loss += loss.item()

    average_loss = running_loss / len(test_loader)

    return average_loss


def main() -> None:
    """
    Load the trained model and evaluate it on the test dataset.
    """

    # Select CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create test DataLoader
    test_loader = create_dataloader("test")

    # Create model
    model = ChestXrayCNN().to(device)

    # Path to saved model
    best_model_path = MODEL_DIR / "best_model.pth"

    # Verify checkpoint exists
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {best_model_path}"
        )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
        )
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate model
    test_loss = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("=" * 50)
    print("Model Evaluation Complete")
    print("=" * 50)
    print(f"Device        : {device}")
    print(f"Test Batches  : {len(test_loader)}")
    print(f"Test Loss     : {test_loss:.4f}")


if __name__ == "__main__":
    main()