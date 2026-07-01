"""
Evaluate the trained Chest X-ray CNN on the test dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MODEL_PATH, CLASS_NAMES
from dataloader import create_dataloader
from model import ChestXrayCNN


def evaluate(
    model,
    test_loader,
    criterion,
    device,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    list[int],
    list[int],
    list[float],
]:
    """
    Evaluate the model on the test dataset.
    """

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []
    all_probabilities = []

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

            # Compute probabilities for ROC-AUC
            probabilities = F.softmax(outputs, dim=1)

            # Predicted class
            predictions = torch.argmax(outputs, dim=1)

            # Store predictions, labels, and probabilities for the positive class (PNEUMONIA)
            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )
            
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    average_loss = running_loss / len(test_loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions,
        all_probabilities,
    )


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

    # Verify checkpoint exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    (
        test_loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions,
        all_probabilities,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # Calculate advanced metrics
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )
    
    # Extract True Negatives, False Positives, False Negatives, and True Positives
    tn, fp, fn, tp = matrix.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    balanced_acc = balanced_accuracy_score(
        all_labels, 
        all_predictions,
    )
    
    roc_auc = roc_auc_score(
        all_labels, 
        all_probabilities,
    )

    print("=" * 50)
    print("Model Evaluation Complete")
    print("=" * 50)

    print(f"Device        : {device}")
    print(f"Test Batches  : {len(test_loader)}")
    print(f"Test Loss     : {test_loss:.4f}\n")

    print("--- Standard Metrics ---")
    print(f"Accuracy      : {accuracy:.2%}")
    print(f"Precision     : {precision:.2%}")
    print(f"Recall        : {recall:.2%}")
    print(f"F1 Score      : {f1:.2%}\n")

    print("--- Advanced Diagnostic Metrics ---")
    print(f"Specificity       : {specificity:.2%}")
    print(f"Balanced Accuracy : {balanced_acc:.2%}")
    print(f"ROC-AUC Score     : {roc_auc:.4f}\n")

    print("--- Classification Report ---")
    print(
        classification_report(
            all_labels, 
            all_predictions, 
            target_names=CLASS_NAMES, 
            digits=4
        )
    )

    print("--- Confusion Matrix Breakdown ---")
    print(f"True NORMAL    (TN): {tn}")
    print(f"False PNEUMONIA (FP): {fp}  <- Healthy lungs misclassified!")
    print(f"False NORMAL   (FN): {fn}  <- Missed pneumonia cases!")
    print(f"True PNEUMONIA  (TP): {tp}")
    print("\nRaw Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()