"""
Evaluate the trained DenseNet121 model on the test dataset.
"""

import torch
import torch.nn as nn
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

from config import MODEL_DIR, CLASS_NAMES
from dataloader import create_dataloader
from densenet_model import ChestXrayDenseNet121
from evaluate_resnet import evaluate_model  # We can safely reuse the exact same math!

DENSENET_MODEL_PATH = MODEL_DIR / "densenet121" / "best_model.pth"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = create_dataloader("test")

    model = ChestXrayDenseNet121(pretrained=False).to(device)

    if not DENSENET_MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {DENSENET_MODEL_PATH}")

    model.load_state_dict(torch.load(DENSENET_MODEL_PATH, map_location=device))
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
    ) = evaluate_model(model, test_loader, criterion, device)

    matrix = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = matrix.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_probabilities)

    print("=" * 50)
    print("DenseNet121 Model Evaluation Diagnostics")
    print("=" * 50)
    print(f"Device        : {device}")
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
    print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES, digits=4))

    print("--- Confusion Matrix Breakdown ---")
    print(f"True NORMAL    (TN): {tn}")
    print(f"False PNEUMONIA (FP): {fp}")
    print(f"False NORMAL   (FN): {fn}")
    print(f"True PNEUMONIA  (TP): {tp}")


if __name__ == "__main__":
    main()