"""
Inference utilities for the ResNet18 model execution path.
"""

from pathlib import Path
import torch

from config import MODEL_DIR
from resnet_model import ChestXrayResNet18

# Safely inherit data ingestion utilities from the baseline inference paths
from predict import prepare_image, build_prediction_result, PredictionResult

RESNET_MODEL_PATH = MODEL_DIR / "resnet18" / "best_model.pth"


def load_resnet_model(device: torch.device) -> ChestXrayResNet18:
    """
    Load the trained ResNet18 model weights for clinical diagnostic execution.
    """
    if not RESNET_MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint archive not found: {RESNET_MODEL_PATH}")

    model = ChestXrayResNet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
    model.eval()
    
    return model


def predict_resnet_image_result(
    image_path: str | Path,
    model: ChestXrayResNet18,
    device: torch.device,
) -> PredictionResult:
    """
    Execute mathematical inference computations on a single input target.
    """
    image = prepare_image(image_path=image_path, device=device)

    with torch.no_grad():
        outputs = model(image)

    return build_prediction_result(outputs)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_resnet_model(device)

    image_path = input("Enter image path for ResNet18 prediction: ")
    result = predict_resnet_image_result(image_path, model, device)

    print("=" * 50)
    print("ResNet18 Prediction Complete")
    print("=" * 50)
    print(f"Prediction : {result.predicted_class}")
    print(f"Confidence : {result.confidence_score:.2f}%")
    print(f"Probabilities (NORMAL, PNEUMONIA): {result.probabilities}")


if __name__ == "__main__":
    main()