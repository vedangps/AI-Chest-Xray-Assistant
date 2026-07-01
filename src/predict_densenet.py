"""
Inference utilities for the DenseNet121 model execution path.
"""

from pathlib import Path

import torch

from calibration import (
    load_threshold_calibration,
    resolve_decision_threshold,
)
from config import (
    DEFAULT_DECISION_THRESHOLD,
    DENSENET_MODEL_PATH,
)
from densenet_model import ChestXrayDenseNet121
from predict import (
    PredictionResult,
    build_prediction_result,
    prepare_image,
)


def load_densenet_model(device: torch.device) -> ChestXrayDenseNet121:
    """
    Load the trained DenseNet121 checkpoint.
    """

    if not DENSENET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint archive not found: {DENSENET_MODEL_PATH}"
        )

    model = ChestXrayDenseNet121(pretrained=False).to(device)
    model.load_state_dict(
        torch.load(
            DENSENET_MODEL_PATH,
            map_location=device,
        )
    )
    model.eval()

    return model


def predict_densenet_image_result(
    image_path: str | Path,
    model: ChestXrayDenseNet121,
    device: torch.device,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> PredictionResult:
    """
    Run DenseNet121 inference on one image.
    """

    image = prepare_image(image_path=image_path, device=device)

    with torch.no_grad():
        outputs = model(image)

    return build_prediction_result(
        outputs,
        decision_threshold=decision_threshold,
    )


def main() -> None:
    """
    Execute a single DenseNet121 prediction from the CLI.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_densenet_model(device)
    calibration = load_threshold_calibration()

    image_path = input("Enter image path for DenseNet121 prediction: ")
    threshold = resolve_decision_threshold(calibration)
    result = predict_densenet_image_result(
        image_path=image_path,
        model=model,
        device=device,
        decision_threshold=threshold,
    )

    print("=" * 50)
    print("DenseNet121 Prediction Complete")
    print("=" * 50)
    print(f"Prediction : {result.predicted_class}")
    print(f"Confidence : {result.confidence_score:.2f}%")
    print(
        "Pneumonia Probability : "
        f"{result.pneumonia_probability * 100:.2f}%"
    )
    print(f"Threshold Used        : {result.decision_threshold:.3f}")


if __name__ == "__main__":
    main()
