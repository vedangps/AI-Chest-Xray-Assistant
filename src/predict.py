"""
Inference utilities for the Chest X-ray CNN.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from config import (
    CLASS_NAMES,
    MODEL_PATH,
)
from model import ChestXrayCNN
from preprocessing import get_transforms


@dataclass
class PredictionResult:
    """
    Structured prediction output for one image.
    """

    class_index: int
    predicted_class: str
    confidence_score: float
    probabilities: tuple[float, ...]


def load_model(device: torch.device) -> ChestXrayCNN:
    """
    Load the trained CNN model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    model = ChestXrayCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    return model


def prepare_image(
    image_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """
    Load and preprocess a single image.
    """

    image = Image.open(image_path).convert("RGB")

    transform = get_transforms(train=False)

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    return image


def build_prediction_result(
    outputs: torch.Tensor,
) -> PredictionResult:
    """
    Convert raw model outputs into prediction metadata.
    """

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1,
    )

    class_index = prediction.item()

    predicted_class = CLASS_NAMES[class_index]

    confidence_score = confidence.item() * 100

    probability_values = tuple(
        float(value)
        for value in probabilities[0].detach().cpu().tolist()
    )

    return PredictionResult(
        class_index=class_index,
        predicted_class=predicted_class,
        confidence_score=confidence_score,
        probabilities=probability_values,
    )


def predict_image_result(
    image_path: str | Path,
    model: ChestXrayCNN,
    device: torch.device,
) -> PredictionResult:
    """
    Run inference on one image and return structured output.
    """

    image = prepare_image(
        image_path=image_path,
        device=device,
    )

    with torch.no_grad():
        outputs = model(image)

    return build_prediction_result(outputs)


def predict_image(
    image_path: str | Path,
    model: ChestXrayCNN,
    device: torch.device,
) -> tuple[str, float]:

    result = predict_image_result(
        image_path=image_path,
        model=model,
        device=device,
    )

    return (
        result.predicted_class,
        result.confidence_score,
    )


def main() -> None:

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    image_path = input(
        "Enter image path: "
    )

    prediction, confidence = predict_image(
        image_path=image_path,
        model=model,
        device=device,
    )

    print("=" * 50)
    print("Prediction Complete")
    print("=" * 50)
    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")


if __name__ == "__main__":
    main()
