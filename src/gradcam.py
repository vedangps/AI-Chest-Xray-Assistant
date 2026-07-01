"""
Generate Grad-CAM visualizations for Chest X-ray predictions.
"""

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import (
    ASSETS_DIR,
    CLASS_NAMES,
    DATA_DIR,
)
from predict import (
    build_prediction_result,
    load_model,
    prepare_image,
)


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
TARGET_LAYER_INDEX = 6


@dataclass
class GradCAMResult:
    """
    Container for Grad-CAM outputs.
    """

    image_path: Path
    class_index: int
    predicted_class: str
    confidence_score: float
    heatmap: np.ndarray
    original_image: np.ndarray
    overlay_image: np.ndarray


@dataclass
class GradCAMSummary:
    """
    Human-readable Grad-CAM summary details.
    """

    dominant_region: str
    peak_activation: float
    coverage_percent: float
    explanation: str


class GradCAMHook(AbstractContextManager):
    """
    Capture activations and gradients from one target layer.
    """

    def __init__(self, target_layer) -> None:
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = None
        self._backward_handle = None

    def __enter__(self):
        self._forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()

        if self._backward_handle is not None:
            self._backward_handle.remove()

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()


def find_default_image_path() -> Path:
    """
    Return one sample image from the test split.
    """

    test_dir = DATA_DIR / "chest_xray" / "test"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    for image_path in sorted(test_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            return image_path

    raise FileNotFoundError(
        f"No supported image found in: {test_dir}"
    )


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor into a displayable RGB image.
    """

    image = image.detach().cpu().squeeze(0)
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


def generate_cam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    Generate a normalized Grad-CAM heatmap.
    """

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True,
    )

    cam = torch.sum(
        weights * activations,
        dim=1,
        keepdim=True,
    )

    cam = torch.relu(cam)

    cam = torch.nn.functional.interpolate(
        cam,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    cam = cam.squeeze().detach().cpu()

    cam -= cam.min()

    max_value = cam.max()

    if max_value > 0:
        cam /= max_value

    return cam.numpy()


def create_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap with the input image.
    """

    heatmap_rgb = cm.jet(heatmap)[..., :3]

    overlay = (1 - alpha) * original_image + alpha * heatmap_rgb

    return np.clip(overlay, 0, 1)


def compute_gradcam(
    image_path: str | Path,
    model,
    device: torch.device,
    target_class_index: int | None = None,
    alpha: float = 0.4,
) -> GradCAMResult:
    """
    Run one forward and backward pass to compute Grad-CAM.
    """

    image_path = Path(image_path)

    image = prepare_image(
        image_path=image_path,
        device=device,
    )

    target_layer = model.features[TARGET_LAYER_INDEX]

    with GradCAMHook(target_layer) as hook:

        outputs = model(image)

        prediction_result = build_prediction_result(outputs)

        if target_class_index is None:
            target_class_index = prediction_result.class_index

        confidence_score = (
            prediction_result.probabilities[target_class_index]
            * 100
        )

        model.zero_grad()

        outputs[0, target_class_index].backward()

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError(
                "Failed to capture activations or gradients "
                "for Grad-CAM."
            )

        heatmap = generate_cam(
            activations=hook.activations,
            gradients=hook.gradients,
            output_size=image.shape[-2:],
        )

    original_image = denormalize_image(image)

    overlay_image = create_overlay(
        original_image=original_image,
        heatmap=heatmap,
        alpha=alpha,
    )

    return GradCAMResult(
        image_path=image_path,
        class_index=target_class_index,
        predicted_class=CLASS_NAMES[target_class_index],
        confidence_score=confidence_score,
        heatmap=heatmap,
        original_image=original_image,
        overlay_image=overlay_image,
    )


def summarize_gradcam(
    result: GradCAMResult,
) -> GradCAMSummary:
    """
    Convert the raw heatmap into a short educational summary.
    """

    heatmap = result.heatmap

    row_slices = np.array_split(
        np.arange(heatmap.shape[0]),
        3,
    )
    column_slices = np.array_split(
        np.arange(heatmap.shape[1]),
        3,
    )

    region_labels = (
        ("upper-left", "upper-center", "upper-right"),
        ("middle-left", "middle-center", "middle-right"),
        ("lower-left", "lower-center", "lower-right"),
    )

    dominant_region = "middle-center"
    dominant_score = float("-inf")

    for row_index, row_values in enumerate(row_slices):
        for column_index, column_values in enumerate(column_slices):
            region_score = float(
                heatmap[
                    row_values[0]:row_values[-1] + 1,
                    column_values[0]:column_values[-1] + 1,
                ].mean()
            )

            if region_score > dominant_score:
                dominant_score = region_score
                dominant_region = region_labels[row_index][column_index]

    peak_activation = float(heatmap.max() * 100)

    coverage_percent = float(
        (heatmap >= 0.6).mean() * 100
    )

    explanation = (
        "Grad-CAM highlighted the "
        f"{dominant_region} region most strongly, "
        f"with a peak activation of {peak_activation:.1f}% "
        f"and elevated attention across {coverage_percent:.1f}% "
        "of the image. This visualization reflects where the "
        "model focused during classification, not a confirmed "
        "site of disease."
    )

    return GradCAMSummary(
        dominant_region=dominant_region,
        peak_activation=peak_activation,
        coverage_percent=coverage_percent,
        explanation=explanation,
    )


def save_overlay_image(
    result: GradCAMResult,
    output_path: str | Path,
) -> Path:
    """
    Save the Grad-CAM overlay image to disk.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay_uint8 = (
        result.overlay_image * 255
    ).astype(np.uint8)

    Image.fromarray(overlay_uint8).save(output_path)

    return output_path


def save_visualization(
    result: GradCAMResult,
    output_path: str | Path,
) -> Path:
    """
    Save the original image, heatmap, and overlay to disk.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(result.original_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(result.heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(result.overlay_image)
    axes[2].set_title(
        f"{result.predicted_class} "
        f"({result.confidence_score:.2f}%)"
    )
    axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate a Grad-CAM visualization for one "
            "Chest X-ray image."
        )
    )

    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help=(
            "Path to the input image. "
            "Defaults to one image from the test split."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save the Grad-CAM visualization. "
            "Defaults to assets/gradcam_<image-name>.png."
        ),
    )

    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        help=(
            "Optional class index to explain. "
            "Defaults to the predicted class."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Heatmap overlay intensity between 0 and 1.",
    )

    return parser


def resolve_output_path(
    image_path: Path,
    output_path: Path | None,
) -> Path:
    """
    Determine the visualization output path.
    """

    if output_path is not None:
        return output_path

    return ASSETS_DIR / f"gradcam_{image_path.stem}.png"


def main() -> None:
    """
    Load the model, generate Grad-CAM, and save the result.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.class_index is not None:
        if arguments.class_index < 0 or arguments.class_index >= len(CLASS_NAMES):
            raise ValueError(
                "class-index must be between "
                f"0 and {len(CLASS_NAMES) - 1}."
            )

    if not 0 <= arguments.alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    image_path = (
        arguments.image_path
        if arguments.image_path is not None
        else find_default_image_path()
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    result = compute_gradcam(
        image_path=image_path,
        model=model,
        device=device,
        target_class_index=arguments.class_index,
        alpha=arguments.alpha,
    )

    output_path = resolve_output_path(
        image_path=result.image_path,
        output_path=arguments.output_path,
    )

    saved_path = save_visualization(
        result=result,
        output_path=output_path,
    )

    print("=" * 50)
    print("Grad-CAM Complete")
    print("=" * 50)
    print(f"Device        : {device}")
    print(f"Input Image   : {result.image_path}")
    print(f"Prediction    : {result.predicted_class}")
    print(f"Confidence    : {result.confidence_score:.2f}%")
    print(f"Saved Output  : {saved_path}")


if __name__ == "__main__":
    main()
