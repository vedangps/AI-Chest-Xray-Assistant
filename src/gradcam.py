"""
Generate Grad-CAM visualizations for Chest X-ray predictions.
"""

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from config import (
    ASSETS_DIR,
    CLASS_NAMES,
    DATA_DIR,
    DEFAULT_DECISION_THRESHOLD,
    GRADCAM_OVERLAY_ALPHA,
    GRADCAM_SMOOTHING_SIGMA,
    GRADCAM_SUPPRESSION_THRESHOLD,
)
from predict import (
    build_prediction_result,
    prepare_image,
)
from calibration import (
    load_threshold_calibration,
    resolve_decision_threshold,
)
from predict_densenet import load_densenet_model


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass
class GradCAMResult:
    """
    Container for Grad-CAM outputs.
    """

    image_path: Path
    class_index: int
    predicted_class: str
    confidence_score: float
    probabilities: tuple[float, ...]
    decision_threshold: float
    heatmap: np.ndarray
    original_image: np.ndarray
    overlay_image: np.ndarray


class GradCAMHook(AbstractContextManager):
    """
    Capture activations and gradients from one target layer.
    """

    def __init__(self, target_layer) -> None:
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = None
        self._gradient_handle = None

    def __enter__(self):
        self._forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()

        if self._gradient_handle is not None:
            self._gradient_handle.remove()

    def _save_activation(self, module, inputs, output) -> None:
        # Clone hook outputs to avoid autograd view/in-place conflicts
        # raised by DenseNet backward execution.
        self.activations = output.detach().clone()
        if output.requires_grad:
            self._gradient_handle = output.register_hook(
                self._save_gradient
            )

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach().clone()


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
    smoothing_sigma: float = GRADCAM_SMOOTHING_SIGMA,
    suppression_threshold: float = GRADCAM_SUPPRESSION_THRESHOLD,
) -> np.ndarray:
    """
    Generate a smoothed, artifact-suppressed Grad-CAM heatmap.

    The raw class-activation map is min-max normalized, Gaussian smoothed to
    remove upsampling haze, and thresholded so low-intensity background/edge
    activations are dropped and the surviving pathology span is rescaled to
    the full [0, 1] range.
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

    cam = cam.squeeze().detach().cpu().numpy()

    # Smooth away the speckled, diffuse activations that come from
    # upsampling the coarse feature map to full image resolution.
    if smoothing_sigma > 0:
        cam = gaussian_filter(cam, sigma=smoothing_sigma)

    # Min-max normalization into [0, 1].
    cam -= cam.min()
    max_value = cam.max()
    if max_value > 0:
        cam /= max_value

    # Suppress low-intensity activations (background noise, rib/heart
    # borders) and rescale the retained high-confidence region back to
    # [0, 1] for clear localization.
    if 0.0 < suppression_threshold < 1.0:
        cam = np.clip(
            (cam - suppression_threshold) / (1.0 - suppression_threshold),
            0.0,
            1.0,
        )

    return cam


def create_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap with the input image.

    Blending opacity is modulated per pixel by activation strength, so
    suppressed/cold regions show the untouched radiograph and only genuine
    pathology is tinted, avoiding a uniform color wash over the whole image.
    """

    heatmap_rgb = cm.jet(heatmap)[..., :3]

    per_pixel_alpha = (alpha * heatmap)[..., np.newaxis]

    overlay = (
        (1 - per_pixel_alpha) * original_image
        + per_pixel_alpha * heatmap_rgb
    )

    return np.clip(overlay, 0, 1)


def resolve_target_layer(model: Any):
    """
    Resolve the last convolutional layer for supported model families.
    """

    if (
        hasattr(model, "backbone")
        and hasattr(model.backbone, "features")
    ):
        # Target the final dense block rather than the whole feature
        # extractor. Its output is the deepest convolutional feature map
        # (before the terminal BatchNorm/ReLU), giving the best balance of
        # semantic meaning and spatial resolution for localization.
        features = model.backbone.features
        if hasattr(features, "denseblock4"):
            return features.denseblock4
        return features

    if hasattr(model, "features"):
        return model.features[16]

    raise AttributeError(
        "Unable to resolve a Grad-CAM target layer for the provided model."
    )


def compute_gradcam(
    image_path: str | Path,
    model,
    device: torch.device,
    target_class_index: int | None = None,
    alpha: float = GRADCAM_OVERLAY_ALPHA,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> GradCAMResult:
    """
    Run one forward and backward pass to compute Grad-CAM.
    """

    image_path = Path(image_path)

    image = prepare_image(
        image_path=image_path,
        device=device,
    )

    target_layer = resolve_target_layer(model)

    with GradCAMHook(target_layer) as hook:

        outputs = model(image)

        prediction_result = build_prediction_result(
            outputs,
            decision_threshold=decision_threshold,
        )

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
        probabilities=prediction_result.probabilities,
        decision_threshold=prediction_result.decision_threshold,
        heatmap=heatmap,
        original_image=original_image,
        overlay_image=overlay_image,
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
        if (
            arguments.class_index < 0
            or arguments.class_index >= len(CLASS_NAMES)
        ):
            raise ValueError(
                "class-index must be between "
                f"0 and {len(CLASS_NAMES) - 1}."
            )

    if not 0 <= arguments.alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    # Inlined default image resolution logic
    image_path = arguments.image_path
    if image_path is None:
        test_dir = DATA_DIR / "chest_xray" / "test"
        if not test_dir.exists():
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}"
            )
        
        found_path = None
        for path in sorted(test_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                found_path = path
                break
        
        if found_path is None:
            raise FileNotFoundError(
                f"No supported image found in: {test_dir}"
            )
        image_path = found_path

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_densenet_model(device)
    calibration = load_threshold_calibration()

    result = compute_gradcam(
        image_path=image_path,
        model=model,
        device=device,
        target_class_index=arguments.class_index,
        alpha=arguments.alpha,
        decision_threshold=resolve_decision_threshold(calibration),
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
