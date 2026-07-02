"""
Input gatekeeper: reject uploads that are not chest radiographs before they
reach the diagnostic model.

Two implementations behind one interface (`validate_chest_xray`):

1. **Learned gate** (preferred) — a trained chest-X-ray-vs-not classifier. Used
   automatically when ``models/xray_gate/best_model.pth`` is present. Train it
   with ``src/train_gate.py``. It catches the hard cases a heuristic can't,
   e.g. a grayscale X-ray of a different body part.
2. **Heuristic fallback** (always available) — a cheap statistical check that
   needs no extra weights. Chest X-rays are grayscale with real radiographic
   contrast; ordinary photos are colorful. Thresholds are calibrated on the
   project's data (real radiographs measure channel spread / saturation ~0.0
   with intensity std ~0.16-0.24, vs ~0.35 / ~0.50 for colored images).

The app calls ``validate_chest_xray`` and never needs to know which path ran.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


# Grayscale tolerance: real radiographs sit at ~0.0; colored photos at ~0.35.
MAX_CHANNEL_SPREAD = 0.06
# Reject near-blank frames (a solid color has std ~0.0; X-rays are >=0.15).
MIN_INTENSITY_STD = 0.03
# Plausible framing for a chest film; rejects banners/panoramas.
MIN_ASPECT_RATIO = 0.33
MAX_ASPECT_RATIO = 3.0


@dataclass(frozen=True)
class XrayValidation:
    """
    Outcome of the chest X-ray input check.
    """

    is_xray: bool
    reason: str
    channel_spread: float
    intensity_std: float
    aspect_ratio: float
    method: str = "heuristic"
    score: float | None = None  # model P(xray) when method == "model"


def _measure(image: Image.Image) -> tuple[float, float, float]:
    """
    Return (channel_spread, intensity_std, aspect_ratio) for one image.
    """

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    channel_spread = float((rgb.max(axis=2) - rgb.min(axis=2)).mean() / 255.0)

    gray = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    intensity_std = float(gray.std())

    width, height = image.size
    aspect_ratio = float(width / height) if height else 0.0

    return channel_spread, intensity_std, aspect_ratio


def _heuristic_validation(
    channel_spread: float,
    intensity_std: float,
    aspect_ratio: float,
) -> XrayValidation:
    """
    Grayscale, contrast, and framing checks; first failure wins.
    """

    if channel_spread > MAX_CHANNEL_SPREAD:
        reason = (
            "The image is in color. Chest X-rays are grayscale, so this "
            "looks like a photo or screenshot rather than a radiograph."
        )
        is_xray = False
    elif intensity_std < MIN_INTENSITY_STD:
        reason = (
            "The image is nearly blank and shows no radiographic contrast."
        )
        is_xray = False
    elif not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        reason = (
            "The image proportions are not consistent with a chest "
            "radiograph."
        )
        is_xray = False
    else:
        reason = "Passed chest X-ray input checks."
        is_xray = True

    return XrayValidation(
        is_xray=is_xray,
        reason=reason,
        channel_spread=channel_spread,
        intensity_std=intensity_std,
        aspect_ratio=aspect_ratio,
        method="heuristic",
    )


def _learned_validation(
    image: Image.Image,
    channel_spread: float,
    intensity_std: float,
    aspect_ratio: float,
) -> XrayValidation | None:
    """
    Use the trained gate if its checkpoint exists, else return None so the
    caller falls back to the heuristic. Never raises: any failure degrades
    gracefully to the heuristic path.
    """

    # Imported lazily so the heuristic path stays torch-free.
    from config import XRAY_GATE_MODEL_PATH, XRAY_GATE_THRESHOLD

    if not XRAY_GATE_MODEL_PATH.exists():
        return None

    try:
        from predict_gate import gate_xray_probability

        probability = gate_xray_probability(image)
    except Exception:
        LOGGER.exception(
            "X-ray gate model failed; falling back to heuristic checks."
        )
        return None

    is_xray = probability >= XRAY_GATE_THRESHOLD
    reason = (
        "Classified as a chest radiograph by the trained input gate."
        if is_xray
        else (
            "The trained input gate does not recognize this as a chest "
            "radiograph."
        )
    )

    return XrayValidation(
        is_xray=is_xray,
        reason=reason,
        channel_spread=channel_spread,
        intensity_std=intensity_std,
        aspect_ratio=aspect_ratio,
        method="model",
        score=probability,
    )


def validate_chest_xray(image: Image.Image) -> XrayValidation:
    """
    Decide whether an image is plausibly a chest radiograph.

    Prefers the learned gate when available and falls back to the heuristic
    statistical checks otherwise.
    """

    channel_spread, intensity_std, aspect_ratio = _measure(image)

    learned = _learned_validation(
        image,
        channel_spread,
        intensity_std,
        aspect_ratio,
    )
    if learned is not None:
        return learned

    return _heuristic_validation(
        channel_spread,
        intensity_std,
        aspect_ratio,
    )
