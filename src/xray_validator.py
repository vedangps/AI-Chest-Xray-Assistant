"""
Input gatekeeper: reject uploads that are not chest radiographs before they
reach the diagnostic model.

A pneumonia classifier will happily return a confident (and meaningless)
prediction for a selfie or a meme, so we screen the input first. Chest X-rays
have two robust, cheap-to-measure properties that separate them from ordinary
photos: they are **grayscale** and they carry real radiographic **contrast**.

This is a lightweight statistical validator, not a learned model: it needs no
extra weights and runs in milliseconds on CPU. Thresholds are calibrated on
the project's own data, where real radiographs measure a per-pixel channel
spread and saturation of ~0.0 (perfectly gray) with intensity std ~0.16-0.24,
versus ~0.35 / ~0.50 for colored images.

Limitation: a grayscale *non*-medical image (e.g. a black-and-white photo)
can still pass. For stricter screening, swap `validate_chest_xray` for a
trained medical/non-medical classifier behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


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


def validate_chest_xray(image: Image.Image) -> XrayValidation:
    """
    Decide whether an image is plausibly a chest radiograph.

    Runs three ordered checks (grayscale, contrast, framing) and returns the
    first failure with a human-readable reason, or a passing result.
    """

    channel_spread, intensity_std, aspect_ratio = _measure(image)

    if channel_spread > MAX_CHANNEL_SPREAD:
        return XrayValidation(
            is_xray=False,
            reason=(
                "The image is in color. Chest X-rays are grayscale, so this "
                "looks like a photo or screenshot rather than a radiograph."
            ),
            channel_spread=channel_spread,
            intensity_std=intensity_std,
            aspect_ratio=aspect_ratio,
        )

    if intensity_std < MIN_INTENSITY_STD:
        return XrayValidation(
            is_xray=False,
            reason=(
                "The image is nearly blank and shows no radiographic "
                "contrast."
            ),
            channel_spread=channel_spread,
            intensity_std=intensity_std,
            aspect_ratio=aspect_ratio,
        )

    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return XrayValidation(
            is_xray=False,
            reason=(
                "The image proportions are not consistent with a chest "
                "radiograph."
            ),
            channel_spread=channel_spread,
            intensity_std=intensity_std,
            aspect_ratio=aspect_ratio,
        )

    return XrayValidation(
        is_xray=True,
        reason="Passed chest X-ray input checks.",
        channel_spread=channel_spread,
        intensity_std=intensity_std,
        aspect_ratio=aspect_ratio,
    )
