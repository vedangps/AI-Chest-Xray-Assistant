"""
Inference for the X-ray input gate.

Exposes ``load_gate_model`` and ``gate_xray_probability`` — the swap-in used by
``xray_validator`` to decide whether an upload is a chest radiograph.
"""

from functools import lru_cache

import torch
from PIL import Image

from config import (
    XRAY_GATE_MODEL_PATH,
    XRAY_GATE_XRAY_INDEX,
)
from gate_model import XrayGateNet
from preprocessing import get_transforms


def load_gate_model(device: torch.device) -> XrayGateNet:
    """
    Load the trained X-ray gate checkpoint.
    """

    if not XRAY_GATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"X-ray gate checkpoint not found: {XRAY_GATE_MODEL_PATH}"
        )

    model = XrayGateNet(pretrained=False).to(device)
    model.load_state_dict(
        torch.load(XRAY_GATE_MODEL_PATH, map_location=device)
    )
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_gate_runtime() -> tuple[XrayGateNet, torch.device]:
    """
    Load the gate model once and cache it for the process.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_gate_model(device), device


def gate_xray_probability(image: Image.Image) -> float:
    """
    Return the model's probability that ``image`` is a chest X-ray.
    """

    model, device = get_gate_runtime()

    transform = get_transforms(train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probability = torch.softmax(logits, dim=1)[0, XRAY_GATE_XRAY_INDEX]

    return float(probability)
