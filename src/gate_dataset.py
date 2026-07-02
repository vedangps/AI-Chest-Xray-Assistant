"""
Dataset utilities for the X-ray input gate.

Expects a two-folder layout::

    data/xray_gate/
        xray/       # chest radiographs (positives)
        not_xray/   # everything else: natural images + other-body-part X-rays

Positives: reuse your chest X-ray images. Negatives: a few thousand natural
images (e.g. Imagenette) plus other-modality / other-body-part X-rays
(e.g. MURA) so the gate learns "chest X-ray", not merely "grayscale".
"""

from collections import Counter
from functools import lru_cache
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import (
    NUM_WORKERS,
    RANDOM_SEED,
    VALIDATION_SPLIT,
    XRAY_GATE_CLASS_NAMES,
    XRAY_GATE_DATA_DIR,
)
from dataset import ChestXrayDataset


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def _collect_samples() -> list[tuple[Path, int]]:
    """
    Scan the gate folders and return (path, label) pairs.
    """

    samples: list[tuple[Path, int]] = []

    for label, class_name in enumerate(XRAY_GATE_CLASS_NAMES):
        class_dir = XRAY_GATE_DATA_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Gate class directory not found: {class_dir}. Create "
                f"{XRAY_GATE_DATA_DIR} with '{'/'.join(XRAY_GATE_CLASS_NAMES)}' "
                "subfolders of images."
            )

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))

    if not samples:
        raise FileNotFoundError(
            f"No images found under {XRAY_GATE_DATA_DIR}."
        )

    return samples


@lru_cache(maxsize=1)
def _stratified_split() -> tuple[
    list[tuple[Path, int]],
    list[tuple[Path, int]],
]:
    """
    Build a reproducible stratified train/validation split.
    """

    samples = _collect_samples()
    labels = [label for _, label in samples]

    train_indices, val_indices = train_test_split(
        list(range(len(samples))),
        test_size=VALIDATION_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    train_samples = [samples[i] for i in sorted(train_indices)]
    val_samples = [samples[i] for i in sorted(val_indices)]
    return train_samples, val_samples


def gate_class_weights() -> tuple[float, ...]:
    """
    Balanced class weights for the training split.
    """

    train_samples, _ = _stratified_split()
    counts = Counter(label for _, label in train_samples)
    total = sum(counts.values())
    num_classes = len(XRAY_GATE_CLASS_NAMES)

    return tuple(
        total / (num_classes * max(counts.get(index, 0), 1))
        for index in range(num_classes)
    )


def create_gate_dataloader(
    split: str,
    batch_size: int,
) -> DataLoader:
    """
    Create a DataLoader for the "train" or "val" gate split.
    """

    if split not in {"train", "val"}:
        raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'val'.")

    train_samples, val_samples = _stratified_split()
    samples = train_samples if split == "train" else val_samples

    dataset = ChestXrayDataset(samples=samples, train=(split == "train"))

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
    )
