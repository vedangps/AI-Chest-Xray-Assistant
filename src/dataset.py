"""
Dataset loading utilities for the Chest X-ray project.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import (
    CLASS_NAMES,
    DATA_DIR,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)
from preprocessing import get_transforms


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class DatasetSummary:
    """
    Summary statistics for one dataset split.
    """

    split: str
    class_counts: dict[str, int]
    total_images: int


class ChestXrayDataset(Dataset):
    """
    Dataset backed by an explicit list of image paths.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        train: bool,
    ) -> None:
        self.samples = samples
        self.targets = [
            label
            for _, label in samples
        ]
        self.transform = get_transforms(train=train)
        self.classes = list(CLASS_NAMES)
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(CLASS_NAMES)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple:
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, label


def _collect_split_samples(
    split: str,
) -> list[tuple[Path, int]]:
    """
    Collect all image paths for one original split.
    """

    split_path = DATA_DIR / "chest_xray" / split

    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_path}"
        )

    samples = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_path = split_path / class_name

        if not class_path.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_path}"
            )

        image_paths = sorted(
            path
            for path in class_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        samples.extend(
            (image_path, class_index)
            for image_path in image_paths
        )

    return samples


@lru_cache(maxsize=1)
def get_original_split_summaries() -> dict[str, DatasetSummary]:
    """
    Summarize the original dataset splits on disk.
    """

    summaries = {}

    for split in ("train", "val", "test"):
        samples = _collect_split_samples(split)

        counts = Counter(
            label
            for _, label in samples
        )

        class_counts = {
            class_name: counts.get(index, 0)
            for index, class_name in enumerate(CLASS_NAMES)
        }

        summaries[split] = DatasetSummary(
            split=split,
            class_counts=class_counts,
            total_images=len(samples),
        )

    return summaries


@lru_cache(maxsize=1)
def get_stratified_training_samples() -> tuple[
    list[tuple[Path, int]],
    list[tuple[Path, int]],
]:
    """
    Build a reproducible stratified train/validation split.
    """

    original_train_samples = _collect_split_samples("train")

    indices = list(range(len(original_train_samples)))
    labels = [
        label
        for _, label in original_train_samples
    ]

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    train_samples = [
        original_train_samples[index]
        for index in sorted(train_indices)
    ]

    validation_samples = [
        original_train_samples[index]
        for index in sorted(validation_indices)
    ]

    return train_samples, validation_samples


@lru_cache(maxsize=1)
def get_operational_split_summaries() -> dict[str, DatasetSummary]:
    """
    Summarize the effective splits used by the ML pipeline.
    """

    train_samples, validation_samples = get_stratified_training_samples()
    test_samples = _collect_split_samples("test")

    split_samples = {
        "train": train_samples,
        "val": validation_samples,
        "test": test_samples,
    }

    summaries = {}

    for split, samples in split_samples.items():
        counts = Counter(
            label
            for _, label in samples
        )

        class_counts = {
            class_name: counts.get(index, 0)
            for index, class_name in enumerate(CLASS_NAMES)
        }

        summaries[split] = DatasetSummary(
            split=split,
            class_counts=class_counts,
            total_images=len(samples),
        )

    return summaries


def get_training_class_weights() -> tuple[float, ...]:
    """
    Compute balanced class weights from the effective training split.
    """

    train_summary = get_operational_split_summaries()["train"]

    class_counts = [
        train_summary.class_counts[class_name]
        for class_name in CLASS_NAMES
    ]

    total_samples = sum(class_counts)
    class_count = len(class_counts)

    return tuple(
        total_samples / (class_count * count)
        for count in class_counts
    )


def load_dataset(split: str) -> ChestXrayDataset:
    """
    Load one effective dataset split.

    Parameters
    ----------
    split : str
        One of:
        - train
        - val
        - test

    Returns
    -------
    ChestXrayDataset
        A PyTorch dataset with the correct transforms.
    """

    valid_splits = {"train", "val", "test"}

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of {valid_splits}."
        )

    if split == "train":
        samples, _ = get_stratified_training_samples()
        return ChestXrayDataset(
            samples=samples,
            train=True,
        )

    if split == "val":
        _, samples = get_stratified_training_samples()
        return ChestXrayDataset(
            samples=samples,
            train=False,
        )

    return ChestXrayDataset(
        samples=_collect_split_samples("test"),
        train=False,
    )


if __name__ == "__main__":

    original_summaries = get_original_split_summaries()
    operational_summaries = get_operational_split_summaries()

    print("=" * 50)
    print("Original Dataset Splits")
    print("=" * 50)

    for split in ("train", "val", "test"):
        summary = original_summaries[split]
        print(
            f"{split:<5} "
            f"Total={summary.total_images:<5} "
            f"NORMAL={summary.class_counts['NORMAL']:<5} "
            f"PNEUMONIA={summary.class_counts['PNEUMONIA']:<5}"
        )

    print("\n" + "=" * 50)
    print("Operational Dataset Splits")
    print("=" * 50)

    for split in ("train", "val", "test"):
        summary = operational_summaries[split]
        print(
            f"{split:<5} "
            f"Total={summary.total_images:<5} "
            f"NORMAL={summary.class_counts['NORMAL']:<5} "
            f"PNEUMONIA={summary.class_counts['PNEUMONIA']:<5}"
        )

    print("\nClass Weights")
    print(get_training_class_weights())
