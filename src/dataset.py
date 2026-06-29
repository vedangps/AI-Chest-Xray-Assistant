"""
Dataset loading utilities for the Chest X-ray project.
"""

from pathlib import Path

from torchvision.datasets import ImageFolder

from config import DATA_DIR
from preprocessing import get_transforms


def load_dataset(split: str) -> ImageFolder:
    """
    Load one dataset split.

    Parameters
    ----------
    split : str
        One of:
        - train
        - val
        - test

    Returns
    -------
    ImageFolder
        A PyTorch ImageFolder dataset.
    """

    valid_splits = {"train", "val", "test"}

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of {valid_splits}."
        )

    dataset_path: Path = DATA_DIR / "chest_xray" / split

    return ImageFolder(
        root=dataset_path,
        transform=get_transforms(train=(split == "train")),
    )


if __name__ == "__main__":

    train_dataset = load_dataset("train")

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(f"Number of Images : {len(train_dataset)}")
    print(f"Classes          : {train_dataset.classes}")
    print(f"Class Mapping    : {train_dataset.class_to_idx}")