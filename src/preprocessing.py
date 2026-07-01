
"""
Image preprocessing utilities for the Chest X-ray project.
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode


from config import IMAGE_SIZE


def get_transforms(train: bool = True):
    """
    Return the preprocessing pipeline.

    Parameters
    ----------
    train : bool
        Indicates whether the transforms are for training data.

    Returns
    -------
    torchvision.transforms.Compose
        A composed sequence of image transformations.
    """

    if train:
        transform_list = [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(
                degrees=7,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]

    else:
        transform_list = [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]

    return transforms.Compose(transform_list)
