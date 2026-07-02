
"""
Image preprocessing utilities for the Chest X-ray project.
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode


from config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    RESIZE_SHORTEST_SIDE,
)


def get_transforms(train: bool = True):
    """
    Return the preprocessing pipeline.

    Both the train and eval pipelines resize the shortest side and center
    crop to ``IMAGE_SIZE`` so the original aspect ratio is preserved. This
    removes the class-correlated geometric distortion introduced by squashing
    images to a square, and normalizes with ImageNet statistics to match the
    pretrained DenseNet121 backbone.

    Parameters
    ----------
    train : bool
        Indicates whether the transforms are for training data.

    Returns
    -------
    torchvision.transforms.Compose
        A composed sequence of image transformations.
    """

    resize_and_crop = [
        transforms.Resize(
            RESIZE_SHORTEST_SIDE,
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(IMAGE_SIZE),
    ]

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),
    ]

    if train:
        transform_list = [
            *resize_and_crop,
            # Horizontal flip is a strong, cheap regularizer for chest films
            # and teaches invariance to laterality when spotting opacities.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=10,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            *normalize,
        ]

    else:
        transform_list = [
            *resize_and_crop,
            *normalize,
        ]

    return transforms.Compose(transform_list)
