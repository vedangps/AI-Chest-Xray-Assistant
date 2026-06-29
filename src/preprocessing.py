
"""
Image preprocessing utilities for the Chest X-ray project.
"""

from torchvision import transforms


IMAGE_SIZE = (224, 224)


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

    transform_list = [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,0.5,0.5),
            std=(0.5,0.5,0.5)
        ),
    ]

    return transforms.Compose(transform_list)