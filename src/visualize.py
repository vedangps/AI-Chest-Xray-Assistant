"""
Visualize sample images from the training DataLoader.
"""

import matplotlib.pyplot as plt
import torch

from dataloader import create_dataloader


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """
    Reverse normalization so the image can be displayed correctly.
    """

    image = image * 0.5 + 0.5

    return image.clamp(0, 1)


def visualize_batch() -> None:
    """
    Display six images from one training batch.
    """

    train_loader = create_dataloader("train")

    images, labels = next(iter(train_loader))

    plt.figure(figsize=(12, 8))

    for i in range(6):

        plt.subplot(2, 3, i + 1)

        image = denormalize(images[i])

        plt.imshow(image.permute(1, 2, 0))

        plt.title(CLASS_NAMES[labels[i].item()])

        plt.axis("off")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    visualize_batch()