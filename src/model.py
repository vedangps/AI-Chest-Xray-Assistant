"""
CNN model architecture for Chest X-ray classification.
"""

import torch
import torch.nn as nn


class ChestXrayCNN(nn.Module):
    """
    Simple CNN for binary chest X-ray classification.
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = ChestXrayCNN()

    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    print(model)

    print(f"\nOutput Shape: {output.shape}")