"""
CNN model architecture for Chest X-ray classification.
"""

import torch
import torch.nn as nn


class ChestXrayCNN(nn.Module):
    """
    Enhanced Custom Deep CNN with Global Average Pooling for binary chest X-ray classification.
    """

    def __init__(self) -> None:
        super().__init__()

        # Feature Extractor: Deep hierarchical processing with Batch Normalization
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 5: 14x14 -> 7x7 (Keeps final conv map accessible for Grad-CAM)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Global Average Pooling: Replaces the huge flattening operation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Block: Regularized dense processing path
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    model = ChestXrayCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("=" * 60)
    print("Model Structural Summary")
    print("=" * 60)
    print(model)

    # Calculate learnable parameter counts precisely
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 60)
    print(f"Output Tensor Shape      : {output.shape}")
    print(f"Total Learnable Parameters: {total_params:,}")
    print("=" * 60)