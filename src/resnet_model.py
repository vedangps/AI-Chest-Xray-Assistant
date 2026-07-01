"""
ResNet18 Transfer Learning model architecture for Chest X-ray classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChestXrayResNet18(nn.Module):
    """
    ResNet18 backbone with a custom fully connected layer for binary classification.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        
        # Load the standard ResNet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Extract the input dimensions of the original classifier head
        in_features = self.backbone.fc.in_features
        
        # Replace the fully connected layer with a custom 2-class linear output head
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet18 network pipeline.
        """
        return self.backbone(x)


if __name__ == "__main__":
    model = ChestXrayResNet18(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("=" * 60)
    print("ResNet18 Model Structural Head Summary")
    print("=" * 60)
    print(f"Output Tensor Shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 60)