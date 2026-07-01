"""
DenseNet121 Transfer Learning model architecture for Chest X-ray classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChestXrayDenseNet121(nn.Module):
    """
    DenseNet121 backbone with a custom classifier for binary classification.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        
        # Load the standard DenseNet121 backbone
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        # Extract the input dimensions of the original classifier head
        in_features = self.backbone.classifier.in_features
        
        # Replace the classifier with a custom 2-class linear output head
        self.backbone.classifier = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DenseNet121 network pipeline.
        """
        return self.backbone(x)


if __name__ == "__main__":
    model = ChestXrayDenseNet121(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("=" * 60)
    print("DenseNet121 Model Structural Head Summary")
    print("=" * 60)
    print(f"Output Tensor Shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 60)