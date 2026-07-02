"""
Model for the X-ray input gate: a binary "chest X-ray vs. not" classifier.

Chest-vs-everything-else is an easy boundary, so a lightweight ResNet18
backbone with a dropout-regularized 2-class head is plenty and keeps CPU
inference fast on the deployment host.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class XrayGateNet(nn.Module):
    """
    ResNet18 backbone with a 2-class head: index 0 = not_xray, 1 = xray.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    model = XrayGateNet(pretrained=False)
    dummy = torch.randn(1, 3, 224, 224)
    print("Output shape:", tuple(model(dummy).shape))
