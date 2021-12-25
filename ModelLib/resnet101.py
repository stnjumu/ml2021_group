import torch
import torch.nn as nn
import torchvision.models as models


class RenNet101_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet101(pretrained=False)
        self.head = nn.Linear(1000, 196)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.head(x)
        return x