import timm

import torch
import torch.nn as nn
import torchvision.models as models
import timm

class CaiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('cait_s24_384', pretrained=True, num_classes=196)
    
    def forward(self, x):
        x = self.backbone(x)
    
        return x