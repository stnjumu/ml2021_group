import timm

import torch
import torch.nn as nn
import torchvision.models as models
import timm

class EffNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=196)
    
    def forward(self, x):
        x = self.backbone(x)
    
        return x