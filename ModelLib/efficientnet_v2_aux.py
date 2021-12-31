import timm

import torch
import torch.nn as nn
import torchvision.models as models
import timm

class EffNetV2_AUX(nn.Module):
    def __init__(self):
        super().__init__()
        # 品牌数49, 年份数16
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=49)
    
    def forward(self, x):
        x = self.backbone(x)
    
        return x