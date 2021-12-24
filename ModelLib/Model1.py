# 模型示例
# 未实现
import torch
import numpy as np

class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        
    def forward(self, x):
        return 1