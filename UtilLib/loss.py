import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, label_path= 'dataset/label_map.txt', weight=0.8, num_classes=196):
        super(SoftCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        with open(label_path, 'r') as f:
            idBrand = {}
            for line in f:
                id, brand= line.split(' ')[:2]
                idBrand[int(id)-1] = brand
            corr_map = []
            for id in range(0, num_classes):
                arr = [0]*num_classes
                for other in range(0, num_classes):
                    arr[other] = int(idBrand[id]==idBrand[other])
                corr_map.append(arr)
        self.register_buffer("corr_map", torch.from_numpy(np.array(corr_map))) #

    def forward(self, input, target):
        input = input.cpu()
        target = target.cpu()
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1) # N*H*W, 1
        
        # 对该id进行 one_hot
        hard_target = torch.full((input.size()[0], self.num_classes), 0.).scatter_(1, torch.LongTensor(target), 1.) #  N*H*W,C
        # 选择该id的相同品牌的其他id
        soft_target = torch.index_select(self.corr_map.to(target.device), 0, torch.LongTensor(target.flatten())) # N*H*W,C

        assert soft_target.shape ==  hard_target.shape
        last_target = hard_target*(2*self.weight-1) + soft_target*(1.0-self.weight)
        loss = torch.sum(-last_target * F.log_softmax(input, dim=-1), dim=-1)
        return loss.mean()

if __name__ == '__main__':
    loss_fn = SoftCrossEntropyLoss()
    x = torch.randn((2,196))
    target = torch.tensor(
        [1]
    )
    print(loss_fn(x[0].view(1,196), target))
    print(loss_fn(x[1].view(1,196), target))