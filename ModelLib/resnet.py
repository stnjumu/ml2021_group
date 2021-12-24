import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # M/2, N/2
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 通道数*4；
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=196, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 第一个layer的plane=64，其第一个block将输入的通道数变为64, 中间的block不改变通道数，最后一个block变为64*4
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 第二个layer的plane=128，其第一个block把输入的64*4通道变为128，中间blocks不变，最后一个变为128*4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 第三个layer通道数：128*4 -> 256 -> 256*4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 第四个layer通道数：256*4 -> 512 -> 512*4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=2)

        self.dropout = nn.Dropout2d(p=0.5,inplace=True)

        #print "block.expansion=",block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个block把输入通道数*4;
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 之后每个block的通道数都不变
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x) # input_channel->64, stride=2, M/2, N/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # stride=2, M/2, N/2

        x = self.layer1(x) # channels: 64 -> 64*4
        x = self.layer2(x) # channels: 64*4 -> 128*4, M/2, N/2
        x = self.layer3(x) # channels: 128*4 -> 256*4, M/2, N/2
        x = self.layer4(x) # channels: 256*4 -> 512*4, M/2, N/2

        x = self.avgpool(x)
        x = self.dropout(x)
        #print "avepool: ",x.data.shape
        x = x.view(x.size(0), -1) # x.size(0)=batch_size
        #print "view: ",x.data.shape
        x = self.fc(x)

        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('resnet50.tar'))
    return model

if __name__ == '__main__':
    model = resnet50(True)
