'''ResNet in PyTorch.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    """
    Pre-activation version of the BasicBlock.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, 
                          stride=stride, bias=False))

    def forward(self, inp):
        x = inp

        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)

        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += shortcut

        return out


class InputNormalize(nn.Module):
    """
    Normalizes inputs according to (x - mean) / std
    """
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


class ResNet(nn.Module):
    """
    PreActResNet
    """
    def __init__(self, block, num_blocks, num_classes=10, 
                 need_normalize=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])
        self.normalize = InputNormalize(self.mean, self.std)
        self.need_normalize = True

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def set_normalize(self, need_normalize):
        """
        Whether to normalize inputs or not.
        You should leave this as True if your adversarial inputs are within
        [0, 1]. If your attack generates perturbations starting from inputs
        that have already been normalized using the CIFAR-10 mean and std,
        then set this to False.
        """
        self.need_normalize = need_normalize

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inp, lin=0, lout=5, 
                with_latent=False, fake_relu=False, no_relu=False):
        x = inp
        if self.need_normalize:
            x = self.normalize(x)
        out = x

        if lin < 1 and lout > -1:
            out = F.relu(self.bn1(self.conv1(out)))

        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        return out

def ResNet18():
    return ResNet(PreActBlock, [2,2,2,2])
