import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet20_cifar10', 'resnet32_cifar10', 'resnet44_cifar10', 'resnet56_cifar10']

NUM_CLASSES = 10


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.stride = stride
        if inplanes != self.expansion * planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes*self.expansion, stride),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x) if hasattr(self, 'downsample') else x

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes*self.expansion, stride),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x) if hasattr(self, 'downsample') else x

        out += identity
        out = self.relu(out)

        return out


class ResNetCifar(nn.Module):

    def __init__(self, block, num_blocks, num_classes=1000, zero_init_residual=False):
        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.group1 = self._make_group(block, 16,  num_blocks[0], stride=1)
        self.group2 = self._make_group(block, 32, num_blocks[1], stride=2)
        self.group3 = self._make_group(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_group(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_cifar10(**kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar10(**kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar10(**kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar10(**kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model