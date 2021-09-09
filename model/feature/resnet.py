"""
残差网络做为特征提取器
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  1月 22, 2021
"""
import torch
import torch.nn as nn
from typing import List


def con_1x1(in_planes: int,
            places: int,
            stride: int = 2):
    """
    :param in_planes:
    :param places:
    :param stride:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class Bottleneck(nn.Module):
    def __init__(self, in_places: int,
                 places: int,
                 stride: int = 1,
                 is_down_sampling: bool = False,
                 expansion: int = 4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.is_down_sampling = is_down_sampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.is_down_sampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bottleneck(x)
        if self.is_down_sampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks: List[int],
                 expansion: int = 4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.con_1 = con_1x1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places: int,
                   places: int,
                   block: int,
                   stride: int) -> nn.Module:
        layer_s = [Bottleneck(in_places, places, stride, is_down_sampling=True)]
        for i in range(1, block):
            layer_s.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layer_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.con_1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


def res_net_50() -> nn.Module:
    return ResNet([3, 4, 6, 3])


def res_net_101() -> nn.Module:
    return ResNet([3, 4, 23, 3])


def res_net_152() -> nn.Module:
    return ResNet([3, 8, 36, 3])
