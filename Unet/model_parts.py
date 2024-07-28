import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
import numpy as np
from torch.nn.modules.utils import _pair
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,strid=1,norm=nn.GroupNorm):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,stride=strid),
            norm(4,mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm(4,out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,strid=1,norm=nn.GroupNorm):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm(4,out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.single_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpBlock,self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])  # 左右上下
        else:
            x2 = x1
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(DownBlock,self).__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,strid=2)
        )

    def forward(self, x):
        res = self.maxpool_conv(x)
        return res,x


class Head(nn.Module):
    def __init__(self,in_channels,out_channels,strid=1,type='head'):
        super(Head, self).__init__()
        self.type = type
        if type == 'head':
            self.ccblock = DoubleConv(in_channels,out_channels)#CrossBlock(in_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels, out_channels // 2)
            self.ccblock = DoubleConv(in_channels,out_channels//2)#CrossBlock(in_channels, out_channels//2)

    def forward(self,x):
        if self.type == 'head':
            out = self.ccblock(x)
        else:
            out = torch.cat([self.conv(x),self.ccblock(x)],dim=1)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x
