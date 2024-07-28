import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
import numpy as np
from torch.nn.modules.utils import _pair
from einops import rearrange
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,strid=1,norm=nn.GroupNorm):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=strid),
            norm(4,out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.single_conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x
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


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SubAttention(nn.Module):
    def __init__(self,in_channels=128,out_channels=32,ratio=1,bias=False):
        super(SubAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qk = nn.Conv2d(in_channels=in_channels,out_channels=out_channels*2,
                            kernel_size=ratio,stride=ratio,bias=bias)
        self.v = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                           kernel_size=1,stride=1)
    def forward(self,x):# BCHW
        qk = self.qk(x)
        v = self.v(x)
        q = qk[:,self.out_channels:,::]
        k = qk[:,:self.out_channels,::]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=1)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out

class MultiScaleAttention(nn.Module):
    def __init__(self, dim=128,bias=False,scales=[1,2,4,8]):
        super(MultiScaleAttention, self).__init__()
        self.num_heads = 4
        self.subatt0 = SubAttention(in_channels=dim,out_channels=dim//4,ratio=scales[0],bias=bias)
        self.subatt1 = SubAttention(in_channels=dim, out_channels=dim // 4, ratio=scales[1], bias=bias)
        self.subatt2 = SubAttention(in_channels=dim, out_channels=dim // 4, ratio=scales[2], bias=bias)
        self.subatt3 = SubAttention(in_channels=dim, out_channels=dim // 4, ratio=scales[3], bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape
        x0 = self.subatt0(x)
        x1 = self.subatt1(x)
        x2 = self.subatt2(x)
        x3 = self.subatt3(x)
        out = torch.cat([x0,x1,x2,x3],dim=1)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


