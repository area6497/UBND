
import torch
from torch import nn

class ConvBNAct(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation='relu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = build_activation(activation)

    def forward(self, x):
 
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super().__init__()
        self.depthwise = ConvBNAct(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, activation=activation)
        self.pointwise = ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, activation=activation)

    def forward(self, x):

        return self.pointwise(self.depthwise(x))

class SqueezeExcitation(nn.Module):


    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden_channels, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):

        weight = self.fc(self.pool(x))
        return x * weight

class InvertedBottleneck(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion=2, activation='relu'):
        super().__init__()
        hidden_channels = int(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(ConvBNAct(in_channels, hidden_channels, kernel_size=1, activation=activation), ConvBNAct(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, groups=hidden_channels, activation=activation), ConvBNAct(hidden_channels, out_channels, kernel_size=1, activation='none'))
        self.out_act = build_activation(activation)

    def forward(self, x):

        out = self.block(x)
        if self.use_residual:
            out = out + x
        return self.out_act(out)

class ConvNeXtLiteBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, mlp_ratio=4):
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1, activation='none') if in_channels != out_channels or stride != 1 else nn.Identity()
        hidden_channels = out_channels * mlp_ratio
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=out_channels, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.mlp = nn.Sequential(nn.Conv2d(out_channels, hidden_channels, kernel_size=1), nn.GELU(), nn.Conv2d(hidden_channels, out_channels, kernel_size=1))

    def forward(self, x):

        identity = self.proj(x)
        out = self.depthwise(identity)
        out = self.norm(out)
        out = self.mlp(out)
        return out + identity

class IRMLP(nn.Module):


    def __init__(self, channels, expansion=4):
        super().__init__()
        hidden_channels = channels * expansion
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x):

        residual = x
        out = self.bn(self.dwconv(x)) + residual
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        return out + residual

def build_activation(name):

    normalized = name.lower()
    if normalized == 'relu':
        return nn.ReLU(inplace=True)
    if normalized == 'gelu':
        return nn.GELU()
    if normalized in {'silu', 'swish'}:
        return nn.SiLU(inplace=True)
    if normalized in {'none', 'identity', 'linear'}:
        return nn.Identity()
    raise ValueError(f'不支持的激活函数: {name}')
