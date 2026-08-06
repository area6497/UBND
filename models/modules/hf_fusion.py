
import torch
from torch import nn
from torch.nn import functional as F
from .common import ConvBNAct, IRMLP, SqueezeExcitation

class SpatialAttention(nn.Module):


    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError('SpatialAttention kernel_size only supports 3 or 7')
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention

class HFFusion(nn.Module):


    def __init__(self, low_channels, high_channels, out_channels, use_irmlp=True):
        super().__init__()
        self.low_projection = ConvBNAct(in_channels=low_channels, out_channels=out_channels, kernel_size=1, activation='relu')
        self.high_projection = ConvBNAct(in_channels=high_channels, out_channels=out_channels, kernel_size=1, activation='relu')
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.channel_attention = SqueezeExcitation(out_channels)
        self.fusion = ConvBNAct(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, activation='relu')
        self.irmlp = IRMLP(out_channels) if use_irmlp else nn.Identity()

    def forward(self, low_feature, high_feature):

        low_feature = self.low_projection(low_feature)
        high_feature = self.high_projection(high_feature)
        if low_feature.shape[-2:] != high_feature.shape[-2:]:
            low_feature = F.interpolate(low_feature, size=high_feature.shape[-2:], mode='bilinear', align_corners=False)
        low_feature = self.spatial_attention(low_feature)
        high_feature = self.channel_attention(high_feature)
        fused = torch.cat([low_feature, high_feature], dim=1)
        fused = self.fusion(fused)
        return self.irmlp(fused)
