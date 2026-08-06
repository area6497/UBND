
import torch
from torch import nn
from .common import ConvBNAct, SqueezeExcitation

class DualIB(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5), use_se=True, activation='relu'):
        super().__init__()
        if len(kernel_sizes) != 2:
            raise ValueError('DualIB requires two deep convolutional branches.')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = in_channels == out_channels
        self.branch_small = ConvBNAct(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_sizes[0], stride=1, groups=in_channels, activation=activation)
        self.branch_large = ConvBNAct(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_sizes[1], stride=1, groups=in_channels, activation=activation)
        fused_channels = in_channels * 2
        self.channel_attention = SqueezeExcitation(fused_channels) if use_se else nn.Identity()
        self.channel_projection = ConvBNAct(in_channels=fused_channels, out_channels=out_channels, kernel_size=1, stride=1, activation='none')
        self.shortcut_projection = nn.Identity() if self.use_residual else ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, activation='none')
        self.out_activation = nn.ReLU(inplace=True)

    def forward(self, x):

        local_feature = self.branch_small(x)
        context_feature = self.branch_large(x)
        fused = torch.cat([local_feature, context_feature], dim=1)
        fused = self.channel_attention(fused)
        fused = self.channel_projection(fused)
        shortcut = self.shortcut_projection(x)
        return self.out_activation(fused + shortcut)
