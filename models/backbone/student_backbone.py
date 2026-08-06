
import torch
from torch import nn
from ..modules import ConvBNAct, ConvNeXtLiteBlock, DepthwiseSeparableConv, DualIB, HFFusion, InvertedBottleneck

class BackboneOutput:

    def __init__(self, features, convnext_features, high_features):
        self.features = features
        self.convnext_features = convnext_features
        self.high_features = high_features
    'Student network backbone output. \n\n    Attributes:\n        features: The final fused feature map is used for classification. \n        convnext_features: Output from the ConvNeXt stage, used for HF Fusion or visualization. \n        high_features: Dual-IB High-level features derived from the Dual-IB output.\n    '

class UBNDStudentBackbone(nn.Module):


    def __init__(self, in_channels=3, stem_channels=32, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.stem = ConvBNAct(in_channels=in_channels, out_channels=stem_channels, kernel_size=3, stride=2, activation='relu')
        self.fused_ib = InvertedBottleneck(in_channels=stem_channels, out_channels=64, kernel_size=3, stride=2, expansion=2, activation='relu')
        self.extra_dw_5x5 = DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=5, stride=2, activation='relu')
        self.ib_blocks = nn.Sequential(InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, expansion=2, activation='relu'), InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, expansion=2, activation='relu'))
        self.convnext = ConvNeXtLiteBlock(in_channels=128, out_channels=384, kernel_size=3, stride=1)
        self.extra_dw_3x3 = DepthwiseSeparableConv(in_channels=384, out_channels=out_channels, kernel_size=3, stride=2, activation='relu')
        self.dual_ib = DualIB(in_channels=out_channels, out_channels=out_channels, kernel_sizes=(3, 5), use_se=True, activation='relu')
        self.hf_fusion = HFFusion(low_channels=384, high_channels=out_channels, out_channels=out_channels, use_irmlp=True)
        self._initialize_weights()

    def forward(self, x, return_intermediate=False):

        x = self.stem(x)
        x = self.fused_ib(x)
        x = self.extra_dw_5x5(x)
        x = self.ib_blocks(x)
        convnext_features = self.convnext(x)
        high_features = self.extra_dw_3x3(convnext_features)
        high_features = self.dual_ib(high_features)
        features = self.hf_fusion(convnext_features, high_features)
        if return_intermediate:
            return BackboneOutput(features=features, convnext_features=convnext_features, high_features=high_features)
        return features

    @staticmethod
    def _initialize_module(module):

        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _initialize_weights(self):

        self.apply(self._initialize_module)
