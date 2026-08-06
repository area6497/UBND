
import torch
from torch import nn
from .backbone import BackboneOutput, UBNDStudentBackbone
from .heads import ClassificationHead

class NetworkOutput:

    def __init__(self, logits, probabilities, features, intermediate=None):
        self.logits = logits
        self.probabilities = probabilities
        self.features = features
        self.intermediate = intermediate
    'Network forward output.\n\n    Attributes:\n        logits: Classification logits。\n        probabilities: softmax probabilities.\n        features: Final fused feature map.\n        intermediate: Optional intermediate feature output.\n    '

class UBNDStudent(nn.Module):


    def __init__(self, num_classes=2, in_channels=3, feature_channels=256, dropout=0.0):
        super().__init__()
        self.backbone = UBNDStudentBackbone(in_channels=in_channels, out_channels=feature_channels)
        self.head = ClassificationHead(in_channels=feature_channels, num_classes=num_classes, dropout=dropout)

    def forward(self, x, return_features=False, return_intermediate=False):

        if return_intermediate:
            backbone_output = self.backbone(x, return_intermediate=True)
            features = backbone_output.features
            intermediate = backbone_output
        else:
            features = self.backbone(x)
            intermediate = None
        logits = self.head(features)
        if not return_features:
            return logits
        probabilities = torch.softmax(logits, dim=1)
        return NetworkOutput(logits=logits, probabilities=probabilities, features=features, intermediate=intermediate)

    def extract_features(self, x, return_intermediate=False):

        return self.backbone(x, return_intermediate=return_intermediate)

    def predict_proba(self, x):

        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
