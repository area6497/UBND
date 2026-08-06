
import torch
from torch import nn

class ClassificationHead(nn.Module):


    def __init__(self, in_channels=256, num_classes=2, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)
        self._initialize_weights()

    def forward(self, features):

        pooled = self.pool(features).flatten(1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)

    def predict_proba(self, features):

        logits = self.forward(features)
        return torch.softmax(logits, dim=1)

    def _initialize_weights(self):

        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)
