
from pathlib import Path
import torch
from torch import nn
from torchvision import models

class DenseNet121Teacher(nn.Module):


    def __init__(self, num_classes=2, pretrained=True, checkpoint=None, freeze=True):
        super().__init__()
        self.model = self._build_densenet121(num_classes=num_classes, pretrained=pretrained)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        if freeze:
            self.freeze()

    def forward(self, x):

        return self.model(x)

    def freeze(self):

        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def load_checkpoint(self, checkpoint):

        state = torch.load(checkpoint, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        cleaned_state = {key.replace('module.', ''): value for key, value in state.items()}
        self.model.load_state_dict(cleaned_state, strict=False)

    @staticmethod
    def _build_densenet121(num_classes, pretrained):

        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)
        except AttributeError:
            model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
