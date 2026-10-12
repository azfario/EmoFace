"""Pretrained training factory; inference reconstruction never downloads weights."""
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_model(pretrained=True):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 7)
    return model
