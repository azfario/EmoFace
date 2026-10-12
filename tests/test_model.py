import torch
from faceattend.models import build_model


def test_seven_logits_without_network_download():
    model = build_model(pretrained=False).eval()
    assert model.classifier[-1].out_features == 7
    with torch.inference_mode():
        result = model(torch.zeros(2, 3, 224, 224))
    assert result.shape == (2, 7) and torch.isfinite(result).all()
