import torch
from faceattend import CLASS_NAMES, MEAN, STD
from faceattend.engine import atomic_save
from faceattend.inference import load_model
from faceattend.models import build_model


def test_portable_round_trip(tmp_path):
    model = build_model(pretrained=False).eval()
    checkpoint = tmp_path / "model.pt"
    atomic_save(checkpoint, {"architecture": "mobilenet_v3_small", "class_names": CLASS_NAMES,
                          "input_size": [224, 224], "normalization": {"mean": MEAN, "std": STD}, "state_dict": model.state_dict()})
    loaded, _, _ = load_model(checkpoint, torch.device("cpu"))
    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        assert torch.equal(model(image), loaded(image))
