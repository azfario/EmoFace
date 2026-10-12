"""Portable checkpoint loading and uncalibrated visible-expression scores."""
from pathlib import Path

import torch
from PIL import Image

from . import CLASS_NAMES
from .data import make_transform
from .engine import assert_gpu
from .models import build_model
from .utils import require_gpu


def load_model(checkpoint, device=None):
    device = device or require_gpu()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["architecture"] != "mobilenet_v3_small" or payload["class_names"] != CLASS_NAMES:
        raise ValueError("Unsupported checkpoint architecture or class order.")
    model = build_model(pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    transform = make_transform(payload, training=False)
    return model, transform, payload


@torch.inference_mode()
def predict_image(checkpoint, image_path):
    device = require_gpu()
    model, transform, metadata = load_model(checkpoint, device)
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    assert_gpu(model, tensor)
    logits = model(tensor)
    assert logits.shape == (1, 7)
    scores = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return dict(zip(CLASS_NAMES, scores))
