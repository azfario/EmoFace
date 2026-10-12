import pytest
import torch
from faceattend.utils import require_gpu
from faceattend.engine import train_one_epoch
from faceattend.models import build_model


def test_guard_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU-ONLY"):
        require_gpu()


def test_cpu_training_forbidden():
    with pytest.raises(RuntimeError, match="CPU epoch"):
        train_one_epoch(build_model(False), [], torch.nn.CrossEntropyLoss(), None, None, torch.device("cpu"))
