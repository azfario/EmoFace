import random
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from faceattend.engine import save_checkpoint, resume_checkpoint
from faceattend.models import build_model


def test_resumable_metadata_and_rng_round_trip(tmp_path):
    config = {"experiment": "toy_schema_test", "input_size": [224, 224], "seed": 42}
    model = build_model(False)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = DataLoader(TensorDataset(torch.zeros(2, 1)), generator=torch.Generator().manual_seed(42))
    loaders = {"train": loader}
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(checkpoint, model, config, 1, {"macro_f1": .25}, [{"epoch": 1}], (.25, .5, -.8), 0, optimizer, scheduler, scaler, loaders)
    expected = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=loader.generator))
    restored = resume_checkpoint(checkpoint, model, config, optimizer, scheduler, scaler, loaders)
    actual = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=loader.generator))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2]) and torch.equal(expected[3], actual[3])
    assert restored["epoch"] == 1 and restored["history"] == [{"epoch": 1}]
    with pytest.raises(ValueError, match="configuration differs"):
        resume_checkpoint(checkpoint, model, {**config, "seed": 0}, optimizer, scheduler, scaler, loaders)
