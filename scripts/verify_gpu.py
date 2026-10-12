"""Run one pretrained CUDA forward/backward update on synthetic tensors only.

This is a computation test, not an experiment or accuracy claim. No RAF-DB data
is consumed. Pretrained weights download into the local project cache.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

import torch
from faceattend.models import build_model
from faceattend.engine import assert_gpu, train_one_epoch
from faceattend.utils import environment, require_gpu, seed_everything, write_json

seed_everything(42)
device = require_gpu()
model = build_model(pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
before = model.classifier[-1].weight.detach().clone()
images = torch.randn(2, 3, 224, 224)
targets = torch.tensor([0, 1])
scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
metrics = train_one_epoch(model, [(images, targets)], torch.nn.CrossEntropyLoss(), optimizer, scaler, device)
assert_gpu(model)
assert not torch.equal(before, model.classifier[-1].weight)
report = environment()
report.update({"pretrained_gpu_forward_backward": "PASSED", "smoke_batch_size": 2,
               "amp": True, "initial_amp_scale": 1024., "optimizer_steps": metrics["optimizer_steps"],
               "amp_skipped_steps": metrics["amp_skipped_steps"],
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
               "scope": "Synthetic tensor computation test only; no dataset accuracy or full training."})
write_json(ROOT / "outputs/gpu_smoke_test.json", report)
print(report)
