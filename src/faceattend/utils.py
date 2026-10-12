"""Configuration, reproducibility and strict GPU guards."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml


def project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Open the FaceAttend project root before running notebooks.")


def load_config(root, experiment):
    config = yaml.safe_load((root / "configs/common.yaml").read_text())
    config.update(yaml.safe_load((root / f"configs/{experiment}.yaml").read_text()))
    return config


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-ONLY TRAINING: CUDA unavailable. Fix driver, CUDA PyTorch wheel and notebook kernel; CPU fallback is forbidden.")
    device = torch.device("cuda:0")
    x = torch.randn(128, 128, device=device)
    y = x @ x
    assert y.is_cuda
    torch.cuda.synchronize()
    return device


def environment():
    device = require_gpu()
    return {"python": sys.version, "executable": sys.executable,
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "cuda_build": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(device),
            "vram_mb": torch.cuda.get_device_properties(device).total_memory / 2**20,
            "device": str(device)}


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
