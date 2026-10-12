"""GPU-only epoch helpers and restart-safe checkpoints."""
import contextlib
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from . import CLASS_NAMES, MEAN, STD
from .metrics import classification_metrics
from .utils import require_gpu


def assert_gpu(model, images=None, targets=None):
    require = next(model.parameters()).device
    if require.type != "cuda" or any(p.device != require for p in model.parameters()):
        raise RuntimeError("GPU-ONLY: model parameters must be on the same CUDA GPU.")
    for tensor in [images, targets]:
        if tensor is not None and tensor.device != require:
            raise RuntimeError("GPU-ONLY: every training tensor must be on the model's GPU.")


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=True):
    if device.type != "cuda":
        raise RuntimeError("GPU-ONLY: CPU epoch execution is forbidden.")
    assert_gpu(model)
    training = optimizer is not None
    model.train(training)
    loss_sum, targets_all, predicted_all = 0., [], []
    optimizer_steps, amp_skipped_steps = 0, 0
    context = contextlib.nullcontext() if training else torch.inference_mode()
    with context:
        for images, targets in tqdm(loader, desc="Train" if training else "Validate", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            assert_gpu(model, images, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; stop and investigate AMP/configuration.")
            if training:
                if scaler is not None and scaler.is_enabled():
                    old_scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < old_scale:
                        amp_skipped_steps += 1
                    else:
                        optimizer_steps += 1
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer_steps += 1
            loss_sum += loss.item() * len(targets)
            targets_all.extend(targets.detach().cpu().tolist())
            predicted_all.extend(logits.detach().argmax(1).cpu().tolist())
    if not targets_all:
        raise ValueError("Cannot run an empty data loader.")
    if training and optimizer_steps == 0:
        raise FloatingPointError("No optimizer updates completed. AMP skipped all batches; investigate loss scaling or explicitly disable AMP.")
    metrics = classification_metrics(targets_all, predicted_all)
    metrics["loss"] = loss_sum / len(targets_all)
    if training:
        metrics.update(optimizer_steps=optimizer_steps, amp_skipped_steps=amp_skipped_steps)
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp=True):
    return _epoch(model, loader, criterion, device, optimizer, scaler, amp)


def validate_one_epoch(model, loader, criterion, device, amp=True):
    return _epoch(model, loader, criterion, device, amp=amp)


def capture_rng(loaders):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "loaders": {name: loader.generator.get_state() for name, loader in loaders.items()}}


def restore_rng(state, loaders):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    for name, value in state["loaders"].items():
        loaders[name].generator.set_state(value.cpu())


def save_checkpoint(path, model, config, epoch, val_metrics, history, best_key,
                    stale, optimizer, scheduler, scaler, loaders):
    payload = {"architecture": "mobilenet_v3_small", "experiment": config["experiment"],
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "class_names": CLASS_NAMES, "input_size": config["input_size"],
               "preprocessing": "RGB; direct bilinear resize to 224x224; tensor; ImageNet normalization",
               "normalization": {"mean": MEAN, "std": STD},
               "epoch": epoch, "best_epoch": epoch, "validation_macro_f1": val_metrics["macro_f1"],
               "validation_metrics": val_metrics, "seed": config["seed"], "config": config,
               "history": history, "best_key": best_key, "stale": stale,
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict(), "rng": capture_rng(loaders)}
    atomic_save(path, payload)


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resume_checkpoint(path, model, config, optimizer, scheduler, scaler, loaders):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config"] != config:
        raise ValueError("Resume configuration differs from saved run. Restore it or use a new run directory.")
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"], loaders)
    return payload


@torch.inference_mode()
def benchmark_gpu(model, device, repeats=100, warmup=20):
    assert_gpu(model)
    sample = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    for _ in range(warmup):
        model(sample)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(sample)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    return {"inference_latency_ms": float(np.median(latencies)), "p95_latency_ms": float(np.percentile(latencies, 95)),
            "repeats": repeats, "warmup": warmup, "batch_size": 1, "precision": "fp32",
            "scope": "model forward only; preprocessing and host transfer excluded"}
