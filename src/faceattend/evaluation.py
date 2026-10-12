"""Freeze validation selection before touching held-out test data."""
import json
import time
from pathlib import Path

import pandas as pd
import torch

from . import CLASS_NAMES
from .data import load_manifests, make_loaders
from .engine import atomic_save, benchmark_gpu, validate_one_epoch
from .inference import load_model
from .metrics import save_metrics
from .utils import file_hash, write_json


EXPERIMENTS = ["m0_pretrained_baseline", "m1_pretrained_augmented"]


def freeze_selection(root):
    rows = []
    configs = []
    for experiment in EXPERIMENTS:
        folder = root / "outputs" / experiment
        if not (folder / "completed.json").exists():
            raise RuntimeError(f"Full training not completed: {experiment}; do not evaluate test yet.")
        payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        config = payload["config"]
        _, fingerprints = load_manifests(root, config)
        if fingerprints != config["manifest_fingerprints"]:
            raise ValueError("Dataset changed since training.")
        configs.append(config)
        metric = payload["validation_metrics"]
        rows.append({"model": experiment, "seed": payload["seed"], "best_epoch": payload["epoch"],
                     "val_accuracy": metric["accuracy"], "val_macro_f1": metric["macro_f1"], "val_loss": metric["loss"],
                     "checkpoint_sha256": file_hash(folder / "best.pt")})
    controlled = ["seed", "epochs", "batch_size", "lr", "weight_decay", "patience", "amp", "amp_initial_scale", "input_size", "weights", "horizontal_flip", "manifest_fingerprints"]
    for field in controlled:
        if configs[0][field] != configs[1][field]:
            raise ValueError(f"Experiment comparability mismatch: {field}")
    rows.sort(key=lambda r: (r["val_macro_f1"], r["val_accuracy"], -r["val_loss"]), reverse=True)
    frozen = {"winner": rows[0]["model"], "selection": "validation macro F1, then accuracy, then lower loss", "models": rows}
    path = root / "outputs/comparison/frozen_selection.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise RuntimeError("Frozen decision changed. Held-out test must not drive a new selection.")
    else:
        write_json(path, frozen)
    return frozen


def evaluate_and_export(root, device):
    frozen = freeze_selection(root)
    destination = root / "outputs/comparison"
    rows = []
    for row in frozen["models"]:
        folder = root / "outputs" / row["model"]
        model, _, payload = load_model(folder / "best.pt", device)
        cache = destination / f"{row['model']}_test_result.json"
        if cache.exists():
            result = json.loads(cache.read_text())
            if result["checkpoint_sha256"] != row["checkpoint_sha256"]:
                raise RuntimeError("Held-out result belongs to a different checkpoint.")
        else:
            marker = destination / f"{row['model']}_test_started.json"
            if marker.exists():
                raise RuntimeError("Prior test attempt was interrupted. Inspect marker and document recovery before explicitly rerunning.")
            loaders = make_loaders(root, payload["config"], include_test=True)
            write_json(marker, {"checkpoint_sha256": row["checkpoint_sha256"], "started": time.time()})
            metrics = validate_one_epoch(model, loaders["test"], torch.nn.CrossEntropyLoss(), device, amp=False)
            result = {"checkpoint_sha256": row["checkpoint_sha256"], "metrics": metrics}
            write_json(cache, result)
        save_metrics(result["metrics"], destination, row["model"] + "_test")
        latency = benchmark_gpu(model, device)
        full_history = pd.read_csv(folder / "history.csv")
        rows.append({**row, "test_accuracy": result["metrics"]["accuracy"], "test_macro_f1": result["metrics"]["macro_f1"],
                     "parameters": sum(p.numel() for p in model.parameters()), **latency,
                     "gpu_name": payload["config"]["environment"]["gpu_name"],
                     "peak_gpu_memory_mb": float(full_history.peak_gpu_memory_mb.max())})
    winner = torch.load(root / "outputs" / frozen["winner"] / "best.pt", map_location="cpu", weights_only=True)
    exported = {key: winner[key] for key in ["architecture", "experiment", "state_dict", "class_names", "input_size", "preprocessing", "normalization", "best_epoch", "validation_macro_f1", "seed", "config"]}
    atomic_save(root / "models/faceattend_best.pt", exported)
    table = pd.DataFrame(rows)
    table.to_csv(destination / "comparison.csv", index=False)
    write_json(destination / "export_receipt.json", {"winner": frozen["winner"], "sha256": file_hash(root / "models/faceattend_best.pt")})
    return table
