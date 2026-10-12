"""Independently reconcile final reported metrics, selection and export evidence.

Reads saved confusion matrices rather than repeating held-out inference.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ["m0_pretrained_baseline", "m1_pretrained_augmented"]
report = {"assessment": "Share with caveats: single seed; mirror provenance and identity leakage limitations",
          "checks": [], "reran_held_out_inference": False}


def check_metrics(metrics, expected_support, label):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    assert matrix.shape == (7, 7) and (matrix >= 0).all()
    assert int(matrix.sum()) == expected_support, (label, matrix.sum(), expected_support)
    true_positive = matrix.diagonal().astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(7), where=denominator != 0)
    accuracy = float(true_positive.sum() / matrix.sum())
    macro_f1 = float(f1.mean())
    assert abs(accuracy - metrics["accuracy"]) < 1e-12, label
    assert abs(macro_f1 - metrics["macro_f1"]) < 1e-12, label
    supports = list(metrics["per_class"].values())[:7]
    assert [int(m["support"]) for m in supports] == matrix.sum(axis=1).tolist()
    report["checks"].append({"metric_set": label, "support": expected_support,
                             "recomputed_accuracy": accuracy, "recomputed_macro_f1": macro_f1})


manifests = {name: pd.read_csv(ROOT / "data/processed" / f"{name}.csv") for name in ["train", "val", "test"]}
fingerprints = json.loads((ROOT / "data/processed/manifest_fingerprints.json").read_text())
payloads = {}
for experiment in EXPERIMENTS:
    directory = ROOT / "outputs" / experiment
    assert (directory / "completed.json").exists(), f"Incomplete training: {experiment}"
    payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    payloads[experiment] = payload
    history = pd.read_csv(directory / "history.csv")
    expected_best = history.sort_values(["val_macro_f1", "val_acc", "val_loss"], ascending=[False, False, True]).iloc[0]
    assert int(expected_best.epoch) == payload["epoch"]
    assert abs(expected_best.val_macro_f1 - payload["validation_macro_f1"]) < 1e-12
    assert (history.optimizer_steps > 0).all()
    assert np.isfinite(history.select_dtypes(include="number")).all().all()
    assert history.epoch.tolist() == list(range(1, len(history) + 1))
    assert payload["config"]["manifest_fingerprints"] == fingerprints
    check_metrics(payload["validation_metrics"], len(manifests["val"]), experiment + " validation")
    receipt = json.loads((ROOT / "outputs/comparison" / f"{experiment}_test_result.json").read_text())
    check_metrics(receipt["metrics"], len(manifests["test"]), experiment + " held-out test")
    report["checks"].append({"experiment": experiment, "epochs": len(history),
                             "best_epoch": payload["epoch"], "amp_skipped_steps": int(history.amp_skipped_steps.sum()),
                             "total_optimizer_steps": int(history.optimizer_steps.sum())})

comparison = pd.read_csv(ROOT / "outputs/comparison/comparison.csv")
frozen = json.loads((ROOT / "outputs/comparison/frozen_selection.json").read_text())
ranked = comparison.sort_values(["val_macro_f1", "val_accuracy", "val_loss"], ascending=[False, False, True])
assert ranked.iloc[0].model == frozen["winner"]
exported = torch.load(ROOT / "models/faceattend_best.pt", map_location="cpu", weights_only=True)
assert exported["experiment"] == frozen["winner"]
for key, tensor in payloads[frozen["winner"]]["state_dict"].items():
    assert torch.equal(tensor, exported["state_dict"][key])
assert all(tensor.device.type == "cpu" for tensor in exported["state_dict"].values())
report["checks"].append({"frozen_winner": frozen["winner"], "export_weights_match": True, "export_device_portable": True})
(ROOT / "outputs/comparison/validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
