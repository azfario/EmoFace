"""Write the Phase 1 readout only from completed, independently checked results."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "outputs/comparison"
validation = json.loads((COMPARISON / "validation_report.json").read_text())
frozen = json.loads((COMPARISON / "frozen_selection.json").read_text())
table = pd.read_csv(COMPARISON / "comparison.csv")
winner = frozen["winner"]
friendly = {"m0_pretrained_baseline": "M0 pretrained baseline", "m1_pretrained_augmented": "M1 pretrained + augmentation"}
rows = []
timing = []
weak_classes = []
for row in table.itertuples():
    history = pd.read_csv(ROOT / "outputs" / row.model / "history.csv")
    rows.append(f"| {friendly[row.model]} | {row.seed} | {len(history)} | {row.best_epoch} | {row.val_accuracy:.2%} | {row.val_macro_f1:.4f} | {row.test_accuracy:.2%} | {row.test_macro_f1:.4f} | {row.inference_latency_ms:.2f} | {row.peak_gpu_memory_mb:.2f} |")
    timing.append(f"- {friendly[row.model]}: {history.epoch_seconds.sum()/60:.2f} minutes of recorded epoch computation (training + validation); {int(history.optimizer_steps.sum())} optimizer updates; {int(history.amp_skipped_steps.sum())} AMP-skipped batches.")
    receipt = json.loads((COMPARISON / f"{row.model}_test_result.json").read_text())
    per_class = receipt["metrics"]["per_class"]
    categories = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    weakest = sorted(categories, key=lambda name: per_class[name]["f1-score"])[:3]
    details = "; ".join(f"{name}: F1 {per_class[name]['f1-score']:.4f}, support {int(per_class[name]['support'])}" for name in weakest)
    weak_classes.append(f"- {friendly[row.model]} — lowest held-out per-class F1 values: {details}.")
indexed = table.set_index("model")
delta = indexed.loc["m1_pretrained_augmented", "val_macro_f1"] - indexed.loc["m0_pretrained_baseline", "val_macro_f1"]
relative = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
body = f'''# Phase 1 results

**Selected model: {friendly[winner]}**, chosen by validation macro F1, with accuracy and lower loss as tie-breakers. Both ImageNet-pretrained MobileNetV3-Small experiments completed on the local NVIDIA GPU. The decision was saved in `outputs/comparison/frozen_selection.json` before held-out test inference.

M1's selected validation macro F1 was **{abs(delta):.4f} {relative}** than M0's (seed 42). This is a single controlled run per variant; it does not establish statistical significance or guarantee that augmentation helps on other data. The test results below did not change the frozen winner.

| Experiment | Seed | Epochs completed | Best epoch | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | GPU forward median ms | Peak training GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Source and method

The source archive has 15,339 aligned 100×100 images. Seed-42 stratification split source train into **10,430 train / 1,841 validation** images; **3,068 source-test** images stayed held out. Train/validation/test have zero path, filename and exact-file-hash overlap. Preserve the mirror/provenance qualifications in [DATASET_AUDIT.md](DATASET_AUDIT.md) and [RAFDB_ALIGNMENT.md](RAFDB_ALIGNMENT.md).

Both experiments use pretrained IMAGENET1K_V1 MobileNetV3-Small, seven logits, full-network fine-tuning, RGB direct resize to 224×224, ImageNet normalization, batch 32, AdamW (lr 1e-4, weight decay 1e-4), cosine scheduling, maximum 30 epochs and patience 8 on strict validation macro-F1 improvement. AMP initial scale is 1024 and DataLoader workers are zero. M1 adds ±10° rotation, up to 3% affine translation and modest brightness/contrast jitter; evaluation transforms stay identical. No initial oversampling or class weighting.

The code asserts CUDA model parameters and batches for training and validation. Actual GPU/kernel/version metadata is preserved in each `config_resolved.json` and checkpoint. VS Code workspace settings and extensions are provided, but **selection of the notebook kernel in the VS Code GUI was not observed**. Actual notebook runs used the verified project Jupyter kernel from the command line.

## Per-class limitations

{chr(10).join(weak_classes)}

These categories contribute equally to macro F1 despite differing support. Minority-class estimates, particularly those with small test counts, are more uncertain. The full precision/recall/F1/support tables and confusion matrices are saved per model. These diagnostics were read after the validation choice was frozen and did not trigger model tuning.

## Timing and inference benchmark

{chr(10).join(timing)}

GPU forward benchmarks use batch 1, FP32, 20 warm-up forwards and 100 synchronized repetitions. The comparison CSV includes median and p95 latency and excludes image preprocessing/host transfer; this is not end-to-end application latency. Peak memory is PyTorch allocated memory, not total driver-reserved memory reported by `nvidia-smi`.

## Validation and export

Independent confusion-matrix recomputation verified accuracy, seven-class macro F1 and validation/test support totals. Best-epoch checks match saved history, manifest fingerprints match training, all epoch histories contain completed optimizer updates, and exported tensors exactly match the selected checkpoint. Held-out receipts were read for QA without rerunning test inference. Evidence: `outputs/comparison/validation_report.json`.

The portable checkpoint is `models/faceattend_best.pt`. It carries CPU state tensors, architecture, class order, RGB/direct-resize preprocessing, normalization, best epoch, seed and resolved config. Notebook 05 checks GPU checkpoint reconstruction without downloading initial weights and prints seven uncalibrated scores on one user-supplied training image. This input tests loading/shape/consistency and provides no additional generalization evidence.

## Limits

This is one seed per model. Original per-image source annotations/checksums remain unauthenticated; the accepted numeric mapping is supported by canonical research and exact published-count/filename agreement. Three exact duplicate pairs exist within source partitions, and four cross-source approximate-hash pairs remain candidates for review. Identity-level leakage cannot be ruled out without source identity annotations. Predictions concern visible annotation categories and uncalibrated scores; they do not measure internal feelings and must not be used for student/employee assessment. No identification, attendance or deployment phase was implemented.

## Inspectable evidence

- `notebooks/02_M0_Pretrained_Baseline_Training.ipynb` and `03_M1_Pretrained_Augmented_Training.ipynb`: executed GPU epoch loops and authentic curves/metrics.
- `notebooks/04_Model_Comparison_and_Export.ipynb`: validation curves, frozen decision, test metrics and export.
- Per-run `history.csv`, `config_resolved.json`, `best.pt`, `last.pt`, validation per-class reports and confusion matrices.
- `outputs/comparison/comparison.csv`, `validation_curves.png`, per-model held-out receipts/per-class reports/confusion matrices and export receipt.

Reproduce the metric reconciliation with `.venv/Scripts/python.exe scripts/validate_results.py`; this reads cached held-out evidence and does not repeat test inference.
'''
(ROOT / "docs/RESULTS.md").write_text(body, encoding="utf-8")
print(f"Wrote docs/RESULTS.md from authentic results; selected {winner}.")
