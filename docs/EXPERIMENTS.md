> For another PC, follow [the source-checkout setup guide](SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Controlled experiment protocol

## Question and fixed settings

Does conservative training augmentation improve validation macro F1 for ImageNet-pretrained MobileNetV3-Small on this audited RAF-DB copy?

Both runs use `IMAGENET1K_V1`, the same seed-42 initialization, seven-logit classifier head, identical manifests, RGB direct bilinear resize to 224×224, ImageNet mean/std, full-network fine-tuning, cross entropy, AdamW (`lr=1e-4`, `weight_decay=1e-4`), cosine annealing, 30-epoch maximum, patience 8 on strict validation macro-F1 improvement, initial batch 32, CUDA AMP and zero DataLoader workers. No initial class weighting or oversampling. All training and validation computation uses the local CUDA GPU.

M0 has no random augmentation. M1 adds affine rotation ±10°, translation up to 3% horizontally/vertically and brightness/contrast factors 0.9–1.1. Both have horizontal flip disabled to keep M0 minimal. Evaluation transforms are identical and deterministic. No vertical flip, aggressive crop, face-obscuring transformation or augmentation tuning on test.

Seed Python, NumPy, torch and CUDA; set deterministic algorithms, deterministic cuDNN and CUBLAS workspace configuration. This reduces repeatability risks but does not guarantee equality across hardware, package versions or floating-point implementations. Store environment, config and manifest fingerprints with every run. AMP initial scale is 1024 for both experiments: the synthetic GPU smoke check at the default higher scale skipped its first update, while scale 1024 passed a real optimizer update. Log completed optimizer steps and AMP-skipped steps, and stop if an epoch completes no updates.

## Selection and evaluation

Select `best.pt` by validation macro F1, then validation accuracy, then lower validation loss. Early stopping resets patience only on a strict macro-F1 increase. Train both experiments to their configured stopping condition before comparison. Freeze configurations/checkpoint hashes and choose a winner from validation only.

Evaluate held-out test once per frozen checkpoint. Cache results with checkpoint hashes; never use test metrics to change the winner. An interrupted evaluation leaves a marker; explicitly document a recovery decision before repeating it. Benchmarks use FP32 batch-1 GPU forwards, 20 warm-ups and 100 synchronized repetitions, median and p95. Latency excludes preprocessing and host transfer.

## Saved evidence and recovery

Per epoch: losses, accuracies, validation macro F1, learning rate, duration and peak GPU allocation. Save best and last checkpoints, with optimizer/scheduler/scaler, Python/NumPy/torch/CUDA/DataLoader RNG state and resolved configuration. `history.csv` survives notebook output clearing. On resume, checkpoint history is authoritative, configuration must match and the interrupted epoch repeats.

OOM stops training. Record the cause and any lower batch size before an explicitly restarted run. Preserve failed-run evidence; never silently retry on CPU. If only one model uses a different batch size/AMP setup, the default comparison guard rejects it; revise and document the experiment design explicitly before continuing.

No results are promised. A single-seed comparison cannot characterize statistical uncertainty. Macro F1 and per-class precision/recall/F1 expose minority-class weaknesses. Visual expression annotation may be ambiguous and does not measure true emotions.

## Inference export

The winner's portable CPU state dictionary is accompanied by architecture, class order, exact RGB/direct-resize preprocessing, mean/std, best epoch, validation macro F1, seed and serializable resolved config. Reconstruct with `pretrained=False` and load saved weights; this is inference reconstruction, not training from scratch. Export smoke testing uses one image from the user-supplied training partition by default, with an optional authorized local-image override and CUDA inference. This is a loading/shape/consistency check, not a generalization measurement.

## Figure and result validation

Final charts use the notebook/static Matplotlib surface specified by the handoff. Epoch curves plot real disk history at one completed-epoch grain; training/validation lines compare the same metric, and the validation-only macro-F1 panel uses all seven classes. Confusion matrices show raw counts with explicit annotated/predicted class axes and per-class support in the adjacent CSV/JSON. The comparison notebook overlays M0/M1 validation curves on shared metric axes. Use descriptive titles, labeled units/axes, legends and readable exported labels; inspect the final PNGs and notebook outputs.

`scripts/validate_results.py` independently recomputes accuracy and seven-class macro F1 from saved confusion matrices, checks validation/test supports, verifies best-epoch selection against history, confirms contiguous epoch records and completed optimizer updates, reconciles manifest fingerprints, validates the frozen validation winner, and confirms exported tensors exactly match the selected checkpoint. It reads held-out receipts without repeating test inference. A successful check still requires the single-seed, mirror-provenance, approximate-duplicate and identity-leakage caveats in the readout.
