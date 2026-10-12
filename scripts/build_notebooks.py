"""Generate genuine nbformat v4 notebooks from reviewable cell templates."""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


GPU = '''
import sys
import torch
import torchvision
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("GPU-ONLY TRAINING: CUDA unavailable in this kernel. Fix NVIDIA driver, CUDA wheel, interpreter or notebook kernel. CPU training is forbidden.")
device = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU properties:", torch.cuda.get_device_properties(0))
x = torch.randn((128, 128), device=device)
y = x @ x
assert y.is_cuda and y.device.type == "cuda"
torch.cuda.synchronize()
print("GPU tensor test PASSED:", y.device)
del x, y
'''

SETUP = '''
from pathlib import Path
import os, sys
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()), None)
if ROOT is None:
    raise RuntimeError("Open the FaceAttend project folder.")
sys.path.insert(0, str(ROOT / "src"))
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from faceattend.utils import environment, load_config, seed_everything, write_json
'''


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (FaceAttend GPU)", "language": "python", "name": "faceattend-gpu"},
        "language_info": {"name": "python", "version": "3.12"}})
    nbf.validate(notebook)
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(notebook, ROOT / "notebooks" / name)


save("00_GPU_Environment_Check.ipynb", [
    md("""# FaceAttend GPU environment verification
    ## Objective
    Prove that this notebook kernel uses the local NVIDIA GPU. Select the project `.venv` in VS Code before running.
    The first executable cell hard fails if CUDA is inaccessible. CPU preprocessing and logic tests are allowed;
    training, smoke-training and inference benchmarking must use CUDA.
    A command-line notebook run proves that kernel's computation; it does not prove VS Code selected that kernel."""),
    code(GPU), md("## Record the verified environment\nSave machine/software metadata locally for subsequent experiment provenance."),
    code(SETUP), code('''
    report = environment()
    write_json(ROOT / "outputs/environment_check.json", report)
    display(pd.Series(report, name="Verified notebook environment"))
    '''),
    md("""## Interpretation and next step
    A passed tensor test demonstrates CUDA computation in the printed Python interpreter. Confirm it ends in
    `.venv/Scripts/python.exe`. Run this cell in VS Code itself before starting either training notebook.
    Driver compatibility and installed wheel versions must be retained with each run; `nvidia-smi` alone is insufficient.""")])

save("01_RAFDB_Dataset_Exploration.ipynb", [
    md("""# RAF-DB dataset exploration and audit
    ## Objective and assumptions
    Classify visible annotations: surprise, fear, disgust, happy, sad, angry, neutral. These labels do not reveal inner feelings.
    Sources: [selected Kaggle mirror](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) and
    [original RAF-DB](https://www.whdeng.cn/RAF/model1.html). The mirror states non-commercial research use and restricted redistribution.
    This notebook performs CPU data inspection, not training. Numeric mirror labels require a verified explicit legend;
    the handoff's original label map is initially provisional for this copy."""),
    md("## Setup and configurable inputs\nPrefer authorized files in `data/raw/rafdb`; inspect the existing archive without moving it when that folder is empty."),
    code(SETUP), code('''
    import yaml
    from faceattend.prepare_data import audit_dataset, make_manifests
    from faceattend import CLASS_NAMES, MEAN, STD
    config = yaml.safe_load((ROOT / "configs/common.yaml").read_text())
    RAW = ROOT / config["raw_root"]
    if not RAW.exists():
        RAW = ROOT / "archive (4)"
    mapping = json.loads((ROOT / "configs/label_map.json").read_text())
    print("Source:", RAW.relative_to(ROOT))
    print("Authorized access recorded:", config["authorized_data"])
    print("Numeric label legend verified:", config["label_mapping_verified"])
    '''),
    md("## Audit annotations, images, hashes and source partitions\nCSV/folder agreement checks consistency, but does not establish label semantics. The audit reports candidate difference-hash matches; they need human review."),
    code('''
    frame, report = audit_dataset(RAW, ROOT, mapping, config["partition_provenance"])
    report["authorized_data"] = config["authorized_data"]
    report["label_mapping_verified"] = config["label_mapping_verified"]
    write_json(ROOT / "outputs/dataset_audit.json", report)
    display(pd.Series({k: v for k, v in report.items() if k not in ["class_counts", "cross_partition_dhash_candidates", "sizes"]}))
    display(frame.groupby(["source_partition", "class_name"]).size().unstack(fill_value=0).reindex(columns=CLASS_NAMES))
    display(pd.DataFrame(report["sizes"]))
    print("Cross-partition near-duplicate candidates:", len(report["cross_partition_dhash_candidates"]))
    '''),
    md("## Provisional class distribution\nCounts use the configured mapping. Interpret class names only after validating the mirror's legend."),
    code('''
    distribution = frame.groupby(["class_name", "source_partition"]).size().unstack(fill_value=0).reindex(CLASS_NAMES)
    axis = distribution.plot.bar(figsize=(10, 4), title="Source class counts (configured label map)")
    axis.set(xlabel="Visible-expression category", ylabel="Images")
    plt.tight_layout()
    plt.show()
    '''),
    md("## Permission and label verification gate\nRecord authorized use and verified numeric semantics in `configs/common.yaml` before generating training manifests. Do not set these flags merely to bypass an error."),
    code('''
    if not config["authorized_data"]:
        raise RuntimeError("Authorized RAF-DB use has not been recorded; confirm permission before training.")
    if not config["label_mapping_verified"]:
        raise RuntimeError("Numeric Kaggle labels are provisional. Verify against original annotations or a trustworthy mirror legend.")
    splits = make_manifests(frame, ROOT / config["manifest_root"], seed=config["seed"])
    report["manifest_counts"] = {name: len(table) for name, table in splits.items()}
    write_json(ROOT / "outputs/dataset_audit.json", report)
    display(pd.DataFrame({name: table.class_name.value_counts() for name, table in splits.items()}).fillna(0).reindex(CLASS_NAMES))
    '''),
    md("## Validate preprocessing using training samples only\nThese local faces must not be published or retained in shared notebook outputs."),
    code('''
    from PIL import Image
    from faceattend.data import make_transform
    seed_everything(config["seed"])
    sample_path = ROOT / splits["train"].iloc[0].path
    image = Image.open(sample_path).convert("RGB")
    transform = make_transform(config, training=False)
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()
    print("RGB evaluation transform passed:", tensor.shape)
    # Optional private preview: remove output before sharing this notebook.
    SHOW_PRIVATE_SAMPLES = False
    if SHOW_PRIVATE_SAMPLES:
        display(image)
    '''),
    md("""## Interpretation
    Preserve source test membership. Train/validation is a seed-42 stratified 85/15 split of source train only.
    Counts matching the benchmark do not authenticate the mirror. Original annotation metadata is absent in this copy,
    identity-level leakage cannot be excluded, and near-duplicate candidates require review.
    Exact path/name/hash overlap causes split creation to stop; do not silently drop or relabel source data.""")])

def training_notebook(experiment, number, title, hypothesis):
    cells = [md(f"""# {title}
    ## Objective and hypothesis
    {hypothesis}
    Fine-tune ImageNet-pretrained MobileNetV3-Small with seven logits on the same audited manifests.
    The controlled variable is training augmentation; this experiment makes no claim to infer internal mood.
    All metrics below must come from this notebook's execution, and the held-out test is untouched here."""),
    md("## Verify this notebook's GPU kernel\nSelect the project `.venv`; this is the first executable cell and forbids CPU fallback."), code(GPU),
    md("## Imports, seed and visible configuration\nAdamW, cosine schedule, full-network fine-tuning, 30-epoch limit and patience 8. Batch size starts at 32. OOM requires an explicitly recorded change, not automatic CPU fallback."), code(SETUP),
    code(f'''
    config = load_config(ROOT, "{experiment}")
    RESUME = False  # Set True only to restore this run's last completed epoch.
    SHOW_PRIVATE_SAMPLES = False
    # Edit documented settings here BEFORE starting a new run, e.g. config["batch_size"] = 16.
    if not config["authorized_data"] or not config["label_mapping_verified"]:
        raise RuntimeError("Complete authorized-use and numeric label verification in common.yaml first.")
    seed_everything(config["seed"])
    output = ROOT / "outputs" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise RuntimeError("This run is already complete. Use comparison notebook; preserve frozen experiments.")
    if (output / "last.pt").exists() and not RESUME:
        raise RuntimeError("A run already exists. Set RESUME=True instead of overwriting it.")
    config["environment"] = environment()
    display(pd.Series(config))
    '''),
    md("## Dataset and augmentation methodology\nRGB direct resize to 224×224 and ImageNet normalization are identical for validation and inference. M0 has no random transformations; M1 adds ±10° rotation, 3% translation and modest brightness/contrast jitter. No vertical flip, rebalancing or aggressive crop."),
    code('''
    from faceattend.data import make_loaders, load_manifests, make_transform
    from faceattend import MEAN, STD
    tables, fingerprints = load_manifests(ROOT, config)
    config["manifest_fingerprints"] = fingerprints
    loaders = make_loaders(ROOT, config)
    display(pd.DataFrame({name: table.class_name.value_counts() for name, table in tables.items() if name != "test"}).fillna(0))
    print("Training transform:", make_transform(config, training=True))
    print("Validation transform:", make_transform(config, training=False))
    '''),
    md("## Optional local augmentation preview\nUse a training image only. Clear face-image outputs before sharing or committing."),
    code('''
    if SHOW_PRIVATE_SAMPLES:
        from PIL import Image
        from torchvision.transforms.functional import to_pil_image
        image = Image.open(ROOT / tables["train"].iloc[0].path).convert("RGB")
        augmentation = make_transform(config, training=True)
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for axis in axes:
            tensor = augmentation(image)
            visible = tensor * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
            axis.imshow(to_pil_image(visible.clamp(0, 1)))
            axis.axis("off")
        fig.suptitle("Training image transformations — private local preview")
        plt.show()
    # Reset RNG after previews to preserve experimental control.
    seed_everything(config["seed"])
    '''),
    md("## Pretrained model, optimizer, scheduler and AMP\nBoth runs load the exact IMAGENET1K_V1 weights and fine-tune all layers. Missing pretrained weights must stop training rather than substitute random initialization."),
    code('''
    from faceattend.models import build_model
    from faceattend.engine import train_one_epoch, validate_one_epoch, assert_gpu, save_checkpoint, resume_checkpoint
    from faceattend.metrics import plot_history, save_metrics
    from faceattend.utils import file_hash
    model = build_model(pretrained=True).to(device)
    assert_gpu(model)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"], init_scale=config["amp_initial_scale"])
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("AMP enabled:", scaler.is_enabled(), "Device:", next(model.parameters()).device)
    history, best_key, stale, start_epoch = [], (-1., -1., float("-inf")), 0, 1
    if RESUME:
        saved = resume_checkpoint(output / "last.pt", model, config, optimizer, scheduler, scaler, loaders)
        history, best_key, stale = saved["history"], tuple(saved["best_key"]), saved["stale"]
        start_epoch = saved["epoch"] + 1
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    write_json(output / "config_resolved.json", config)
    '''),
    md("""## Visible training orchestration
    Select checkpoints by validation macro F1, then accuracy, then loss. Early stopping resets only on a strict F1 improvement.
    Save resumable `last.pt` and selected `best.pt` after epochs; disk logs survive cleared notebook output.
    Restart after interruption by rerunning from the top with `RESUME=True`. An interrupted epoch is repeated from the previous saved RNG state.
    On OOM, inspect usage and record a smaller batch size for a fresh, comparable pair of runs."""),
    code('''
    import time
    from tqdm.auto import tqdm
    run_start = time.perf_counter()
    try:
        for epoch in tqdm(range(start_epoch, config["epochs"] + 1), desc=config["experiment"]):
            if stale >= config["patience"]:
                break
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            epoch_start = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, config["amp"])
            val = validate_one_epoch(model, loaders["val"], criterion, device, config["amp"])
            torch.cuda.synchronize()
            row = {"epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
                   "train_acc": train["accuracy"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
                   "learning_rate": learning_rate, "epoch_seconds": time.perf_counter() - epoch_start,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                   "optimizer_steps": train["optimizer_steps"], "amp_skipped_steps": train["amp_skipped_steps"]}
            history.append(row)
            key = (val["macro_f1"], val["accuracy"], -val["loss"])
            stale = 0 if key[0] > best_key[0] else stale + 1
            improved = key > best_key
            if improved:
                best_key = key
            scheduler.step()
            if improved:
                save_checkpoint(output / "best.pt", model, config, epoch, val, history, best_key, stale, optimizer, scheduler, scaler, loaders)
            save_checkpoint(output / "last.pt", model, config, epoch, val, history, best_key, stale, optimizer, scheduler, scaler, loaders)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            print(f"Epoch {epoch}: train loss={train['loss']:.4f}; val accuracy={val['accuracy']:.4f}; macro F1={val['macro_f1']:.4f}; GPU MB={row['peak_gpu_memory_mb']:.1f}")
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM: training stopped. Record a lower batch size and restart a controlled pair of runs; no CPU fallback.")
        raise
    except KeyboardInterrupt:
        print("Interrupted: resume last.pt from the previous completed epoch. Do not treat this run as finished.")
        raise
    write_json(output / "completed.json", {"last_epoch": history[-1]["epoch"], "stop_reason": "patience" if stale >= config["patience"] else "epoch_limit", "session_seconds": time.perf_counter() - run_start})
    '''),
    md("## Authentic validation results\nReload the selected checkpoint and report its recorded metrics. Held-out testing is reserved for notebook 04."),
    code('''
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    display(pd.DataFrame(history).tail())
    plot_history(history, output / "training_curves.png")
    plt.show()
    save_metrics(selected["validation_metrics"], output, "validation")
    write_json(output / "metrics.json", selected["validation_metrics"])
    plt.show()
    print("Best epoch:", selected["epoch"], "Validation macro F1:", selected["validation_macro_f1"])
    '''),
    md("""## Interpretation and limitations
    Results exist only after successful execution. A single seed does not quantify run-to-run uncertainty.
    Class imbalance motivates macro F1 and per-class inspection. GPU nondeterminism may still vary across hardware/library versions.
    Augmentation may help or hurt; do not select settings based on the held-out test. Predictions remain visual categories with uncalibrated scores.""")]
    save(f"{number}_{title.replace(' ', '_')}.ipynb", cells)


training_notebook("m0_pretrained_baseline", "02", "M0 Pretrained Baseline Training", "Establish a minimal-augmentation pretrained baseline.")
training_notebook("m1_pretrained_augmented", "03", "M1 Pretrained Augmented Training", "Test whether conservative augmentation improves validation macro F1; improvement is not guaranteed.")

save("04_Model_Comparison_and_Export.ipynb", [
    md("""# Model comparison, frozen evaluation and export
    ## Objective
    Compare completed experiments using validation evidence, freeze the winning choice, then evaluate held-out test exactly once per frozen checkpoint.
    Cached test receipts prevent routine reruns. Never tune after viewing test results."""),
    code(GPU), md("## Setup and load authentic run histories"), code(SETUP),
    code('''
    from faceattend.evaluation import EXPERIMENTS, freeze_selection, evaluate_and_export
    from faceattend.metrics import plot_history, save_metrics
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for experiment in EXPERIMENTS:
        folder = ROOT / "outputs" / experiment
        if not (folder / "completed.json").exists():
            raise RuntimeError(f"TRAINING NOT COMPLETED: {experiment}. No comparison metrics may be invented.")
        history = pd.read_csv(folder / "history.csv")
        color, style = ("#1f77b4", "-") if experiment == "m0_pretrained_baseline" else ("#ff7f0e", "--")
        axes[0].plot(history.epoch, history.val_macro_f1, label=experiment, color=color, linestyle=style)
        axes[1].plot(history.epoch, history.val_acc, label=experiment, color=color, linestyle=style)
    for axis, label in zip(axes, ["Validation macro F1", "Validation accuracy"]):
        axis.set(xlabel="Epoch", ylabel=label, title=label)
        axis.legend()
    fig.suptitle("Source: authentic per-run history.csv files")
    fig.tight_layout()
    (ROOT / "outputs/comparison").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "outputs/comparison/validation_curves.png", dpi=140)
    plt.show()
    '''),
    md("## Freeze the validation decision\nAssert matching controlled settings and manifests. Winner uses macro F1, then accuracy, then lower loss."),
    code('''
    frozen = freeze_selection(ROOT)
    display(pd.DataFrame(frozen["models"]))
    print("Frozen winner:", frozen["winner"])
    for row in frozen["models"]:
        checkpoint = torch.load(ROOT / "outputs" / row["model"] / "best.pt", map_location="cpu", weights_only=True)
        save_metrics(checkpoint["validation_metrics"], ROOT / "outputs/comparison", row["model"] + "_validation")
        plt.show()
    '''),
    md("## Held-out evaluation, warmed GPU latency and portable export\nTest results cannot change the already saved winner. Batch-1 FP32 latency uses 20 warm-ups, 100 synchronized forwards and reports median and p95; excludes image preprocessing and host transfer."),
    code('''
    comparison = evaluate_and_export(ROOT, device)
    display(comparison)
    print("Export:", ROOT / "models/faceattend_best.pt")
    '''),
    md("""## Interpretation
    Discuss validation and test accuracy, macro F1, weak classes, latency and peak training memory from the authentic comparison table.
    Do not promise that M1 wins. The portable checkpoint carries architecture, class order and exact preprocessing;
    inference reconstructs the network without downloading initial weights. Results cannot establish true emotional state.""")])

save("05_Inference_Smoke_Test.ipynb", [
    md("""# GPU inference smoke test
    ## Objective
    Load the exported state dictionary without retraining or downloading pretrained weights, and score one authorized user-supplied local image.
    Softmax values are uncalibrated model scores, not emotional measurements."""),
    code(GPU), md("## Setup and choose a local image\nThe default smoke input is one image from the user-supplied training partition. This checks checkpoint loading and output shape, not generalization. Set `IMAGE_PATH` to another authorized local file if desired. Do not use held-out test images for iterative model tuning."), code(SETUP),
    code('''
    IMAGE_PATH = ROOT / pd.read_csv(ROOT / "data/processed/train.csv").iloc[0].path
    # Optional override: IMAGE_PATH = ROOT / "data/local_example.jpg"
    CHECKPOINT = ROOT / "models/faceattend_best.pt"
    if not CHECKPOINT.exists():
        raise FileNotFoundError("No exported model. Complete both training runs and notebook 04 first.")
    if IMAGE_PATH is None or not Path(IMAGE_PATH).is_file():
        raise FileNotFoundError("Set IMAGE_PATH to your authorized local image.")
    '''),
    md("## Reconstruct matching preprocessing and run on CUDA\nScores follow the fixed seven-class order; repeated inference checks consistent output."),
    code('''
    from faceattend.inference import predict_image
    from faceattend import CLASS_NAMES
    import numpy as np
    scores = predict_image(CHECKPOINT, IMAGE_PATH)
    repeat = predict_image(CHECKPOINT, IMAGE_PATH)
    assert list(scores) == CLASS_NAMES and len(scores) == 7
    assert np.isclose(sum(scores.values()), 1.0)
    assert np.allclose(list(scores.values()), list(repeat.values()), atol=1e-6)
    display(pd.Series(scores, name="Uncalibrated visible-expression score"))
    print("Predicted annotation category:", max(scores, key=scores.get))
    '''),
    md("""## Limitations
    Out-of-distribution faces, occlusion, lighting, pose and ambiguous expressions can cause mistakes.
    No identity matching or attendance action occurs. Scores are not calibrated confidence and must not be used for student/employee assessment.
    Do not publish local face-image outputs.""")])

print("Created six valid nbformat v4 notebooks.")
