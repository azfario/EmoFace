"""Seven-class metrics and labeled aggregate visualizations."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import CLASS_NAMES
from .utils import write_json


def classification_metrics(targets, predictions):
    return {"accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=list(range(7)), average="macro", zero_division=0)),
            "per_class": classification_report(targets, predictions, labels=list(range(7)), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(7))).tolist()}


def plot_history(history, output):
    history = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"train": "#1f77b4", "val": "#ff7f0e"}
    styles = {"train": "-", "val": "--"}
    for axis, metric in zip(axes[:2], ["loss", "acc"]):
        for split in ["train", "val"]:
            axis.plot(history.epoch, history[f"{split}_{metric}"], label=split, color=colors[split], linestyle=styles[split])
        metric_name = "Cross-entropy loss" if metric == "loss" else "Accuracy"
        axis.set(xlabel="Epoch", ylabel=metric_name, title=f"Training and validation {metric_name.lower()}")
        axis.legend()
    axes[2].plot(history.epoch, history.val_macro_f1, label="Validation macro F1", color=colors["train"])
    axes[2].set(xlabel="Epoch", ylabel="Macro F1", title="Seven-class validation F1")
    axes[2].legend()
    fig.suptitle("Source: recorded history.csv")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    return fig


def save_metrics(metrics, destination, prefix):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / f"{prefix}_metrics.json", metrics)
    pd.DataFrame(metrics["per_class"]).T.to_csv(destination / f"{prefix}_per_class.csv")
    fig, axis = plt.subplots(figsize=(8, 7))
    matrix = metrics["confusion_matrix"]
    artist = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Predicted visible expression", ylabel="Annotated visible expression", title=f"{prefix} confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for i in range(7):
        for j in range(7):
            color = "white" if matrix[i][j] > max(max(row) for row in matrix) / 2 else "black"
            axis.text(j, i, str(matrix[i][j]), ha="center", va="center", color=color)
    fig.colorbar(artist, ax=axis)
    fig.tight_layout()
    fig.savefig(destination / f"{prefix}_confusion_matrix.png", dpi=140)
    return fig
