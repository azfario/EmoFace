"""Audit RAF-DB mirrors without assuming numeric folder order."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import CLASS_NAMES
from .utils import file_hash, write_json

ALIASES = {"happiness": "happy", "anger": "angry", "sadness": "sad"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_label(value, mapping=None):
    value = str(value).strip().lower()
    if value.isdigit():
        if mapping is None or value not in mapping:
            raise ValueError(f"Numeric label {value} requires an explicit, verified label map.")
        value = str(mapping[value]).lower()
    value = ALIASES.get(value, value)
    if value not in CLASS_NAMES:
        raise ValueError(f"Unknown expression label: {value}")
    return CLASS_NAMES.index(value), value


def audit_dataset(raw_root, root, mapping=None, provenance="unspecified"):
    raw_root, root = Path(raw_root), Path(root)
    images = sorted(p for p in raw_root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No RAF-DB images in {raw_root}. Provide authorized local files.")
    index = {}
    for image in images:
        index.setdefault(image.name, []).append(image)
    annotations = raw_root / "EmoLabel/list_patition_label.txt"
    rows = []
    if annotations.exists():
        canonical = {str(i + 1): name for i, name in enumerate(CLASS_NAMES)}
        for line in annotations.read_text().splitlines():
            name, label = line.split()
            aligned = Path(name).stem + "_aligned.jpg"
            choices = index.get(aligned, index.get(name, []))
            if len(choices) != 1:
                raise ValueError(f"Missing or ambiguous annotated image: {name}")
            partition = name.split("_", 1)[0]
            rows.append((choices[0], label, partition, canonical))
    elif all((raw_root / f"{s}_labels.csv").exists() for s in ["train", "test"]):
        for partition in ["train", "test"]:
            table = pd.read_csv(raw_root / f"{partition}_labels.csv", dtype=str)
            if not {"image", "label"}.issubset(table.columns):
                raise ValueError("CSV requires image and label columns.")
            for row in table.itertuples(index=False):
                if Path(row.image).name != row.image:
                    raise ValueError(f"CSV image must be a basename: {row.image}")
                choices = index.get(row.image, [])
                if len(choices) != 1:
                    raise ValueError(f"Missing or ambiguous image: {row.image}")
                path = choices[0]
                if partition not in path.parts or not path.name.startswith(partition + "_"):
                    raise ValueError(f"Partition mismatch: {path}")
                if path.parent.name.isdigit() and path.parent.name != row.label:
                    raise ValueError(f"Folder/CSV label mismatch: {path}")
                rows.append((path, row.label, partition, mapping))
    else:
        for path in images:
            partition = next((p for p in reversed(path.parts[:-1]) if p in {"train", "test"}), None)
            if partition is None:
                raise ValueError("Official train/test metadata missing; provide audited partition annotations. No automatic derived benchmark split.")
            rows.append((path, path.parent.name, partition, mapping))
    records = []
    for path, label, partition, row_map in rows:
        if partition not in {"train", "test"}:
            raise ValueError(f"Unknown source partition: {partition}")
        class_id, class_name = normalize_label(label, row_map)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                small = np.asarray(image.convert("L").resize((9, 8)))
                dhash = format(int("".join(map(str, (small[:, 1:] > small[:, :-1]).astype(int).flatten())), 2), "016x")
        except Exception as exc:
            raise ValueError(f"Corrupt/unreadable image: {path}: {exc}") from exc
        records.append({"path": path.resolve().relative_to(root.resolve()).as_posix(),
                        "filename": path.name, "class_id": class_id, "class_name": class_name,
                        "source_partition": partition, "provenance": provenance,
                        "sha256": file_hash(path), "width": width, "height": height, "dhash": dhash})
    frame = pd.DataFrame(records)
    if frame.path.duplicated().any() or frame.filename.duplicated().any():
        raise ValueError("Duplicate annotation paths/filenames; audit before splitting.")
    if len(frame) != len(images):
        raise ValueError(f"Unannotated images: {len(images) - len(frame)}")
    report = {"source_counts": frame.source_partition.value_counts().to_dict(),
              "class_counts": frame.groupby(["source_partition", "class_name"]).size().unstack(fill_value=0).to_dict(),
              "sizes": frame.groupby(["width", "height"]).size().rename("count").reset_index().to_dict("records"),
              "duplicate_hash_rows": int(frame.sha256.duplicated(keep=False).sum()),
              "canonical_counts_match": bool((frame.source_partition == "train").sum() == 12271 and (frame.source_partition == "test").sum() == 3068),
              "partition_provenance": provenance,
              "identity_leakage": "Cannot rule out: identity annotations unavailable.",
              "near_duplicate_check": "Equal 64-bit difference hashes across source partitions are candidates, not proof."}
    candidates = frame.groupby("dhash").filter(lambda group: group.source_partition.nunique() > 1)
    report["cross_partition_dhash_candidates"] = candidates[["filename", "source_partition", "dhash"]].to_dict("records")
    return frame, report


def assert_no_overlap(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for column in ["path", "filename", "sha256"]:
                if set(splits[a][column]) & set(splits[b][column]):
                    raise ValueError(f"Leakage: {column} overlap between {a} and {b}. Resolve source duplicates before training.")


def make_manifests(frame, destination, seed=42):
    original_train = frame[frame.source_partition == "train"]
    test = frame[frame.source_partition == "test"].copy()
    train, val = train_test_split(original_train, test_size=0.15, stratify=original_train.class_id, random_state=seed)
    splits = {"train": train.copy(), "val": val.copy(), "test": test}
    assert_no_overlap(splits)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in splits.items():
        table["split"] = name
        table.sort_values("path").to_csv(destination / f"{name}.csv", index=False)
    write_json(destination / "manifest_fingerprints.json", {name: file_hash(destination / f"{name}.csv") for name in splits})
    return splits
