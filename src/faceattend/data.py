"""Single-source RGB transforms and manifest-backed loaders."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import CLASS_NAMES, MEAN, STD
from .prepare_data import assert_no_overlap
from .utils import file_hash


def make_transform(config=None, training=False):
    config = config or {}
    operations = [T.Resize(tuple(config.get("input_size", [224, 224])), antialias=True)]
    if training:
        if config.get("horizontal_flip", False):
            operations.append(T.RandomHorizontalFlip())
        if config.get("augmentation") == "conservative":
            operations += [T.RandomAffine(config.get("rotation_degrees", 10), translate=(config.get("translation", .03),) * 2),
                           T.ColorJitter(brightness=config.get("brightness", .1), contrast=config.get("contrast", .1))]
    operations += [T.ToTensor(), T.Normalize(config.get("normalization", {}).get("mean", MEAN), config.get("normalization", {}).get("std", STD))]
    return T.Compose(operations)


class ExpressionDataset(Dataset):
    def __init__(self, manifest, root, transform):
        self.table = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        self.root = Path(root).resolve()
        self.transform = transform
        for row in self.table.itertuples():
            path = (self.root / row.path).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Missing image or unsafe manifest path: {row.path}")
            if not 0 <= row.class_id < 7 or CLASS_NAMES[row.class_id] != row.class_name:
                raise ValueError(f"Manifest label mismatch: {row.path}")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        with Image.open(self.root / row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        if tensor.shape != (3, 224, 224) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid transformed tensor: {row.path}")
        return tensor, int(row.class_id)


def load_manifests(root, config):
    import json
    folder = root / config["manifest_root"]
    tables = {name: pd.read_csv(folder / f"{name}.csv") for name in ["train", "val", "test"]}
    assert_no_overlap(tables)
    recorded = json.loads((folder / "manifest_fingerprints.json").read_text())
    for name in tables:
        if file_hash(folder / f"{name}.csv") != recorded[name]:
            raise ValueError(f"Manifest changed after audit: {name}")
    return tables, recorded


def make_loaders(root, config, include_test=False):
    tables, _ = load_manifests(root, config)
    loaders = {}
    for name in (["train", "val", "test"] if include_test else ["train", "val"]):
        dataset = ExpressionDataset(tables[name], root, make_transform(config, name == "train"))
        generator = torch.Generator().manual_seed(config["seed"])
        loaders[name] = DataLoader(dataset, batch_size=config["batch_size"], shuffle=name == "train", num_workers=config["num_workers"], pin_memory=True, generator=generator)
    return loaders
