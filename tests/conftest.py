from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from faceattend import CLASS_NAMES


@pytest.fixture
def source(tmp_path):
    rows = []
    for partition in ["train", "test"]:
        for label in range(7):
            count = 20 if partition == "train" else 3
            for index in range(count):
                name = f"{partition}_{label}_{index}.png"
                path = tmp_path / "raw" / "DATASET" / partition / str(label + 1) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (10, 12), (label * 30, index * 8, 50 if partition == "train" else 150)).save(path)
                rows.append({"image": name, "label": str(label + 1), "partition": partition})
        pd.DataFrame([r for r in rows if r["partition"] == partition]).drop(columns="partition").to_csv(tmp_path / "raw" / f"{partition}_labels.csv", index=False)
    mapping = {str(i + 1): name for i, name in enumerate(CLASS_NAMES)}
    return tmp_path, mapping
