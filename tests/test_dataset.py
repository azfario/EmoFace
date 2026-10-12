import pytest
import torch
from PIL import Image

from faceattend.data import make_transform
from faceattend.prepare_data import audit_dataset, normalize_label


@pytest.mark.parametrize("alias,label", [("happiness", 3), ("anger", 5), ("sadness", 4), ("surprise", 0)])
def test_labels(alias, label):
    assert normalize_label(alias)[0] == label


def test_numeric_requires_mapping():
    with pytest.raises(ValueError, match="explicit"):
        normalize_label("1")
    with pytest.raises(ValueError, match="Unknown"):
        normalize_label("confused")


def test_audit_and_missing_file(source):
    root, mapping = source
    frame, report = audit_dataset(root / "raw", root, mapping, "toy test fixture")
    assert len(frame) == 161 and report["source_counts"] == {"train": 140, "test": 21}
    (root / frame.iloc[0].path).unlink()
    with pytest.raises(ValueError, match="Missing"):
        audit_dataset(root / "raw", root, mapping)


def test_corrupt_file(source):
    root, mapping = source
    image = next((root / "raw").rglob("*.png"))
    image.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Corrupt"):
        audit_dataset(root / "raw", root, mapping)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_rgb_transform(mode):
    image = Image.new(mode, (19, 23)).convert("RGB")
    tensor = make_transform()(image)
    assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()
    assert torch.allclose(tensor[:, 0, 0], torch.tensor([-.485/.229, -.456/.224, -.406/.225]))
