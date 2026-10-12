"""Compare mirror CSV numeric labels to a user-provided original RAF label file.

No images are downloaded or shown. This records evidence without silently
changing authorization/config flags. Invoke from the project root:
python scripts/verify_mirror_labels.py --original-labels <path> --mirror "archive (4)"
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceattend.utils import file_hash, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--original-labels", type=Path, required=True)
parser.add_argument("--mirror", type=Path, default=ROOT / "archive (4)")
args = parser.parse_args()
original = {}
for line in args.original_labels.read_text().splitlines():
    name, label = line.split()
    aligned = Path(name).stem + "_aligned.jpg"
    if aligned in original or label not in {str(i) for i in range(1, 8)}:
        raise ValueError(f"Invalid original basic-expression annotation: {line}")
    original[aligned] = label
mirror = {}
for split in ["train", "test"]:
    with (args.mirror / f"{split}_labels.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            name, label = row["image"], row["label"]
            if name in mirror or not name.startswith(split + "_"):
                raise ValueError(f"Duplicate or mismatched mirror annotation: {name}")
            mirror[name] = label
if original != mirror:
    missing = len(set(original) - set(mirror))
    extra = len(set(mirror) - set(original))
    mismatched = sum(original[name] != mirror[name] for name in set(original) & set(mirror))
    raise ValueError(f"Mirror/original mismatch: missing={missing}, extra={extra}, wrong_labels={mismatched}")
report = {"matched_annotations": len(mirror), "original_label_file_sha256": file_hash(args.original_labels),
          "mirror_csv_hashes": {s: file_hash(args.mirror / f"{s}_labels.csv") for s in ["train", "test"]},
          "scope": "Verified equality to the provided annotation file. User must establish its original RAF-DB provenance."}
write_json(ROOT / "outputs/label_verification.json", report)
print(json.dumps(report, indent=2))
