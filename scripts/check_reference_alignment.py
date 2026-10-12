"""Compare local CSVs and filenames with published RAF-DB basic statistics.

This verifies structure/distribution, not original pixel or per-image label
identity. References are recorded with the result; no training flags change.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "archive (4)"
NAMES = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
REFERENCE = {"train": [1290, 281, 717, 4772, 1982, 705, 2524],
             "test": [329, 74, 160, 1185, 478, 162, 680]}
report = {"conclusion": "STRUCTURALLY CONSISTENT with RAF-DB basic-expression benchmark",
          "counts_reference": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312359#pone-0312359-t003",
          "label_order_reference": "https://www.whdeng.cn/RAF/li_RAFDB_2017_CVPR.pdf",
          "filename_reference": "https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning",
          "original_per_image_annotations_verified": False,
          "original_image_checksums_verified": False,
          "permission_status": "not assessed by this structural check", "partitions": {}}
tables = []
for split, expected in REFERENCE.items():
    table = pd.read_csv(RAW / f"{split}_labels.csv")
    tables.append(table)
    names = set(table.image)
    sequence = {f"{split}_{index:0{5 if split == 'train' else 4}d}_aligned.jpg"
                for index in range(1, sum(expected) + 1)}
    counts = table.label.value_counts().sort_index().to_dict()
    detail = {"rows": len(table), "expected_rows": sum(expected),
              "missing_cells": int(table.isna().sum().sum()),
              "duplicate_filenames": int(table.image.duplicated().sum()),
              "missing_expected_names": len(sequence - names), "extra_names": len(names - sequence),
              "class_counts": [{"numeric_id": i, "canonical_class": NAMES[i-1],
                                "local": int(counts.get(i, 0)), "published": n,
                                "matches": int(counts.get(i, 0)) == n} for i, n in enumerate(expected, 1)]}
    detail["all_class_counts_match"] = all(item["matches"] for item in detail["class_counts"]) and len(counts) == 7
    report["partitions"][split] = detail
combined = pd.concat(tables, ignore_index=True)
files = list((RAW / "DATASET").rglob("*.jpg"))
file_names = {path.name for path in files}
report["actual_jpeg_files"] = len(files)
report["duplicate_image_basenames"] = len(files) - len(file_names)
report["missing_annotated_files"] = len(set(combined.image) - file_names)
report["unannotated_files"] = len(file_names - set(combined.image))
report["checks_passed"] = all(d["all_class_counts_match"] and not any(d[key] for key in ["missing_cells", "duplicate_filenames", "missing_expected_names", "extra_names"]) for d in report["partitions"].values()) and not any(report[k] for k in ["duplicate_image_basenames", "missing_annotated_files", "unannotated_files"])
if not report["checks_passed"]:
    report["conclusion"] = "Reference alignment check failed; inspect discrepancies"
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs/reference_alignment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
