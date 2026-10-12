> For another PC, follow [the source-checkout setup guide](SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# RAF-DB dataset audit

## Published-reference comparison

A follow-up comparison found **strong structural agreement with RAF-DB's basic-expression subset**: all fourteen published train/test class counts match, all expected aligned filenames are present, and there are no extra names or missing CSV values. The canonical label order is now independently corroborated by the original paper and an official research implementation. This supports the mirror mapping at distribution level; original per-image annotations and image checksums remain unverified. See [the reference-alignment assessment](RAFDB_ALIGNMENT.md) for sources and precise limits. The subsequent user instruction to proceed is recorded in configuration, with these limits retained.

## Source and authorization

The pre-existing local folder is `archive (4)/`, containing `DATASET/train/<numeric label>/`, `DATASET/test/<numeric label>/`, `train_labels.csv` and `test_labels.csv`. CSV columns are `image,label`; aligned JPEG basenames carry `train_` or `test_` prefixes. No code or Git repository existed at the project root. Original source files are preserved in place.

The user-selected source is [shuvoalok/raf-db-dataset](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset). Its public metadata reports version 2, updated 20 September 2023, license **Other (specified in description)**. Public description was fetched on 9 October 2026 via Kaggle's dataset API and states non-commercial research use, restrictions on distribution and limited internal copying at a single site. Following the terms/alignment review, the user instructed proceeding with the supplied files. `authorized_data: true` records permission to perform this local project work; it does not establish third-party rights, license a commercial use or authorize redistribution.

The [original RAF-DB page](https://www.whdeng.cn/RAF/model1.html) could not be retrieved during this run (HTTPS 502; HTTP lowercase URL unavailable). No original label annotation file is present locally. This mirror has not been authenticated against original RAF-DB checksums or per-image labels.

## Numeric label semantics

`configs/label_map.json` explicitly proposes the original RAF basic-expression convention from the supplied handoff:

| Source ID | Model ID | Proposed category |
| --- | --- | --- |
| 1 | 0 | surprise |
| 2 | 1 | fear |
| 3 | 2 | disgust |
| 4 | 3 | happy |
| 5 | 4 | sad |
| 6 | 5 | angry |
| 7 | 6 | neutral |

The numeric Kaggle mapping is now **accepted for this experiment based on the published-reference alignment**, following the user's instruction to proceed. The canonical meanings are corroborated by primary research, and all fourteen train/test class counts and aligned filename sequences match the benchmark. `label_mapping_verified: true` records this distribution-level verification; original per-image labels/checksums are still **not authenticated**. Agreement between CSV numbers and folder names alone would not have been sufficient. For stronger source verification, compare all filenames/labels to provenance-verified original `EmoLabel/list_patition_label.txt` using `scripts/verify_mirror_labels.py --original-labels <path> --mirror "archive (4)"`. The script writes a fingerprinted receipt without toggling authorization flags.

Aliases `happiness/happy`, `anger/angry` and `sadness/sad` are supported explicitly. Unknown labels, missing/ambiguous images, corrupt images, unannotated images and CSV/folder/partition conflicts raise actionable errors.

## Executed audit evidence

The real machine-readable audit is `outputs/dataset_audit.json`; the private inventory is `outputs/dataset_inventory.csv`. All **15,339 images decoded successfully**, all at **100×100**. Source membership is **12,271 train / 3,068 test**. Three SHA256-identical pairs exist: two pairs within source train and one pair within source test; there is no exact hash overlap across source train/test. Seed-42 provisional split checks passed zero cross-split path/filename/hash overlap; provisional manifests are private QA evidence under `.cache/provisional_manifests`, not authorized training inputs. Duplicate rows are preserved. Notebook 01 reproduces the audit without GPU training.

Provisional per-class counts (mapping semantics still awaiting verification):

| Category | Source train | Source test |
| --- | ---: | ---: |
| surprise | 1290 | 329 |
| fear | 281 | 74 |
| disgust | 717 | 160 |
| happy | 4772 | 1185 |
| sad | 1982 | 478 |
| angry | 705 | 162 |
| neutral | 2524 | 680 |

Expected benchmark counts are 12,271 train and 3,068 test; a count match alone is not proof of original benchmark provenance. Source train/test CSVs and filename prefixes are retained. Missing identity annotations mean identity-level leakage cannot be ruled out.

Image decode checks, dimensions, class imbalance, SHA256 duplicates and equal 64-bit difference-hash candidates across source partitions are audited. Four cross-source pairs (8 images) were flagged. Difference hashes are a limited near-duplicate screen with false matches, not an exhaustive face-identity or similarity test. Candidate filenames stay in private outputs; review before trusting robustness claims.

## Split policy

After authorization and label verification, preserve source test membership and derive stratified 85% train / 15% validation from source train, seed 42. Source provenance is retained in every row. Manifests include repository-relative paths, filename, class ID/name, source split, SHA256, dimensions, difference hash and final split. Fingerprints detect later manifest changes.

Cross-split path, filename or exact-file-hash overlap aborts generation. Duplicates are reported, not silently removed or transferred. If duplicates exist, document a source-resolution decision before training. Test data cannot choose augmentations, epochs, hyperparameters or the winner.

## Sharing

Keep archives, face images, private inventory, manifests and local outputs out of Git. Notebook face previews default off. Even ignored image files can become embedded in `.ipynb` outputs: clear private image outputs before sharing. Aggregate summaries may be retained only as permitted by the dataset's terms.
