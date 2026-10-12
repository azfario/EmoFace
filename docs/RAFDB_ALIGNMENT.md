> For another PC, follow [the source-checkout setup guide](SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# Alignment with published RAF-DB

Checked on 9 October 2026 for Phase 1's seven-class basic-expression benchmark. Unit of analysis: one annotated aligned image, keyed by its filename.

**Verdict: the local archive is strongly consistent with RAF-DB's basic-expression subset.** Its total counts, all fourteen train/test class counts, numeric label range, aligned filename sequences and 100×100 dimensions match published benchmark characteristics. This is structural/distribution verification, not proof of every original image byte or annotation.

The canonical numeric order is 1 surprise, 2 fear, 3 disgust, 4 happiness, 5 sadness, 6 anger, 7 neutral, documented in the [original RAF-DB authors' CVPR 2017 paper](https://www.whdeng.cn/RAF/li_RAFDB_2017_CVPR.pdf). The paper also describes alignment to 100×100. The source website/PDF could not be fetched directly in this session; the paper's indexed text corroborated the convention. The [official RUL research implementation](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning), coauthored by Weihong Deng, independently documents the zero-based class order, original annotation location and aligned filename structure.

All local counts below equal the published values in [Song and Liu (2025), Table 3](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312359#pone-0312359-t003). That research article cites the original RAF-DB source rather than this local archive.

| Original label | Category | Local train = published train | Local test = published test |
| --- | --- | ---: | ---: |
| 1 | surprise | 1290 | 329 |
| 2 | fear | 281 | 74 |
| 3 | disgust | 717 | 160 |
| 4 | happy | 4772 | 1185 |
| 5 | sad | 1982 | 478 |
| 6 | angry | 705 | 162 |
| 7 | neutral | 2524 | 680 |
| **Total** | | **12271** | **3068** |

Current CSV checks found zero missing cells, duplicate CSV filenames, missing annotated JPEGs or unannotated JPEGs. There are exactly 15,339 JPEGs and 15,339 annotations. Training filenames span `train_00001_aligned.jpg`–`train_12271_aligned.jpg`; test filenames span `test_0001_aligned.jpg`–`test_3068_aligned.jpg`. Both sequences have zero missing or extra names. The source data is repackaged into numeric folders/CSV files rather than the original `EmoLabel`/`Image` layout; that organizational change does not itself indicate data loss.

The earlier full decode/hash audit found all images readable at 100×100, three exact duplicate pairs within source partitions and no exact hash overlap between source train and test. Four equal difference-hash pairs across source partitions remain similarity candidates; this is a limited screening method, not an identity determination. These checks and the counts are reproducible in notebook 01; the dedicated published-count/filename comparison runs with `python scripts/check_reference_alignment.py` and writes `outputs/reference_alignment.json`.

## Assessment and limits

- **High confidence:** agreement with the seven-class basic-expression benchmark's structure and per-class distribution. Exact matching class counts in both partitions strongly corroborate the intended numeric mapping.
- **Unresolved provenance:** original `list_patition_label.txt` and original image checksums are unavailable locally. Labels could theoretically be swapped between images without changing class counts. Do not claim per-image annotations or image bytes have been authenticated against the original release.
- **Low-volume data-quality issue:** six rows (about 0.039% of the dataset) belong to three exact duplicate pairs, all within source partitions. Preserve and document these benchmark rows; do not silently rewrite the official test split.
- **Potential leakage requiring review:** four approximate-hash train/test pairs. Equal hashes alone do not prove a duplicate or a shared identity. Identity-level leakage cannot be assessed without appropriate source annotations.

For the strict handoff's original-source verification requirement, compare all filename/label pairs to a provenance-verified original annotation file using `scripts/verify_mirror_labels.py`. This review does not establish dataset permission, authorize training or change configuration flags. No temporal trend assessment applies to this static archive.
