# Runtime model assets

These three files are stored through Git LFS in the private EmoFace repository
for the owner's non-commercial research use across devices. They contain model
parameters, not participant enrollment templates or source photographs. The
Docker build excludes this directory and mounts local runtime copies read-only.

- `faceattend_best.pt` is the selected RAF-DB MobileNetV3-Small expression model.
  This portable inference export preserves every trained tensor and preprocessing
  setting while omitting local paths, environment details and training manifests.
  The original research checkpoint remains separate and unchanged. No RAF-DB
  images or annotations are included, and this publication does not grant rights
  to the training dataset or imply permission for commercial use.
- `det_10g.onnx` and `w600k_r50.onnx` are the SCRFD and recognition models from
  InsightFace's `buffalo_l` package, provided by the InsightFace authors.
  InsightFace designates its pretrained models for **non-commercial research
  only**, separately from its MIT-licensed library code. See the upstream
  [license policy](https://github.com/deepinsight/insightface#license) and
  [model package release](https://github.com/deepinsight/insightface/releases/tag/v0.7).
  Broader distribution or commercial use requires establishing the applicable
  permissions; keeping this repository private does not replace those terms.

File sizes and SHA-256 values are recorded in
[`configs/model-assets.json`](../configs/model-assets.json). The launcher checks
them before preparing a missing runtime file. It preserves existing local models.
