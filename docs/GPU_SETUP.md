> For another PC, follow [the source-checkout setup guide](SETUP_NEW_PC.md). Historical test/preservation launchers referenced below belong to the original local workspace; use `scripts/run_checks.py` in a new clone.

# GPU setup and troubleshooting

## Observed hardware and software

On 9 October 2026, `nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv` reported NVIDIA GeForce RTX 3060 Laptop GPU, driver 576.52, 6144 MiB and capability 8.6. Windows CIM enumeration was denied in the sandbox; NVIDIA's tool supplied hardware evidence instead. `nvidia-smi` displayed CUDA 12.9 driver capability, not an installed PyTorch runtime.

Python on PATH was a Windows Store alias. Existing `anaconda` and `anaconda3` Python interpreters were both 3.8.3, unsuitable for the selected current wheel. A local `.venv` was created from the available bundled CPython 3.12.14 runtime. This virtual environment depends on that local base runtime: it is not portable to another machine. Recreate it with an installed supported CPython interpreter if the desktop runtime moves or is removed. No global Python installation or security policy was changed.

Official PyTorch pages exposed inconsistent selector snapshots: the local page displayed an older selector while the official version-specific installation page listed 2.13.0. The exact matching Windows cu126 wheels were successfully obtained from the official index and validated by actual CUDA computation. CUDA 12.6 was selected for driver 576.52 rather than a CUDA 13 build, whose driver requirements differ. See [PyTorch commands](https://pytorch.org/get-started/previous-versions/) and [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Exact project commands

The environment has already been created on this laptop. On a new machine first inspect hardware and install a supported Python version, then create `.venv` with that interpreter. Recheck official wheel support instead of copying package versions blindly.

```powershell
python -m venv .venv
New-Item -ItemType Directory -Path '.cache\tmp' -Force | Out-Null
$env:TEMP = Join-Path (Get-Location) '.cache\tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126 --cache-dir .cache/pip
.\.venv\Scripts\python.exe -m pip install ipykernel jupyter nbformat nbclient numpy pandas pillow scikit-learn matplotlib PyYAML tqdm pytest --cache-dir .cache/pip
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name faceattend-gpu --display-name 'Python (FaceAttend GPU)'
```

The install uses project-local temp/cache folders because the sandbox's default temporary folder was not writable. `pyproject.toml` deliberately does not declare torch/torchvision dependencies, so installing project metadata cannot replace the validated GPU wheels. The notebook adds `src/` to its import path. Direct `.venv` Python invocation avoids PowerShell activation-policy changes.

Terminal verification actually printed:

```text
Python: <project>\.venv\Scripts\python.exe
PyTorch: 2.13.0+cu126
Torchvision: 0.28.0+cu126
PyTorch CUDA build: 12.6
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
Tensor matrix product device: cuda:0
```

Notebook 00 also passed in a fresh project Jupyter kernel launched for command-line QA; its authentic outputs are retained. A pretrained two-tensor smoke batch completed a real CUDA backward pass and AdamW update with AMP initial scale 1024 (94.48 MiB peak allocation, 1 completed update, 0 skipped). The first smoke attempt at the default higher AMP scale skipped its update; this is documented rather than counted as a successful training step. Both experiments use the same configurable lower starting scale. These checks provide no RAF-DB accuracy result.

## VS Code kernel verification

The Python, Pylance, Python Environments and Jupyter extensions were listed by VS Code's CLI. Workspace settings and extension recommendations are provided. Native GUI control is unavailable in this session, so the actual VS Code interpreter/kernel selection is **not claimed as verified**.

Open this project root. Use **Python: Select Interpreter** and select `.venv\Scripts\python.exe`. Open each notebook, click **Select Kernel**, choose **Python Environments**, then the same `.venv`. Alternatively choose the named FaceAttend kernel if VS Code lists it. Run the first executable cell. Confirm the Python path, CUDA availability and tensor test; the correct Python interpreter selection alone does not select a notebook kernel.

After reopening VS Code, repeat this check before training. Notebook metadata names the expected kernel but does not enforce your GUI choice. The kernel spec lives under `.venv/share/jupyter/kernels/faceattend-gpu`. For command-line execution, the QA script sets `JUPYTER_PATH` to that folder's parent and stores Jupyter runtime/IPython files locally.

## Common failures

- **Wrong kernel:** printed Python path points outside `.venv`; reselect the notebook kernel.
- **CPU-only torch:** version lacks a CUDA build or `torch.version.cuda` is `None`; install the verified CUDA wheel with the project interpreter.
- **Driver/wheel mismatch:** inspect `nvidia-smi`, check official compatibility, and choose an appropriate wheel. Do not install a driver silently or use CPU as a workaround.
- **CUDA OOM:** stop, inspect `nvidia-smi`, reduce batch size explicitly and document the new experiment settings. No automatic retries or fallback.
- **Windows worker spawn errors:** keep `num_workers=0` until notebook execution works end-to-end.
- **AMP instability:** stop on non-finite loss; investigate and explicitly disable AMP for both controlled runs if needed.
- **Interrupted training:** restore `last.pt` using `RESUME=True`; a partially processed epoch is repeated.
- **Environment disappears:** recreate `.venv` from a supported installed Python and repeat terminal and notebook CUDA tests.

CPU image decoding and unit tests are allowed. Model parameters, batches, logits, loss, backward pass, optimizer updates and benchmark operations must use CUDA. Validation uses `torch.inference_mode()`. Batch-1 latency excludes preprocessing/host transfer and synchronizes around each warmed forward.
