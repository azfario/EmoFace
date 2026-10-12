"""Execute explicitly named notebooks using the project kernel; never implicit training."""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".cache/jupyter/runtime")
os.environ["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["TEMP"] = str(ROOT / ".cache/tmp")
os.environ["TMP"] = os.environ["TEMP"]
for key in ["JUPYTER_RUNTIME_DIR", "IPYTHONDIR", "MPLCONFIGDIR", "TEMP"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("notebook")
parser.add_argument("--allow-errors", action="store_true")
parser.add_argument("--timeout", type=int, default=600, help="Seconds per cell; -1 permits explicitly requested full training cells.")
args = parser.parse_args()
path = (ROOT / args.notebook).resolve()
if not path.is_relative_to(ROOT / "notebooks"):
    raise ValueError("Notebook must be within this project's notebooks folder.")
notebook = nbformat.read(path, as_version=4)


class ProgressClient(NotebookClient):
    """Forward authentic epoch summaries while preserving all notebook outputs."""

    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            content = msg.get("content", {})
            message = content.get("text", "")
            if any(word in message for word in ["Epoch ", "GPU tensor test PASSED", "Best epoch:", "Frozen winner:", "Export:", "Predicted annotation category:", "Interrupted:", "CUDA OOM"]):
                print(message, end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def record_cell(cell, cell_index, execute_reply):
    nbformat.write(notebook, path)
    print(f"{path.name}: completed code cell at index {cell_index}", flush=True)


try:
    ProgressClient(notebook, timeout=args.timeout, startup_timeout=60, kernel_name="faceattend-gpu", allow_errors=args.allow_errors,
                   on_cell_executed=record_cell, store_widget_state=False,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
finally:
    nbformat.write(notebook, path)
print(f"Executed {path.name}; inspect outputs for validation status.")
