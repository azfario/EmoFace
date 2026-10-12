"""Keep authentic summary/plot outputs; remove expired progress-widget views."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
count = 0
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None:
            raise RuntimeError(f"Notebook incomplete: {path.name}; cell {cell.id}")
        if any(output.output_type == "error" for output in cell.outputs):
            raise RuntimeError(f"Notebook contains an execution error: {path.name}; do not hide it.")
        retained = [output for output in cell.outputs if "application/vnd.jupyter.widget-view+json" not in output.get("data", {})]
        count += len(cell.outputs) - len(retained)
        cell.outputs = retained
    notebook.metadata.pop("widgets", None)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"{path.name}: executed, no errors; {path.stat().st_size:,} bytes")
print(f"Removed {count} expired progress-widget views; genuine summaries and PNG plots retained.")
