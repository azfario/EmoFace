import ast
from pathlib import Path
import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_six_notebooks():
    assert len(NOTEBOOKS) == 6


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_valid_ordered_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.kernelspec.name == "faceattend-gpu"
    assert notebook.cells[0].cell_type == "markdown"
    codes = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for source in codes:
        ast.parse(source)
    if path.name[:2] != "01":
        assert "torch.cuda.is_available()" in codes[0] and "x @ x" in codes[0]
    if path.name[:2] in ["02", "03"]:
        assert any("for epoch in" in source for source in codes)
