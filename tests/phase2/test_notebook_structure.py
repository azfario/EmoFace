import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[2]

def test_six_phase2_notebooks_valid_and_opt_in():
    paths = sorted((ROOT/'notebooks/phase2').glob('*.ipynb'))
    assert len(paths) == 6
    for path in paths:
        notebook = nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata.kernelspec.name == 'faceattend-gpu'
        assert notebook.cells[0].cell_type == 'markdown'
        codes = [c.source for c in notebook.cells if c.cell_type == 'code']
        for source in codes:
            ast.parse(source)
        assert 'RUN_REAL_FACES = False' in codes[0]
        assert 'ADULT_CONSENT = False' in codes[0]
        assert 'IDENTITY_CONSENT = False' in codes[0]
        assert all('faceattend_best.pt' not in c for c in codes)
