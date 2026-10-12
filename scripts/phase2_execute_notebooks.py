"""Execute only the six Phase 2 notebooks in separate fresh project kernels."""
import os
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--consented-image-smoke',action='store_true')
parser.add_argument('--private-inputs',type=Path)
parser.add_argument('--notebook',choices=[p.name for p in sorted((ROOT/'notebooks/phase2').glob('*.ipynb'))])
args = parser.parse_args()
if args.consented_image_smoke:
    if not args.private_inputs or not (args.private_inputs/'input_manifest.json').is_file():
        parser.error('Supply the prepared private image folder and its consent manifest.')
    os.environ['FACEATTEND_CONSENTED_SMOKE'] = '1'
    os.environ['FACEATTEND_PRIVATE_INPUTS'] = str(args.private_inputs.resolve())
else:
    os.environ.pop('FACEATTEND_CONSENTED_SMOKE',None)
for key,value in {'JUPYTER_PATH':'.venv/share/jupyter','JUPYTER_RUNTIME_DIR':'.cache/jupyter/runtime',
    'IPYTHONDIR':'.cache/ipython','MPLCONFIGDIR':'.cache/matplotlib','TEMP':'.cache/tmp','TMP':'.cache/tmp'}.items():
    os.environ[key] = str(ROOT/value)
    (ROOT/value).mkdir(parents=True,exist_ok=True)
results = []
paths = sorted((ROOT/'notebooks/phase2').glob('*.ipynb'))
if args.notebook:
    paths = [p for p in paths if p.name == args.notebook]
for path in paths:
    notebook = nbformat.read(path,as_version=4)
    def save_cell(cell,cell_index,execute_reply):
        nbformat.write(notebook,path)
        print(f'{path.name}: cell {cell_index} executed',flush=True)
    try:
        NotebookClient(notebook,kernel_name='faceattend-gpu',timeout=180,startup_timeout=60,
            allow_errors=False,on_cell_executed=save_cell,store_widget_state=False,
            resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook,path)
    nbformat.validate(notebook)
    results.append({'notebook':path.relative_to(ROOT).as_posix(),'status':'EXECUTED_WITH_CONSENTED_IMAGE_DIAGNOSTIC' if args.consented_image_smoke else 'EXECUTED_DIAGNOSTIC_AND_SYNTHETIC_CELLS',
        'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells),
        'real_face_cells':'DETECTION_AND_REFUSAL_PATHS_EXECUTED; FULL_ENROLLMENT_AND_CALIBRATION_INSUFFICIENT_DATA' if args.consented_image_smoke else 'EXPLICIT_OPT_IN_DISABLED','error_outputs':sum(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs)})
import json
if args.notebook and (ROOT/'reports/phase2/notebook_execution.json').exists():
    previous = json.loads((ROOT/'reports/phase2/notebook_execution.json').read_text())
    replaced = {r['notebook'] for r in results}
    results = sorted(results+[r for r in previous if r['notebook'] not in replaced],key=lambda r:r['notebook'])
(ROOT/'reports/phase2/notebook_execution.json').write_text(json.dumps(results,indent=2))
print('Execution finished; inspect the Phase 2 execution report for enabled sections and remaining data limitations.',flush=True)
