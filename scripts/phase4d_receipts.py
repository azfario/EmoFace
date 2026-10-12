"""Keep new legacy-regression copies free of individual expression predictions."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def redact_legacy_expression(value):
    copied=json.loads(json.dumps(value))
    results=copied.get('results',{})
    if 'expression-predict' in results:
        previous=results['expression-predict']
        results['expression-predict']={key:previous[key] for key in ('status','confidence_calibrated','device') if key in previous}
        results['expression-predict']['individual_prediction_omitted']=True
    copied['expression_prediction_omitted_from_additive_receipt']=True
    return copied

def sanitize():
    folder=ROOT/'reports/phase4d';count=0
    for path in folder.glob('phase3_gpu_integration_*.json'):
        if path.is_symlink() or path.parent!=folder:raise ValueError('Unexpected receipt scope.')
        value=redact_legacy_expression(json.loads(path.read_text()))
        path.write_text(json.dumps(value,indent=2),encoding='utf-8');count+=1
    return count

if __name__=='__main__':print(json.dumps({'additive_legacy_expression_receipts_redacted':sanitize()}))
