"""Use a private consent manifest; never write raw captures or pair identities publicly."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from face_verification import load_config,initialize_models
from face_verification.protocol import load_protocol,score_protocol
from face_verification.calibration import calibrate_development,evaluate_held_out,save_calibration

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--confirm-consented-adult-data',action='store_true')
args = parser.parse_args()
config = load_config(root=ROOT)
manifest = load_protocol(args.manifest,consent_confirmed=args.confirm_consented_adult_data)
bundle = initialize_models(config)
pairs = score_protocol(manifest,bundle)
calibration = calibrate_development(pairs['development'],bundle.model_id,config)
save_calibration(calibration,config.path(config.threshold_file))
evaluation = evaluate_held_out(pairs['development'],pairs['test'],calibration)
save_calibration(evaluation,ROOT/'reports/phase2/evaluation_metrics.json')
from face_verification.plotting import plot_scores
if pairs['test']:
    plot_scores(pairs['test'],ROOT/'reports/phase2/held_out_scores.png',title='Consented held-out subjects — descriptive verification scores')
print(json.dumps({'calibration_status':calibration['status'],'evaluation_status':evaluation['status'],
    'threshold':calibration['threshold'],'production_ready':False},indent=2))
