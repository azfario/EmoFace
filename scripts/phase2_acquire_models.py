"""Explicit official acquisition; never triggered by an inference import."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from face_verification import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--accept-noncommercial-research-terms', action='store_true')
parser.add_argument('--confirm-consented-adult-data-available', action='store_true')
args = parser.parse_args()
if not (args.accept_noncommercial_research_terms and args.confirm_consented_adult_data_available):
    parser.error('Both model-use authorization and consented adult data availability must be confirmed, as required by the Phase 2 handoff.')
config = load_config(root=ROOT)
from insightface.utils.storage import ensure_available
folder = ensure_available('models', config.pack, root=str(config.path(config.model_root)))
print(f'Official pack acquired locally: {Path(folder).relative_to(ROOT)}. Weights must not be redistributed.')
