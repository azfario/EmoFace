"""Start only the single-process loopback service, without private access logs."""
from pathlib import Path
import sys
import uvicorn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    uvicorn.run('faceattend_api.main:app', host='127.0.0.1', port=8000, workers=1,
                reload=False, access_log=False, proxy_headers=False, log_level='warning',
                timeout_keep_alive=5, limit_concurrency=8, h11_max_incomplete_event_size=16384)
