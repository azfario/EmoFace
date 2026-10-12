"""Loopback HTTP smoke: readiness, OpenAPI, Swagger and synthetic rejection only."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
import httpx
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-server',action='store_true',help='Start a temporary one-process loopback server and stop it afterward.')
    args=parser.parse_args()
    process=None
    token=uuid4().hex[:12]
    evidence={'scope':'Actual loopback Uvicorn HTTP smoke with synthetic input; no camera/private image access',
              'raw_images_saved':False,'server_started_by_smoke':args.start_server,'checks':{},'server_stopped':False}
    try:
        with httpx.Client(base_url='http://127.0.0.1:8000',trust_env=False,timeout=10) as client:
            if args.start_server:
                # Refuse to claim/start a smoke process when this port is already occupied.
                import socket
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1',8000))
                env=os.environ.copy()
                env['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
                process=subprocess.Popen([sys.executable,str(ROOT/'scripts/phase3_serve.py')],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                deadline=time.monotonic()+120
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Local service process exited before readiness.')
                    try:
                        response=client.get('/health')
                        break
                    except httpx.TransportError:
                        if time.monotonic()>=deadline:
                            raise RuntimeError('Local service startup deadline expired.')
                        time.sleep(.25)
            response=client.get('/health')
            evidence['checks']['health']={'http':response.status_code,'body':response.json()}
            assert response.status_code==200 and response.json()['gpu_ready'] is True
            info=client.get('/api/v1/models/info')
            evidence['checks']['model_info']={'http':info.status_code,'body':info.json()}
            assert info.status_code==200
            for route in ('/docs','/openapi.json'):
                response=client.get(route)
                evidence['checks'][route]={'http':response.status_code,'content_type':response.headers.get('content-type')}
                assert response.status_code==200
            image=np.random.default_rng(3).integers(60,190,(200,200,3),dtype=np.uint8)
            stream=io.BytesIO()
            Image.fromarray(image).save(stream,format='PNG')
            response=client.post('/api/v1/face/detect',data={'adult_consent':'true','research_opt_in':'true'},
                files={'image':('synthetic.png',stream.getvalue(),'image/png')})
            evidence['checks']['synthetic_no_face']={'http':response.status_code,'status':response.json()['status']}
            assert response.status_code==422 and response.json()['status']=='NO_FACE'
            assert client.get('/health',headers={'origin':'https://example.invalid'}).status_code==403
            evidence['checks']['unsafe_origin']={'http':403}
            evidence['status']='PASSED'
    except Exception:
        evidence['status']='FAILED'
        evidence['message']='Local smoke did not meet readiness/HTTP contracts; inspect local setup without publishing private logs.'
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            evidence['server_stopped']=True
        path=ROOT/'reports/phase3'/f'http_smoke_{token}.json'
        path.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        print(json.dumps(evidence,indent=2))
    return 0 if evidence['status']=='PASSED' else 1

if __name__=='__main__':
    sys.exit(main())
