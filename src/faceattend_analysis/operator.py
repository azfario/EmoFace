"""Explicit manual local per-face diagnostics; no automatic capture/history."""
import argparse,getpass,json,sys
import httpx
from faceattend_enrollment.security import protected_path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--confirm-adult-consent',action='store_true')
    parser.add_argument('--confirm-visible-expression-research',action='store_true')
    parser.add_argument('--include-candidates',action='store_true',help='Also opt in to authorized unverified candidate ranking.')
    parser.add_argument('--top-k',type=int,choices=range(1,6),default=3)
    parser.add_argument('--private-output',action='store_true',help='Show transient expression prediction and optional unverified names locally; do not redirect/share.')
    args=parser.parse_args();files=[]
    try:
        if not args.confirm_adult_consent or not args.confirm_visible_expression_research or not 1024<=args.port<=65535:raise ValueError
        path=protected_path(args.image)
        with path.open('rb') as stream:data=stream.read(8*1024*1024+1)
        if not 0<len(data)<=8*1024*1024:raise ValueError
        if data[:8]==b'\x89PNG\r\n\x1a\n':mime='image/png'
        elif data[:3]==b'\xff\xd8\xff':mime='image/jpeg'
        else:raise ValueError
        files.append(('image',('operator-probe',data,mime)))
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',trust_env=False,timeout=120) as client:
            login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':getpass.getpass('Operator password: ')})
            if login.status_code!=200:raise ValueError
            headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-expression-opt-in':'1'}
            if args.include_candidates:headers['x-faceattend-candidate-opt-in']='1'
            try:
                response=client.post('/api/v1/demo/analyze-face',params={'include_candidates':str(args.include_candidates).lower(),'top_k':args.top_k},
                    headers=headers,data={'adult_consent':'true','research_opt_in':'true'},files=files)
                if response.status_code!=200:
                    print(json.dumps({'status':response.json().get('status','REQUEST_FAILED')}));return 1
                value=response.json()
                safe={key:value[key] for key in ('status','face_count','verified','threshold','not_a_mood_or_diagnosis','persistent_inference_history','warning')}
                print(json.dumps(value if args.private_output else safe));return 0
            finally:
                files.clear();client.delete('/api/v1/demo/session',headers=headers)
    except Exception:
        print('Local face analysis failed; no path, secret or individual prediction was logged.',file=sys.stderr);return 1
    finally:files.clear()
