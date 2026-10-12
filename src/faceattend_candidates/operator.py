"""Manual loopback candidate request; default output omits personal metadata."""
import argparse,getpass,json,sys
import httpx
from faceattend_enrollment.security import protected_path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,help='Consented single-face JPEG/PNG in protected local storage.')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--top-k',type=int,choices=range(1,6),default=3)
    parser.add_argument('--confirm-adult-consent',action='store_true')
    parser.add_argument('--confirm-research-search',action='store_true')
    parser.add_argument('--private-output',action='store_true',help='Show authorized candidate display names and diagnostic scores locally; do not redirect/share.')
    args=parser.parse_args();loaded=[]
    try:
        if not args.confirm_adult_consent or not args.confirm_research_search or not 1024<=args.port<=65535:
            raise ValueError
        path=protected_path(args.image)
        if not path.is_file():raise ValueError
        # A bounded read also protects against concurrent growth after stat().
        with path.open('rb') as stream:data=stream.read(8*1024*1024+1)
        if not 0<len(data)<=8*1024*1024:raise ValueError
        if data[:8]==b'\x89PNG\r\n\x1a\n':mime='image/png'
        elif data[:3]==b'\xff\xd8\xff':mime='image/jpeg'
        else:raise ValueError
        loaded.append(('image',('operator-probe',data,mime)))
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',trust_env=False,timeout=120) as client:
            login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':getpass.getpass('Operator password: ')})
            if login.status_code!=200:raise ValueError
            headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-candidate-opt-in':'1'}
            try:
                response=client.post('/api/v1/demo/candidates',params={'top_k':args.top_k},headers=headers,
                    data={'adult_consent':'true','research_opt_in':'true'},files=loaded)
                if response.status_code!=200:
                    print(json.dumps({'status':response.json().get('status','REQUEST_FAILED'),'verified':False,'threshold':None}));return 1
                value=response.json()
                safe={key:value[key] for key in ('status','face_detected','candidate_count','identification_status','identity_status','verified','threshold','reason','warning')}
                print(json.dumps(value if args.private_output else safe));return 0
            finally:
                loaded.clear()
                client.delete('/api/v1/demo/session',headers=headers)
    except Exception:
        print('Local candidate request failed; no secret, path or personal input was logged.',file=sys.stderr);return 1
    finally:loaded.clear()
