import json
from types import SimpleNamespace
import pytest
from faceattend_candidates import operator

@pytest.mark.parametrize('private_output',[False,True])
def test_operator_local_auth_optin_output_and_logout(monkeypatch,private_root,capfd,private_output):
    photo=private_root/'probe.jpg';photo.write_bytes(b'\xff\xd8\xffsynthetic-only')
    calls=[]
    result={'status':'CANDIDATE_DIAGNOSTIC_COMPLETED','face_detected':True,'candidate_count':1,
        'identification_status':'UNCALIBRATED_DIAGNOSTIC','identity_status':'Unknown (unverified)',
        'verified':False,'threshold':None,'reason':'THRESHOLD_NOT_CALIBRATED','warning':'Research only',
        'possible_candidates':[{'display_name':'Fictional private alias','cosine_similarity':0.1}]}
    class Client:
        def __init__(self,**kwargs):
            assert kwargs['base_url']=='http://127.0.0.1:8000' and kwargs['trust_env'] is False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,url,**kwargs):
            calls.append((url,kwargs))
            return SimpleNamespace(status_code=200,json=lambda:{'csrf_token':'synthetic-csrf'} if url.endswith('/session') else result)
        def delete(self,url,**kwargs):calls.append((url,kwargs))
    monkeypatch.setattr(operator.httpx,'Client',Client)
    monkeypatch.setattr(operator.getpass,'getpass',lambda _: 'synthetic-password')
    monkeypatch.setattr(operator.sys,'argv',['candidate','--image',str(photo),'--confirm-adult-consent','--confirm-research-search']+(['--private-output'] if private_output else []))
    assert operator.main()==0
    output=capfd.readouterr().out
    assert ('Fictional private alias' in output)==private_output
    assert str(photo) not in output and 'synthetic-password' not in output
    assert calls[1][1]['headers']['x-faceattend-candidate-opt-in']=='1'
    assert calls[-1][0]=='/api/v1/demo/session'

def test_operator_refuses_without_consent_before_network(monkeypatch,private_root,capfd):
    monkeypatch.setattr(operator.sys,'argv',['candidate','--image',str(private_root/'unused.jpg')])
    monkeypatch.setattr(operator.httpx,'Client',lambda **kwargs:pytest.fail('Network accessed before consent.'))
    assert operator.main()==1
    assert 'personal input was logged' in capfd.readouterr().err
