from types import SimpleNamespace
import pytest
from faceattend_analysis import operator

@pytest.mark.parametrize('with_candidates,private_output',[(False,False),(True,False),(False,True),(True,True)])
def test_operator_explicit_optins_private_output_and_logout(monkeypatch,private_root,capfd,with_candidates,private_output):
    path=private_root/'probe.jpg';path.write_bytes(b'\xff\xd8\xffsynthetic')
    value={'status':'PER_FACE_RESEARCH_DIAGNOSTIC_COMPLETED','face_count':1,'verified':False,'threshold':None,
        'not_a_mood_or_diagnosis':True,'persistent_inference_history':False,'warning':'Research only',
        'faces':[{'visible_expression_prediction':{'label':'neutral'},'possible_candidates':[{'display_name':'Fictional personal reference'}]}]}
    calls=[]
    class Client:
        def __init__(self,**kwargs):assert kwargs['base_url']=='http://127.0.0.1:8000' and kwargs['trust_env'] is False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,url,**kwargs):
            calls.append((url,kwargs));return SimpleNamespace(status_code=200,json=lambda:{'csrf_token':'synthetic'} if url.endswith('/session') else value)
        def delete(self,url,**kwargs):calls.append((url,kwargs))
    monkeypatch.setattr(operator.httpx,'Client',Client);monkeypatch.setattr(operator.getpass,'getpass',lambda _: 'synthetic password')
    monkeypatch.setattr(operator.sys,'argv',['analyze','--image',str(path),'--confirm-adult-consent','--confirm-visible-expression-research']+
        (['--include-candidates'] if with_candidates else [])+(['--private-output'] if private_output else []))
    assert operator.main()==0
    output=capfd.readouterr().out
    assert ('Fictional personal reference' in output)==private_output and ('neutral' in output)==private_output
    assert ('x-faceattend-candidate-opt-in' in calls[1][1]['headers'])==with_candidates
    assert calls[-1][0].endswith('/session') and str(path) not in output and 'synthetic password' not in output

def test_operator_refuses_without_explicit_research_consent(monkeypatch,private_root):
    monkeypatch.setattr(operator.sys,'argv',['analyze','--image',str(private_root/'unused')])
    monkeypatch.setattr(operator.httpx,'Client',lambda **kw:pytest.fail('No network without consent.'))
    assert operator.main()==1
