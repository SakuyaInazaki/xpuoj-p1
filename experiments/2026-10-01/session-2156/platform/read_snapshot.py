import sys,json,datetime,hashlib
from pathlib import Path
R=Path.cwd();O=R/'experiments/2026-10-01/session-2156/platform';sys.path[:0]=[str(R/'scripts'),str(R/'reports')]
from xpuoj_web import API,_session
from xpuoj_pow import Client
from collect_timing_anomalies_20260930 import parse_detail,TERMINAL
from pool_client import status
s=_session()
def read(e,b):
 r=s.post(API+e,json=b,timeout=25);r.raise_for_status();return r.json()
def save(n,d): (O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
rows=[];maxid=None
for _ in range(2):
 b={'contestId':13,'problemOrder':1,'takeCount':10,'locale':'zh_CN'}
 if maxid:b['maxId']=maxid
 d=read('contest/play/querySubmissions',b);p=d.get('submissions',[]);rows+=p
 if not p:break
 maxid=min(x['id'] for x in p)-1
entry=read('contest/play/getContestScoreboardMe',{'contestId':13}).get('entry')or{};q=status()
out={'at':datetime.datetime.now().isoformat(),'production_sha':hashlib.sha256((R/'p1/kernel.py').read_bytes()).hexdigest(),'me':{k:entry.get(k) for k in ('rank','totalScore','problemScores')},'recent':rows,'details':{},'pool':{k:q.get(k) for k in ('browser_online','fresh','need','target')},'credit':Client().credit(),'custom':read('judgeClient/checkAvailability',{'language':'triton-dist','requiredFlags':['custom-test']})};save('snapshot.json',out)
print(json.dumps({k:out[k] for k in ('at','production_sha','me','pool','credit','custom')},ensure_ascii=False),flush=True)
for x in rows:
 sid=x['id'];d=read('submission/getSubmissionDetail',{'submissionId':str(sid),'locale':'zh_CN'});v=parse_detail(d);d.pop('content',None);d.pop('progressSubscriptionKey',None);save(f'{sid}-raw.json',d);save(f'{sid}-audit.json',v);out['details'][str(sid)]=v;save('snapshot.json',out);print(sid,x.get('submitTime'),v['status'],v['displayScore'],v['source_sha256'],len(v['cases']),flush=True)
