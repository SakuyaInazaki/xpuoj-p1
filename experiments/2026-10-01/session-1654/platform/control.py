import sys,json,hashlib,datetime,py_compile,shutil,math
from pathlib import Path
R=Path.cwd(); O=R/'experiments/2026-10-01/session-1654/platform'
sys.path[:0]=[str(R/'scripts'),str(R/'reports')]
from xpuoj_web import API,_session,submit
from collect_timing_anomalies_20260930 import parse_detail,TERMINAL
S=_session()
def read(e,b):
 r=S.post(API+e,json=b,timeout=25);r.raise_for_status();d=r.json()
 if 'error' in d: raise RuntimeError('API error')
 return d
def save(n,d): (O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def detail(sid):
 d=read('submission/getSubmissionDetail',{'submissionId':str(sid),'locale':'zh_CN'})
 d.pop('content',None);save(f'{sid}-raw.json',d)
 # source separately fetched/hashed without persisting returned source
 return d
def snapshot():
 rows=[];maxid=None
 for i in range(12):
  b={'contestId':13,'problemOrder':1,'takeCount':10,'locale':'zh_CN'}
  if maxid:b['maxId']=maxid
  d=read('contest/play/querySubmissions',b);p=d.get('submissions',[]);rows+=p
  if not p or min(int(x['id']) for x in p)<=153006:break
  maxid=min(int(x['id']) for x in p)-1
 out={'time':datetime.datetime.now().isoformat(),'me':read('contest/play/getContestScoreboardMe',{'contestId':13}).get('entry'),'recent':rows,'details':{}}
 for x in rows:
  d=read('submission/getSubmissionDetail',{'submissionId':str(x['id']),'locale':'zh_CN'});v=parse_detail(d);v.pop('logs',None);out['details'][str(x['id'])]=v
 save('snapshot.json',out)
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':snapshot()
