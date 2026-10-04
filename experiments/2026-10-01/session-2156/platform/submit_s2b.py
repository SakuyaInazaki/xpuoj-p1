import sys,json,hashlib,datetime,py_compile,contextlib,io
from pathlib import Path
R=Path.cwd();O=R/'experiments/2026-10-01/session-2156/platform';sys.path[:0]=[str(R/'scripts'),str(R/'reports')]
from xpuoj_web import API,_session,submit
from pool_client import status
from xpuoj_pow import Client
from collect_timing_anomalies_20260930 import parse_detail,TERMINAL
s=_session()
def read(e,b):
 r=s.post(API+e,json=b,timeout=25);r.raise_for_status();return r.json()
def save(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
sha='c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b';p=O/'S2b-c82b6ee4286c.py';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;py_compile.compile(str(p),cfile=str(O/'S2b.pyc'),doraise=True)
assert not (O/'S2b-submit.json').exists()
assert datetime.datetime.now()<datetime.datetime(2026,10,1,23,25), 'After23:25 requires fresh root decision'
states=[json.loads(p.read_text()) for p in O.glob('*-audit.json')]
# Root deadline adjustment: independent S2b first trial; no parent-result gate.
assert not any(v.get('full_ac_audit',False) and v.get('displayScore',0)>=90 for v in states)
l=read('contest/play/querySubmissions',{'contestId':13,'problemOrder':1,'takeCount':10,'locale':'zh_CN'})
for row in l['submissions']:
 d=read('submission/getSubmissionDetail',{'submissionId':str(row['id']),'locale':'zh_CN'});v=parse_detail(d)
 if v['status']=='Accepted' and (v['displayScore'] or 0)>=90:raise SystemExit(f'New accepted raw>=90 {row["id"]}')
 if v['status'] not in TERMINAL and row['id'] not in (153550,153620):raise SystemExit(f'Other in flight {row["id"]}')
 if row['id']>153151 and v['source_sha256']==sha:raise SystemExit(f'Already repeated same SHA {row["id"]}')
q=status();safe={k:q.get(k) for k in ('browser_online','fresh','need','target')}
client=Client();credit=client.credit();assert credit['api_token']['available']>=1;save('S2b-credit-before.json',credit)
start=datetime.datetime.now().isoformat();save('S2b-preflight.json',{'start':start,'sha':sha,'pool':safe})
sid=client.submit(b.decode())
save('S2b-submit.json',{'sid':sid,'sha256':sha,'channel':'api','start_at':start,'submitted_at':datetime.datetime.now().isoformat(),'source':str(p)});print('SID',sid,'SHA',sha)
