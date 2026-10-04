import sys,json,hashlib,datetime,py_compile,contextlib,io
from pathlib import Path
R=Path.cwd();O=R/'experiments/2026-10-01/session-2156/platform';sys.path[:0]=[str(R/'scripts'),str(R/'reports')]
from xpuoj_web import API,_session,submit
from pool_client import status
from collect_timing_anomalies_20260930 import parse_detail,TERMINAL
s=_session()
def read(e,b):
 r=s.post(API+e,json=b,timeout=25);r.raise_for_status();return r.json()
def save(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
sha='f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a';p=O/'R151-f9ca009609bc.py';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;py_compile.compile(str(p),cfile=str(O/'R151.pyc'),doraise=True)
assert not (O/'R151-submit.json').exists()
l=read('contest/play/querySubmissions',{'contestId':13,'problemOrder':1,'takeCount':10,'locale':'zh_CN'})
for row in l['submissions']:
 d=read('submission/getSubmissionDetail',{'submissionId':str(row['id']),'locale':'zh_CN'});v=parse_detail(d)
 if v['status'] not in TERMINAL:raise SystemExit(f'In flight {row["id"]}')
 if row['id']>153151 and v['source_sha256']==sha:raise SystemExit(f'Already repeated same SHA {row["id"]}')
q=status();safe={k:q.get(k) for k in ('browser_online','fresh','need','target')};assert safe['fresh']['submit_problem']>0
start=datetime.datetime.now().isoformat();save('R151-preflight.json',{'start':start,'sha':sha,'pool':safe})
with contextlib.redirect_stderr(io.StringIO()):sid=submit(b.decode())
save('R151-submit.json',{'sid':sid,'sha256':sha,'channel':'web','start_at':start,'submitted_at':datetime.datetime.now().isoformat(),'source':str(p)});print('SID',sid,'SHA',sha)
