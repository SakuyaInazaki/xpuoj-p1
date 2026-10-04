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
sha='ac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8';p=O/'S2-ac338895e390.py';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;py_compile.compile(str(p),cfile=str(O/'S2.pyc'),doraise=True)
assert not (O/'S2-submit.json').exists()
states=[json.loads((O/f'{sid}-audit.json').read_text()) for sid in [153507,153526]]
assert any(v['status'] in TERMINAL for v in states)
assert not any(v['full_ac_audit'] and v['displayScore']>=90 for v in states)
l=read('contest/play/querySubmissions',{'contestId':13,'problemOrder':1,'takeCount':10,'locale':'zh_CN'})
for row in l['submissions']:
 d=read('submission/getSubmissionDetail',{'submissionId':str(row['id']),'locale':'zh_CN'});v=parse_detail(d)
 if v['status'] not in TERMINAL and row['id'] not in (153507,153526):raise SystemExit(f'Other in flight {row["id"]}')
 if row['id']>153151 and v['source_sha256']==sha:raise SystemExit(f'Already repeated same SHA {row["id"]}')
q=status();safe={k:q.get(k) for k in ('browser_online','fresh','need','target')};assert safe['fresh']['submit_problem']>0
start=datetime.datetime.now().isoformat();save('S2-preflight.json',{'start':start,'sha':sha,'pool':safe})
with contextlib.redirect_stderr(io.StringIO()):sid=submit(b.decode())
save('S2-submit.json',{'sid':sid,'sha256':sha,'channel':'web','start_at':start,'submitted_at':datetime.datetime.now().isoformat(),'source':str(p)});print('SID',sid,'SHA',sha)
