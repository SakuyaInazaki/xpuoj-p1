import sys,json,hashlib,datetime,py_compile,contextlib,io
from pathlib import Path
R=Path.cwd();O=R/'experiments/2026-10-01/session-2156/platform';sys.path[:0]=[str(R/'scripts'),str(R/'reports')]
from xpuoj_web import API,_session,submit
from pool_client import status,take
from collect_timing_anomalies_20260930 import parse_detail,TERMINAL
P={'C12':('experiments/2026-10-01/session-2156/s1/p1_s1_c3567_unified.py','1c235efab417aeed0e60f41278d31b874979cc9dbd78e9a780add50812e16d15'),'S1V1':('experiments/2026-10-01/session-1654/s1/p1_s1_c6_v1.py','05828fef7359fd70159046bc026b414219d7fd28559146cd8793799aaaa15fa7'),'C6U':('experiments/2026-10-01/session-2156/s1/p1_s1_c6_unified.py','befd19dcc7a0824ae49c93c91151e051ee6d3d4c8e8da17e2b1fdf40e9a02b68'),'C9':('experiments/2026-10-01/session-2156/combined/p1_c356_unified_c4_layout.py','df5c5f77eadccd9466719d32f8755c381c7686e1dad6d660789d8ba75cac1318'),'COMB':('experiments/2026-10-01/session-2156/combined/p1_c356_unified_c4_flatten.py','79549c5bc0151c940d72ee776cdc571f78edf4ab9740fd0a3f6d0d1fb870bd5e'),'S2b':('experiments/2026-10-01/session-2156/s2/p1_s2_c4_flatten.py','c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b'),'C356U':('experiments/2026-10-01/session-2156/s1/p1_s1_c356_unified.py','75c702478afa8c7d282064fad23d6a542556f30593cae8cb41a201c38da0ea7a'),'E1':('experiments/2026-10-01/session-2156/s1/p1_s1_c3567_vector.py','0328320d02ba2425fe9904b0817cebea28ebfbbd780efc0729e9e642ca7a987b')}
name=sys.argv[1];src,sha=P[name];assert not (O/f'{name}-submit.json').exists();raw=(R/src).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;p=O/f'{name}-{sha[:12]}.py';p.write_bytes(raw);py_compile.compile(str(p),cfile=str(O/f'{name}.pyc'),doraise=True)
s=_session()
def read(e,b):
 r=s.post(API+e,json=b,timeout=25);r.raise_for_status();return r.json()
def save(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
l=read('contest/play/querySubmissions',{'contestId':13,'problemOrder':1,'takeCount':10,'locale':'zh_CN'});count=0
for x in l['submissions']:
 d=read('submission/getSubmissionDetail',{'submissionId':str(x['id']),'locale':'zh_CN'});v=parse_detail(d)
 if v['source_sha256']==sha:raise SystemExit(f'Already has same SHA {x["id"]}')
 if v['status'] not in TERMINAL:count+=1
 if v['status']=='Accepted' and (v['displayScore']or 0)>=90:raise SystemExit(f'New accepted raw90 {x["id"]}')
assert count<7,'Seven already in flight'
q=status();safe={k:q.get(k) for k in ('browser_online','fresh','need','target')};start=datetime.datetime.now().isoformat();save(f'{name}-preflight.json',{'start':start,'sha256':sha,'pool':safe,'visible_inflight':count})
with contextlib.redirect_stderr(io.StringIO()):
 captcha=take('submit_problem',timeout=50)
 sid=submit(raw.decode(),captcha=captcha)
save(f'{name}-submit.json',{'sid':sid,'sha256':sha,'channel':'web','start_at':start,'submitted_at':datetime.datetime.now().isoformat(),'source':str(p)});print(name,'SID',sid,'SHA',sha,flush=True)
