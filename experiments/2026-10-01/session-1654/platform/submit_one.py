from control import *
import contextlib,io,subprocess
from pool_client import status
from xpuoj_pow import Client
PLANS={'R151':('f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a','153151-source.py'),'A1':('08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9','A1-08dd08eb51b1.py'),'B1':('05828fef7359fd70159046bc026b414219d7fd28559146cd8793799aaaa15fa7','B1-05828fef7359.py')}
name=sys.argv[1];sha,filename=PLANS[name]
if (O/f'{name}-submit.json').exists():raise SystemExit('Already has submit record; inspect SID, never repeat')
b=read('contest/play/querySubmissions',{'contestId':13,'problemOrder':1,'takeCount':10,'locale':'zh_CN'})
for r in b.get('submissions',[]):
 d=read('submission/getSubmissionDetail',{'submissionId':str(r['id']),'locale':'zh_CN'});v=parse_detail(d)
 if v['status'] not in TERMINAL:raise SystemExit(f'In flight {r["id"]}: {v["status"]}')
 if int(r['id'])>153151 and not any(json.loads(p.read_text()).get('sid')==r['id'] for p in O.glob('*-submit.json')):raise SystemExit(f'Unrecognized concurrent SID {r["id"]}')
if name=='B1':
 a=json.loads((O/'A1-terminal-audit.json').read_text());assert a['full_ac_audit'] and not a['zero_cases'] and not a['low_cases']
p=O/filename;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
assert hashlib.sha256((R/'p1/kernel.py').read_bytes()).hexdigest()=='08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9'
py_compile.compile(str(p),cfile=str(O/f'{name}.pyc'),doraise=True)
q=status();safe={k:q.get(k) for k in ('browser_online','fresh','need','target')};save(f'{name}-preflight.json',{'at':datetime.datetime.now().isoformat(),'sha256':sha,'pool':safe,'me':{k:v for k,v in (read('contest/play/getContestScoreboardMe',{'contestId':13}).get('entry')or{}).items() if k in ('rank','totalScore','problemScores')}})
# Suppress pool_client's token tail diagnostics; no secret reaches disk or stdout.
channel=sys.argv[2] if len(sys.argv)>2 else 'web'
if channel=='api':
 client=Client();credit=client.credit();save(f'{name}-credit-before.json',credit);assert credit['api_token']['available']>=1
 sid=client.submit(raw.decode())
else:
 with contextlib.redirect_stderr(io.StringIO()):sid=submit(raw.decode())
save(f'{name}-submit.json',{'sid':sid,'channel':channel,'sha256':sha,'frozen_source':str(p),'submitted_at':datetime.datetime.now().isoformat()});print(name,'SID',sid,'SHA',sha)
