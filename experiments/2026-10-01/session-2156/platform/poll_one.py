import sys,json,datetime,re
from pathlib import Path
R=Path.cwd();O=R/'experiments/2026-10-01/session-2156/platform';sys.path[:0]=[str(R/'scripts'),str(R/'reports')]
from xpuoj_web import API,_session
from collect_timing_anomalies_20260930 import parse_detail,TERMINAL
sid=int(sys.argv[1]);s=_session()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(e,b):
 r=s.post(API+e,json=b,timeout=25);r.raise_for_status();return r.json()
d=read('submission/getSubmissionDetail',{'submissionId':str(sid),'locale':'zh_CN'});v=parse_detail(d);code=(d.get('content')or{}).get('code');d.pop('content',None);d.pop('progressSubscriptionKey',None)
prior_path=O/f'{sid}-audit.json';prior=json.loads(prior_path.read_text()) if prior_path.exists() else {};save(f'{sid}-raw.json',d)
v['sid']=sid;v['checked_at']=datetime.datetime.now().isoformat();base=json.loads((R/'reports/2026-10-01-cohort-readonly.json').read_text())['submissions']['151806']['cases']
v['zero_cases']=sorted(int(k) for k,c in v['cases'].items() if c['metrics']['tk_time_ms']==0)
v['low_cases']=sorted(int(k) for k,c in v['cases'].items() if c['metrics']['tk_time_ms']<base[k]['metrics']['tk_time_ms']*.5)
v['full_ac_audit']=v['status']=='Accepted' and len(v['cases'])==12 and all(c['status']=='Accepted' and c['metrics'].get('pass') and len(c['sqnr_db'])>=2 and min(c['sqnr_db'])>=22 and c['determinism_checks_passed']>=2 for c in v['cases'].values())
v['net']=round(v['displayScore']-10,2) if v['displayScore'] is not None else None
if v['status'] in TERMINAL:
 v['first_terminal_observed_at']=prior.get('first_terminal_observed_at') or (prior.get('checked_at') if prior.get('status') in TERMINAL else v['checked_at'])
 if prior.get('wall_observations'):v['wall_observations']=prior['wall_observations']
 if code:(O/f'{sid}-source.py').write_bytes(code.encode())
for p in O.glob('*-submit.json'):
 a=json.loads(p.read_text())
 if str(a.get('sid'))==str(sid):
  v['submitted_source_match']=v['source_sha256']==a['sha256']
  observed=v.get('first_terminal_observed_at') or v['checked_at']
  v['wall_since_submit_s']=(datetime.datetime.fromisoformat(observed)-datetime.datetime.fromisoformat(a['submitted_at'])).total_seconds()
v['visible_log_time_markers']=[{'case':int(m.group(2)),'per_rank_directory_stamp':m.group(1)} for log in v['logs'] for m in re.finditer(r'/run-logs/(\d{8}_\d{6})_tc(\d+)_',log['text'])]
save(f'{sid}-audit.json',v);print(v['checked_at'],sid,v['status'],v['displayScore'],len(v['cases']),v['zero_cases'],v['low_cases'],v['full_ac_audit'])
if v['status'] in TERMINAL:
 print('SHA',v['source_sha256'],'match',v.get('submitted_source_match'),'wall',v.get('wall_since_submit_s'))
 for k,c in sorted(v['cases'].items(),key=lambda x:int(x[0])):print(k,c)
if v['full_ac_audit'] and v['displayScore']>89.75:
 e=read('contest/play/getContestScoreboardMe',{'contestId':13}).get('entry')or{};m={k:e.get(k) for k in ('rank','totalScore','problemScores')};save(f'{sid}-scoreboard.json',m);print('SCOREBOARD',m)
