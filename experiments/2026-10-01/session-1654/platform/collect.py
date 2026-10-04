from control import *
for sid in [153072,153115,153123,153131,153137,153151]:
 d=read('submission/getSubmissionDetail',{'submissionId':str(sid),'locale':'zh_CN'})
 v=parse_detail(d);code=(d.get('content')or{}).get('code')
 if sid in [153072,153115,153123,153131] and code:
  p=O/f'{sid}-source.py';p.write_bytes(code.encode());py_compile.compile(str(p),cfile=str(O/f'{sid}.pyc'),doraise=True)
 d.pop('content',None);save(f'{sid}-raw.json',d)
 v['sid']=sid;v['checked_at']=datetime.datetime.now().isoformat()
 v['zero_cases']=[k for k,c in v['cases'].items() if c['metrics']['tk_time_ms']==0]
 v['full_ac_audit']=v['status']=='Accepted' and len(v['cases'])==12 and all(c['status']=='Accepted' and c['metrics'].get('pass') and len(c['sqnr_db'])>=2 and min(c['sqnr_db'])>=22 and c['determinism_checks_passed']>=2 for c in v['cases'].values())
 v['net']=round(v['displayScore']-10,2) if v['displayScore'] is not None else None
 save(f'{sid}-audit.json',v);print(sid,v['status'],v['displayScore'],v['full_ac_audit'])
p=O/'snapshot.json';s=json.loads(p.read_text());m=s['me'];s['me']={k:m.get(k) for k in ('rank','totalScore','problemScores')};save('snapshot.json',s)
(O/'snapshot-output.json').unlink(missing_ok=True)
