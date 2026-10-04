import json,re
from pathlib import Path
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1'); OUT=Path(__file__).resolve().parent
patterns={
 'range_counter':r'Out[ _-]?of[ _-]?range',
 'range_detail':r'TraceActivity outside of profiling window|Profile time range:',
 'order_counter':r'CPU GPU out[ _-]?of[ _-]?order',
 'order_warning':r'GPU op timestamp|wrong order',
 'kineto_record_counts':r'Record counts:|Processed \d+ GPU records|CuptiActivityProfiler\.cpp|GenericActivityProfiler\.cpp',
 'trace_empty':r'GPU trace is empty|CPU trace is empty|No activities found',
 'cycle_warning':r'Profiler clears events at the end of each cycle',
}
def strings(v):
 if isinstance(v,str):yield v
 elif isinstance(v,dict):
  for a in v.values():yield from strings(a)
 elif isinstance(v,list):
  for a in v:yield from strings(a)
cohort=json.loads((ROOT/'reports/2026-10-01-cohort-readonly.json').read_text())
result={'cohort_count':len(cohort['submissions']),'cohort_hits':{},'recent_raw_files':[],'recent_hits':{},'legacy_matches':[]}
for label,pat in patterns.items():
 rx=re.compile(pat,re.I)
 result['cohort_hits'][label]=[sid for sid,v in cohort['submissions'].items() if any(rx.search(s) for s in strings(v))]
for p in sorted((ROOT/'experiments/2026-10-01/session-1654/platform').glob('*-raw.json')):
 result['recent_raw_files'].append(str(p.relative_to(ROOT)))
 v=json.loads(p.read_text())
 for label,pat in patterns.items():
  if any(re.search(pat,s,re.I) for s in strings(v)):result['recent_hits'].setdefault(label,[]).append(p.name)
# Original log/result files only; exclude new note and prose summaries.
rx=re.compile('|'.join(patterns[k] for k in patterns if k!='cycle_warning'),re.I)
for folder in ['logs','experiments']:
 for p in (ROOT/folder).rglob('*'):
  if OUT in p.parents or p.suffix not in ['.log','.txt','.json'] or not p.is_file():continue
  text=p.read_text(errors='replace')
  if rx.search(text):result['legacy_matches'].append(str(p.relative_to(ROOT)))
(OUT/'scan-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
