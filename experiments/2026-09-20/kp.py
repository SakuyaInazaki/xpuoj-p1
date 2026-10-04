"""kp.py <cand_sid> <anchor_sid> <touched_case>  — cache the pair, register the anchor, print mnorm + strat2."""
import sys, os, json, subprocess
sys.path.insert(0, '/Users/sakimi/Desktop/xpuoj-p1/scripts')
H = '/Users/sakimi/Desktop/xpuoj-p1/experiments/2026-09-20/knobs6_data'
SC = '/tmp/claude-501/-Users-sakimi/ae83d52e-5b78-411d-a55b-02ef354ac22d/scratchpad/knobs6'
cand, anch, t = sys.argv[1], sys.argv[2], sys.argv[3]
subprocess.run([sys.executable, SC + '/cache.py', cand, anch], capture_output=True)
p = SC + '/v838.txt'; lst = open(p).read().split()
if anch not in lst:
    lst.append(anch); open(p, 'w').write(' '.join(lst))
for f in ('tk.json', 'v838.txt', 'v837.txt'):
    open(H + '/' + f, 'w').write(open(SC + '/' + f).read())
from xpuoj_pow import Client
from cases import fetch
import mnorm
c = Client()
m, d, resid, s = mnorm.verdict(int(cand), int(anch), [int(t)], c)
_, v = fetch(int(cand), c); _, a = fetch(int(anch), c)
bad = [i for i in range(1, 13) if v[i]['sqnr'] != a[i]['sqnr']]
tb = v[t if False else int(t)]['tb']; tk = v[int(t)]['tk']
sc = int(100*tb/(tb+tk)); need = tb*(100-(sc+1))/(sc+1)
print(f"mnorm m={m:+.3f}%  c{t} d={d[int(t)]:+.3f}% resid={resid[int(t)]:+.3f}%  "
      f"raw {a[int(t)]['tk']:.4f}->{tk:.4f}  SQNR {'EQUAL' if not bad else 'MISMATCH '+str(bad)}  "
      f"pts {sc}/{int(100*a[int(t)]['tb']/(a[int(t)]['tb']+a[int(t)]['tk']))} need<={need:.4f} "
      f"crossed={'YES' if tk<=need else 'no'}")
