"""pooled.py  — pooled same-code raw comparison per case.
Usage: pooled.py <case> anchors: a,b,c  cands: d,e
"""
import sys, statistics as st
sys.path.insert(0, '/Users/sakimi/Desktop/xpuoj-p1/scripts')
from xpuoj_pow import Client
from cases import fetch

case = int(sys.argv[1])
anch = [int(x) for x in sys.argv[2].split(',')]
cand = [int(x) for x in sys.argv[3].split(',')]
c = Client()
def tks(sids):
    out = []
    for s in sids:
        try:
            _, o = fetch(s, c)
            if case in o: out.append(o[case]['tk'])
        except Exception as e: print('skip', s, type(e).__name__)
    return out
A = tks(anch); C = tks(cand)
ma, sa = st.mean(A), (st.stdev(A) if len(A) > 1 else float('nan'))
mc = st.mean(C)
print(f"c{case} anchors n={len(A)} mean={ma:.4f} sd={sa:.4f}  {['%.4f'%x for x in A]}")
print(f"c{case} cand    n={len(C)} mean={mc:.4f}  {['%.4f'%x for x in C]}")
se = sa / (len(A) ** .5 + 0) if sa == sa else float('nan')
z = (mc - ma) / sa if sa == sa and sa > 0 else float('nan')
print(f"delta={mc-ma:+.4f} ms ({100*(mc-ma)/ma:+.3f}%)  z_vs_anchor_sd={z:+.2f}")
