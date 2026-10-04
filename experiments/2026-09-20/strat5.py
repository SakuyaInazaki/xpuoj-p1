"""Single-sided companion to strat4: shared machine slope, VERSION-SPECIFIC intercept.

b is fit once on the large pre-v841 anchor pool (the slope is a bandwidth property of the case and is
not changed by a quantizer launch tweak). Each version pool then contributes only its own intercept
a_v = mean(tk_anchor - b*M_anchor). A candidate's residual is measured off its own version's line.
Caveat, stated because it matters: the v841 intercept rests on few anchors, so all v841 candidate
residuals share that intercept's standard error as a common-mode term. strat4 (paired) has no such term.
Usage: strat5.py <cand:anchor,...>
"""
import sys, json, os, statistics as st
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knobs6_data')
db = json.load(open(os.path.join(H, 'tk.json')))
def rd(f):
    p = os.path.join(H, f)
    return [s for s in open(p).read().split() if s in db] if os.path.exists(p) else []
old = rd('v837.txt') + rd('v838.txt'); new = rd('v841.txt')
cases = [str(i) for i in (1, 2, 3, 5, 7, 8, 9, 10)]
base = {c: st.mean(db[s][c][0] for s in old + new) for c in cases}
def M(s): return st.mean(db[s][c][0] / base[c] for c in cases)
pairs = [p.split(':') for p in sys.argv[1].split(',') if p]
for t in ('11', '12'):
    A = [(M(s), db[s][t][0]) for s in old]
    mM = st.mean(x for x, _ in A); mY = st.mean(y for _, y in A)
    b = sum((x-mM)*(y-mY) for x, y in A) / sum((x-mM)**2 for x, _ in A)
    a = {}; res = []
    for tag, pool in (('old', old), ('new', new)):
        a[tag] = st.mean(db[s][t][0] - b*M(s) for s in pool)
        res += [db[s][t][0] - a[tag] - b*M(s) for s in pool]
    sd = st.stdev(res)
    se_new = st.stdev([db[s][t][0] - b*M(s) for s in new]) / len(new)**.5 if len(new) > 1 else float('nan')
    print(f"c{t}: b={b:.4f}  a_old={a['old']:.4f} (n={len(old)})  a_new={a['new']:.4f} (n={len(new)}, "
          f"se={100*se_new/mY:.3f}%)  intercept shift {100*(a['new']-a['old'])/mY:+.3f}%  "
          f"resid sd={sd:.4f} ms ({100*sd/mY:.3f}%)")
    out = []
    for c, an in pairs:
        tag = 'new' if an in new else 'old'
        r = db[c][t][0] - a[tag] - b*M(c); out.append(r)
        print(f"  {c} [{tag}]: tk={db[c][t][0]:.4f} resid={r:+.4f} ms ({100*r/mY:+.3f}%) z={r/sd:+.2f}")
    if len(out) > 1:
        m = st.mean(out)
        print(f"  MEAN resid={m:+.4f} ms ({100*m/mY:+.3f}%)  z_of_mean={m/(sd/len(out)**.5):+.2f}  "
              f"n={len(out)}  negative {sum(1 for r in out if r < 0)}/{len(out)}")
