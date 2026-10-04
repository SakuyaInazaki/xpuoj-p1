"""dM-adjusted paired estimator (knobs10 x13). Regress the paired D on the pair's own dM and take
the INTERCEPT as the effect. This is robust to (a) any residual misspecification of the shared machine
slope b and (b) imbalance in the pairs' dM, both of which leak straight into a plain mean of D.
Usage: strat6.py <cand:anchor,...>
"""
import sys, json, os, statistics as st, math
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
    X = [M(c)-M(a) for c, a in pairs]
    Y = [(db[c][t][0]-db[a][t][0]) - b*(M(c)-M(a)) for c, a in pairs]
    n = len(X); mx = st.mean(X); my = st.mean(Y)
    sxx = sum((x-mx)**2 for x in X)
    s1 = sum((x-mx)*(y-my) for x, y in zip(X, Y)) / sxx
    a0 = my - s1*mx
    r = [y - (a0 + s1*x) for x, y in zip(X, Y)]
    se_res = math.sqrt(sum(v*v for v in r)/(n-2))
    se_a0 = se_res*math.sqrt(1.0/n + mx*mx/sxx)
    corr = st.correlation(X, Y)
    print(f"c{t}: n={n}  plain mean D={my:+.5f} ms ({100*my/mY:+.3f}%)   mean dM={mx:+.5f} (imbalance)")
    print(f"     corr(dM, D)={corr:+.2f}   extra slope on dM={s1:+.3f}")
    print(f"     dM-ADJUSTED effect (intercept) = {a0:+.5f} ms ({100*a0/mY:+.3f}%)  "
          f"se={100*se_a0/mY:.3f}%  t={a0/se_a0:+.2f}  negative {sum(1 for y in Y if y<0)}/{n}")
