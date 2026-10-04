"""Version-agnostic PAIRED estimator for c11/c12 (knobs10, x13 across v840 and v841).

Why paired: the v841 anchor differs from the v840 anchor at the quantizer, which is a c11/c12
kernel, so the anchor regression INTERCEPT moves between versions. Taking the candidate minus its
own same-version anchor cancels the intercept exactly:
    D = (tk_c - a - b*M_c) - (tk_a - a - b*M_a) = (tk_c - tk_a) - b*(M_c - M_a)
Only the machine slope b survives, and b is a property of the case's bandwidth profile, not of the
quantizer launch, so it is shared. b is taken from the large pre-v841 anchor regression.
M uses cases {1,2,3,5,7,8,9,10}: c4/c6 differ across v837..v840, c11/c12 are the touched cases.

Null sd is EMPIRICAL: the same D formed from disjoint anchor-anchor pairs within each version pool.

Usage: strat4.py <cand:anchor,cand:anchor,...>
"""
import sys, json, os, statistics as st
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knobs6_data')
db = json.load(open(os.path.join(H, 'tk.json')))
def rd(f):
    p = os.path.join(H, f)
    return [s for s in open(p).read().split() if s in db] if os.path.exists(p) else []
old = rd('v837.txt') + rd('v838.txt')      # v837/v838/v839/v840 anchors
new = rd('v841.txt')                        # v841 anchors
allanch = old + new
cases = [str(i) for i in (1, 2, 3, 5, 7, 8, 9, 10)]
base = {c: st.mean(db[s][c][0] for s in allanch) for c in cases}
def M(s): return st.mean(db[s][c][0] / base[c] for c in cases)

pairs = [p.split(':') for p in sys.argv[1].split(',') if p]
for t in ('11', '12'):
    A = [(M(s), db[s][t][0]) for s in old]
    mM = st.mean(x for x, _ in A); mY = st.mean(y for _, y in A)
    b = sum((x-mM)*(y-mY) for x, y in A) / sum((x-mM)**2 for x, _ in A)
    def D(c, a): return (db[c][t][0]-db[a][t][0]) - b*(M(c)-M(a))
    null = []
    for pool in (old, new):
        for i in range(0, len(pool)-1, 2):
            null.append(D(pool[i], pool[i+1]))
    sd = st.stdev(null) if len(null) > 2 else float('nan')
    print(f"c{t}: slope b={b:.4f} (from {len(old)} pre-v841 anchors); "
          f"empirical paired null: n={len(null)} mean={st.mean(null):+.5f} sd={sd:.4f} ms ({100*sd/mY:.3f}%)")
    out = []
    for c, a in pairs:
        if c not in db or a not in db: print('  missing', c, a); continue
        d = D(c, a); out.append(d)
        tb = db[c][t][1]; sc = int(100*tb/(tb+db[c][t][0]))
        tgt = 86 if t == '11' else 84
        need = tb*(100-tgt)/tgt
        print(f"  {c}/{a}: tk {db[a][t][0]:.4f}->{db[c][t][0]:.4f}  dM={M(c)-M(a):+.4f}  "
              f"D={d:+.4f} ms ({100*d/db[a][t][0]:+.3f}%) z={d/sd:+.2f} | pts={sc} "
              f"crossed={'YES' if db[c][t][0] <= need else 'no'}")
    if len(out) > 1:
        m = st.mean(out)
        print(f"  MEAN D={m:+.4f} ms ({100*m/mY:+.3f}%)  z_of_mean={m/(sd/len(out)**.5):+.2f}  "
              f"n={len(out)}  negative {sum(1 for d in out if d < 0)}/{len(out)}")
