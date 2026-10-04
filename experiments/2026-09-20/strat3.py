"""knobs10 estimator.  Every knobs10 candidate touches c11 AND c12 (same aux
launches), so the machine index M excludes {4, 6, 11, 12}:
  - 4 and 6 because the pooled anchors span v837/v838/v839/v840, which differ
    only at the c6 dn (GROUP_M, num_stages) and the c4 dn num_stages;
  - 11/12 because they are the touched cases.
c11/c12 code is byte-identical across v837..v840, so all anchors pool.
Usage: strat3.py <cand_sids,comma>
"""
import sys, json, os, statistics as st
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knobs6_data')
db = json.load(open(os.path.join(H, 'tk.json')))
anch = [s for s in (open(os.path.join(H,'v837.txt')).read().split()
                    + open(os.path.join(H,'v838.txt')).read().split()) if s in db]
cands = [x for x in sys.argv[1].split(',') if x] if len(sys.argv) > 1 else []
cases = [str(i) for i in (1,2,3,5,7,8,9,10)]
base = {c: st.mean(db[s][c][0] for s in anch) for c in cases}
def M(s): return st.mean(db[s][c][0] / base[c] for c in cases)
for t in ('11', '12'):
    A = [(M(s), db[s][t][0]) for s in anch]
    mM = st.mean(x for x, _ in A); mY = st.mean(y for _, y in A)
    b = sum((x-mM)*(y-mY) for x, y in A) / sum((x-mM)**2 for x, _ in A); a = mY - b*mM
    res = [y - (a + b*x) for x, y in A]; sd = st.stdev(res)
    rawsd = st.stdev([y for _, y in A])
    tgt = 86 if t == '11' else 84
    print(f"c{t}: anchors n={len(A)}  tk = {a:.4f} + {b:.4f}*M   resid sd={sd:.4f} ms "
          f"({100*sd/mY:.3f}%)  [raw sd {rawsd:.4f}, M explains {100*(1-(sd/rawsd)**2):.0f}%]")
    out = []
    for s in cands:
        if s not in db: print('  missing', s); continue
        x = M(s); y = db[s][t][0]; r = y - (a + b*x); out.append(r)
        tb = db[s][t][1]; sc = int(100*tb/(tb+y))
        need = tb*(100-tgt)/tgt
        print(f"  {s}: M={x:.4f} tk={y:.4f} pred={a+b*x:.4f} resid={r:+.4f} ms "
              f"({100*r/(a+b*x):+.3f}%) z={r/sd:+.2f} | pts={sc} need<={need:.4f}(for {tgt}) "
              f"crossed={'YES' if y<=need else 'no'}")
    if len(out) > 1:
        mr = st.mean(out)
        print(f"  MEAN resid={mr:+.4f} ms ({100*mr/mY:+.3f}%)  z_of_mean={mr/(sd/len(out)**.5):+.2f}  "
              f"n={len(out)}  negative {sum(1 for r in out if r<0)}/{len(out)}")
