"""Machine-stratified pooled estimator.
For touched case t: machine index M(run) = mean over untouched cases of tk_i / anchor_mean_i.
Fit anchors: tk_t = a + b*M (OLS). Report each candidate's residual vs that line, in ms, %, and z
(z uses the anchors' residual sd, which removes the machine dimension).
Usage: strat.py <t> <cand_sids,comma>
"""
import sys, json, os, statistics as st
H=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knobs6_data')
db=json.load(open(os.path.join(H,'tk.json')))
V837=open(os.path.join(H,'v837.txt')).read().split()
V838=open(os.path.join(H,'v838.txt')).read().split()
t=sys.argv[1]
cands=[x for x in sys.argv[2].split(',') if x]
anch=[s for s in (V838 if t=='6' else V837+V838) if s in db]
cases=[str(i) for i in range(1,13) if i!=int(t)]
base={c: st.mean(db[s][c][0] for s in anch) for c in cases}
def M(s): return st.mean(db[s][c][0]/base[c] for c in cases)
A=[(M(s), db[s][t][0]) for s in anch]
mM=st.mean(x for x,_ in A); mY=st.mean(y for _,y in A)
b=sum((x-mM)*(y-mY) for x,y in A)/sum((x-mM)**2 for x,_ in A)
a=mY-b*mM
res=[y-(a+b*x) for x,y in A]
sd=st.stdev(res)
print(f"c{t}: anchors n={len(A)}  fit tk = {a:.4f} + {b:.4f}*M   resid sd={sd:.4f} ms ({100*sd/mY:.3f}%)")
print(f"     raw anchor sd={st.stdev([y for _,y in A]):.4f} ms  -> machine index explains "
      f"{100*(1-(sd/st.stdev([y for _,y in A]))**2):.0f}% of variance")
out=[]
for s in cands:
    if s not in db: print('  missing', s); continue
    x=M(s); y=db[s][t][0]; r=y-(a+b*x)
    out.append(r)
    print(f"  {s}: M={x:.4f} tk={y:.4f} pred={a+b*x:.4f} resid={r:+.4f} ms ({100*r/(a+b*x):+.3f}%) z={r/sd:+.2f}")
if len(out)>1:
    mr=st.mean(out)
    print(f"  MEAN resid={mr:+.4f} ms ({100*mr/mY:+.3f}%)  z_of_mean={mr/(sd/len(out)**.5):+.2f}  n={len(out)}")
