"""anz.py <cand_sid> <anchor_sid> <touched_case_idx>
Prints per-case delta, mnorm residual, SQNR equality, and next-point crossing."""
import sys
sys.path.insert(0, '/Users/sakimi/Desktop/xpuoj-p1/scripts')
from xpuoj_pow import Client
from cases import fetch
import mnorm

cand, anch, t = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
c = Client()
m, d, resid, s = mnorm.verdict(cand, anch, [t], c)
dsv, v = fetch(cand, c)
dsa, a = fetch(anch, c)
print(f"cand {cand} display={dsv}   anchor {anch} display={dsa}")
print(f"machine term m = {m:+.3f}%")
print("case  d%      resid%   sig    sqnr_c/sqnr_a  sc_c/sc_a")
for i in range(1, 13):
    mark = " <==" if i == t else ""
    print(f"c{i:<3} {d[i]:+7.3f} {resid[i]:+7.3f}  {s[i]:+5.2f}  "
          f"{v[i]['sqnr']}/{a[i]['sqnr']}  "
          f"{int(100*v[i]['tb']/(v[i]['tb']+v[i]['tk']))}/{int(100*a[i]['tb']/(a[i]['tb']+a[i]['tk']))}{mark}")
dt_ms = v[t]['tk'] - a[t]['tk']
print(f"\ntouched c{t}: tk {a[t]['tk']:.4f} -> {v[t]['tk']:.4f}  ({dt_ms:+.4f} ms raw)")
print(f"  resid {resid[t]:+.3f}%  => resid ms {a[t]['tk']*resid[t]/100:+.4f}")
for tag, r in (("cand", v[t]), ("anch", a[t])):
    sc = int(100 * r['tb'] / (r['tb'] + r['tk'])); tb = r['tb']
    need = tb * (100 - (sc + 1)) / (sc + 1)
    print(f"  {tag}: score={sc} tb={tb:.4f} tk={r['tk']:.4f} need<= {need:.4f} "
          f"(gap {r['tk']-need:+.4f} ms) crossed={'YES' if r['tk'] <= need else 'no'}")
bad = [i for i in range(1, 13) if v[i]['sqnr'] != a[i]['sqnr']]
print("SQNR mismatch cases:", bad if bad else "none")
