"""解析剖析提交回传的 [PROF] 行 → 逐 case 逐内核耗时表。
用法: python3 scripts/prof.py <SID>"""
import sys, re, collections
sys.path.insert(0, 'scripts')
from xpuoj_pow import Client

K12 = [(16384,4096,8,8192,2),(16384,4096,8,14336,2),(16384,2048,32,2048,4),(16384,2048,32,1024,4),
       (8192,3584,64,2560,8),(8192,3584,64,1024,8),(16384,4096,96,2048,3),(16384,4096,96,1024,3),
       (4096,4096,256,2048,8),(4096,4096,256,1536,8),(65536,1024,32,1024,2),(65536,1024,32,2048,2)]

def main(sid):
    c = Client(); p = c.get_detail(sid).json().get('progress') or {}
    tcr = p.get('testcaseResult')
    it = list(tcr.items()) if isinstance(tcr, dict) else list(enumerate(tcr or []))
    per = {}
    for n, (h, tc) in enumerate(it):
        d = (tc.get('userError') or {}); d = d.get('data','') if isinstance(d,dict) else str(d)
        tkm = re.search(r'"tk_time_ms":([\d.]+)', d)
        tk = float(tkm.group(1)) if tkm else None
        best = None
        for line in re.findall(r'\[PROF\][^\n]*', d):
            kv = dict(re.findall(r'(\w+)=([-\d.]+|\w+)', line))
            tot = float(kv.get('tot', 1e9))
            if best is None or tot < float(best.get('tot', 1e9)): best = kv
        if best: per[n] = (tk, best, d)
    if not per:
        print("没有 [PROF] 行。前 800 字符：")
        for n,(h,tc) in enumerate(it[:2]):
            d=(tc.get('userError') or {}); d=d.get('data','') if isinstance(d,dict) else str(d)
            print(f"--tc{n} {tc.get('status')}:", d[:800])
        return
    keys = []
    for _, b, _ in per.values():
        for k in b:
            if k not in ('n','T','H','E','I','k','tot','dnq','br') and k not in keys: keys.append(k)
    hdr = "  shape(T,H,E,I,k)          tk  " + " ".join(f"{k:>7}" for k in keys) + "    tot  dnq br"
    print(hdr); print("-"*len(hdr))
    agg = collections.Counter()
    for n in sorted(per):
        tk, b, _ = per[n]
        sh = f"({b.get('T')},{b.get('H')},{b.get('E')},{b.get('I')},{b.get('k')})"
        row = " ".join(f"{float(b.get(k,0)):7.3f}" for k in keys)
        for k in keys: agg[k] += float(b.get(k,0))
        print(f"{sh:>26} {tk if tk else 0:6.3f}  {row} {float(b.get('tot',0)):6.3f}  {b.get('dnq','?'):>3} {b.get('br','?'):>3}")
    print("-"*len(hdr))
    print(f"{'Σ':>26} {'':6}  " + " ".join(f"{agg[k]:7.3f}" for k in keys))
    tt = sum(agg.values())
    print(f"{'占比%':>26} {'':6}  " + " ".join(f"{agg[k]/tt*100:6.1f}%" for k in keys))

main(int(sys.argv[1]))
