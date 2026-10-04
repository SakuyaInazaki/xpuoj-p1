"""逐 testcase 读 SQNR。Accepted 的提交也带这行，所以可以把它当精度仪表用：
基线余量约 1.1~1.9 dB（23.12~23.87 dB 对 22.0 阈值）。"""
import sys, re
sys.path.insert(0, 'scripts')
from xpuoj_pow import Client


def sqnr(sid, c=None):
    c = c or Client()
    p = c.get_detail(sid).json().get('progress') or {}
    tcr = p.get('testcaseResult')
    it = list(tcr.items()) if isinstance(tcr, dict) else list(enumerate(tcr or []))
    out = []
    for h, tc in it:
        d = (tc.get('userError') or {})
        d = d.get('data', '') if isinstance(d, dict) else str(d)
        out.append((tc.get('status'), [float(x) for x in re.findall(r'SQNR=([\d.]+) dB', d)]))
    return out


if __name__ == "__main__":
    c = Client()
    for sid in sys.argv[1:]:
        rows = sqnr(int(sid), c)
        print(f"=== {sid} ===")
        worst = 99
        for n, (st, v) in enumerate(rows):
            tag = "样例" if n == 0 else f"c{n}"
            if v: worst = min(worst, min(v))
            print(f"  {tag:>4}: {st:<20} " + " ".join(f"{x:.2f}" for x in v))
        print(f"  最差余量 {worst - 22.0:+.2f} dB")
