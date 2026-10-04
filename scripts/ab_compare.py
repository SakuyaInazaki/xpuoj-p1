#!/usr/bin/env python3
"""Compare per-case tk/tb between two submissions (same-window A/B pair).

Usage: python3 scripts/ab_compare.py <candidate_id> <control_id>
Prints per-case tk diff (negative = candidate faster) and sum tk.
"""
import sys, json, base64, re
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from xpuoj_api import Client


def fetch(c, sid):
    d = c.get_detail(sid).json()
    rows = {}
    tcr = d.get("progress", {}).get("testcaseResult", {}) or {}
    for _, v in tcr.items():
        m = re.search(r"OJRESULT v1 \S+ (\S+)", v.get("userOutput", "") or "")
        if not m:
            continue
        j = json.loads(base64.b64decode(m.group(1)))
        cm = re.search(r"(\d+)", v.get("input", "") or "")
        case = int(cm.group(1)) if cm else len(rows) + 1
        rows[case] = (j["tk_time_ms"], j["tb_time_ms"], v.get("displayScore"))
    return rows


def main():
    a, b = sys.argv[1], sys.argv[2]
    c = Client()
    ra, rb = fetch(c, a), fetch(c, b)
    if not ra or not rb:
        print(f"missing data: A has {len(ra)} cases, B has {len(rb)} cases")
        return
    print(f"case |  A({a}) tk |  B({b}) tk |  diff A-B |     A tb |     B tb | sA | sB")
    sum_a = sum_b = 0.0
    for case in sorted(set(ra) & set(rb)):
        ka, ba, sa = ra[case]
        kb, bb, sb = rb[case]
        sum_a += ka
        sum_b += kb
        flag = " <-- " if abs(ka - kb) > 0.05 else ""
        print(f"{case:4d} | {ka:12.3f} | {kb:12.3f} | {ka-kb:+9.3f} | {ba:8.3f} | {bb:8.3f} | {sa} | {sb}{flag}")
    print(f"sum tk: A={sum_a:.3f}  B={sum_b:.3f}  diff={sum_a-sum_b:+.3f} ms")


if __name__ == "__main__":
    main()
