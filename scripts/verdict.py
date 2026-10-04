"""扣掉机器项的配对评判：python3 scripts/verdict.py <变体SID> <锚SID> [标签]
残差 = 相对 #76@快机(138610) 的真实效应，正号=变慢。"""
import sys
sys.path.insert(0, 'scripts')
from xpuoj_pow import Client
from cases import fetch
from mnorm import signature

SEN = [0, 3.4, 2.1, 9.5, 14.0, 5.1, 9.3, 6.4, 9.6, 6.9, 8.7, 11.5, 8.1]


def resid(sid, c, s, base):
    o = fetch(sid, c)[1]
    if len(o) < 12:
        raise SystemExit(f"SID {sid} 只有 {len(o)} 个 case 的结果（未 Accepted），无法评判")
    d = {i: (o[i]['tk'] / base[i]['tk'] - 1) * 100 for i in range(1, 13)}
    m = sum(d[i] * s[i] for i in range(1, 13)) / sum(s[i] * s[i] for i in range(1, 13))
    r = {i: d[i] - m * s[i] for i in range(1, 13)}
    return o, m, r


def main(vs, as_, tag):
    c = Client(); s = signature(c); base = fetch(138610, c)[1]
    ov, mv, rv = resid(vs, c, s, base)
    oa, ma, ra = resid(as_, c, s, base)
    dv, _ = fetch(vs, c)[0], None
    print(f"{tag} sid={vs} display={fetch(vs,c)[0]} Σtk={sum(ov[i]['tk'] for i in ov):.3f} m={mv:+.2f}%")
    print(f"锚   sid={as_} display={fetch(as_,c)[0]} Σtk={sum(oa[i]['tk'] for i in oa):.3f} m={ma:+.2f}%")
    print("\n case  锚残差  变体残差   净效应    分")
    tot = 0.0
    for i in range(1, 13):
        net = rv[i] - ra[i]
        g = -net / 100 * base[i]['tk'] * SEN[i] / 12
        tot += g
        print(f" c{i:<3d} {ra[i]:+6.2f}% {rv[i]:+6.2f}%  {net:+6.2f}%  {g:+.3f}")
    print(f" 净 raw {tot:+.3f}   (噪声地板约 ±0.6%/案)")
    if not (-0.5 <= mv <= 2.0) or not (-0.5 <= ma <= 2.0):
        print(" ⚠ 机器项超出两簇范围，残差不可信，这一发只能重做")
    elif abs(mv - ma) > 0.9:
        print(" ⚠ 变体与锚不在同一台机器（m 差 %.2f%%），原始 tk 不可直接比，但残差已扣机器项" % (mv - ma))


main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "变体")
