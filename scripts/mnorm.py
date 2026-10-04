"""逐案机器签名回归：机器差不是均匀缩放，compute-bound 案(c1/c2/c3/c7)被拖慢 ~2-2.9%，
memory-bound 案(c8/c11/c12)几乎为 0。因此「未触碰案取平均」并不能替触碰案消掉机器项。
做法：用两次同码锚对拿到签名 s_i，再对每个变体在未触碰案上最小二乘拟合 m，
把 m*s_i 从触碰案里减掉，残差才是真实效应。"""
import sys
sys.path.insert(0,'scripts')
from xpuoj_pow import Client
from cases import fetch

SIG_PAIRS = [(138154, 138157), (138164, 138166)]   # 同码锚，不同机

def signature(c=None):
    c = c or Client()
    acc = {i: [] for i in range(1, 13)}
    for lo, hi in SIG_PAIRS:
        _, a = fetch(lo, c); _, b = fetch(hi, c)
        for i in range(1, 13):
            acc[i].append((b[i]['tk'] / a[i]['tk'] - 1) * 100)
    s = {i: sum(v) / len(v) for i, v in acc.items()}
    m = sum(s.values()) / 12
    return {i: s[i] / m for i in s}          # 归一到均值 1

def verdict(var_sid, anchor_sid, touched, c=None, s=None):
    c = c or Client(); s = s or signature(c)
    _, v = fetch(var_sid, c); _, a = fetch(anchor_sid, c)
    d = {i: (v[i]['tk'] / a[i]['tk'] - 1) * 100 for i in range(1, 13)}
    ref = [i for i in range(1, 13) if i not in touched]
    num = sum(d[i] * s[i] for i in ref); den = sum(s[i] * s[i] for i in ref)
    m = num / den if den else 0.0
    resid = {i: d[i] - m * s[i] for i in range(1, 13)}
    return m, d, resid, s

if __name__ == "__main__":
    c = Client(); s = signature(c)
    print("机器签名(均值归一):", " ".join(f"c{i}{s[i]:+.2f}" for i in range(1, 13)))
