"""一发到底：提交一批变体 + 锚，轮询到出分，再用机器签名回归 + SQNR 仪表出判据。
用法: python3 scripts/run_batch.py "文件:标签:触碰案(逗号分隔或空)" ... --anchor p1/kernel.py
触碰案留空表示全案触碰（此时只报 Σtk 与 SQNR，不做归一化）。"""
import sys, time, json
sys.path.insert(0, 'scripts')
from xpuoj_web import submit
from xpuoj_pow import Client
from cases import fetch
from mnorm import signature
from sqnr import sqnr

SENS = [0, 3.4, 2.1, 9.5, 14.0, 5.1, 9.3, 6.4, 9.6, 6.9, 8.7, 11.5, 8.1]


def main(argv):
    specs, anchor = [], 'p1/kernel.py'
    i = 0
    while i < len(argv):
        if argv[i] == '--anchor':
            anchor = argv[i + 1]; i += 2; continue
        f, tag, touched = argv[i].split(':')
        specs.append((f, tag, set(int(x) for x in touched.split(',') if x))); i += 1
    specs.append((anchor, '锚', set()))

    ids = []
    for f, tag, _ in specs:
        for att in range(5):
            try:
                sid = submit(open(f).read()); ids.append(sid)
                print(f"{tag}: SID={sid}", flush=True); time.sleep(30); break
            except Exception as e:
                print("  retry", att, str(e)[:70], flush=True); time.sleep(70)
        else:
            ids.append(None)

    c = Client(); t0 = time.time()
    busy = ('-', 'Pending', 'Waiting', 'Compiling', 'Running', 'Judging')
    while time.time() - t0 < 1200:
        sts = [(c.get_detail(s).json().get('progress') or {}).get('status') or '-' if s else 'x' for s in ids]
        print(time.strftime("%H:%M:%S"), [x[:4] for x in sts], flush=True)
        if all(x not in busy for x in sts): break
        time.sleep(30)

    res = {}
    for sid, (f, tag, _) in zip(ids, specs):
        if sid is None: continue
        p = c.get_detail(sid).json().get('progress') or {}
        st = p.get('status')
        rows = sqnr(sid, c)
        worst = min([min(v) for _, v in rows if v] or [99])
        if st == 'Accepted':
            ds, o = fetch(sid, c); res[tag] = o
            print(f"{tag} {sid} display={ds} Σtk={sum(v['tk'] for v in o.values()):.3f} SQNR最差余量{worst-22:+.2f}dB")
        else:
            print(f"{tag} {sid} {st}  SQNR最差 {worst:.2f}dB")
            for n, (s2, v) in enumerate(rows):
                if s2 not in ('Accepted', None):
                    print(f"    首个失败 tc{n}: {s2} {v}"); break

    a = res.get('锚')
    if a:
        s = signature(c)
        for f, tag, touched in specs[:-1]:
            if tag not in res: continue
            d = {i: (res[tag][i]['tk'] / a[i]['tk'] - 1) * 100 for i in range(1, 13)}
            if touched:
                ref = [i for i in range(1, 13) if i not in touched]
                m = sum(d[i] * s[i] for i in ref) / sum(s[i] * s[i] for i in ref)
                r = {i: d[i] - m * s[i] for i in range(1, 13)}
                gain = -sum(r[i] / 100 * a[i]['tk'] * SENS[i] for i in sorted(touched)) / 12
                print(f"{tag}: 机器项{m:+.2f}%  净残差 " +
                      " ".join(f"c{i}{r[i]:+.2f}" for i in sorted(touched)) +
                      f"   ⇒ raw {gain:+.3f}")
            else:
                print(f"{tag}: Σtk比 {(sum(v['tk'] for v in res[tag].values())/sum(v['tk'] for v in a.values())-1)*100:+.2f}%  " +
                      " ".join(f"c{i}{d[i]:+.2f}" for i in range(1, 13)))


main(sys.argv[1:])
