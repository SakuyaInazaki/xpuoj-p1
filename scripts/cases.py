"""拉某次提交的 12 个 case 的 tk/tb/score/SQNR。"""
import sys, json, re
sys.path.insert(0, __file__.rsplit('/',1)[0])
from xpuoj_pow import Client

def fetch(sid, c=None):
    c = c or Client()
    d = c.get_detail(sid).json()
    p = d['progress']
    out = {}
    for h, tc in p['testcaseResult'].items():
        ue = tc.get('userError') or ''
        if isinstance(ue, dict):          # 超长时平台返回 {content, omittedLength}
            ue = ue.get('content') or ''
        m = re.search(r'tc=(\d+)', ue)
        j = re.search(r'\{"schema_version".*?\}', ue)
        s = re.findall(r'SQNR=([0-9.]+) dB', ue)
        if not m or not j:
            continue
        v = json.loads(j.group(0))
        out[int(m.group(1))] = dict(tk=v['tk_time_ms'], tb=v['tb_time_ms'],
                                    score=tc.get('score'), sqnr=min(map(float, s)) if s else None,
                                    status=tc.get('status'))
    return d['progress'].get('displayScore'), out

if __name__ == '__main__':
    c = Client()
    for sid in sys.argv[1:]:
        ds, o = fetch(int(sid), c)
        tot = sum(v['tk'] for v in o.values())
        print(f"--- SID {sid}  display={ds}  Σtk={tot:.3f}")
        for i in sorted(o):
            v = o[i]
            print(f"  tc{i:<3} tk={v['tk']:7.3f} tb={v['tb']:8.3f} sc={v['score']:>3} sqnr={v['sqnr']}")
