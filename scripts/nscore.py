"""用历史中位 tb 归一化的分数：剔除窗口运气，只反映 tk 真实快慢。"""
import sys, json, statistics as st
sys.path.insert(0, __file__.rsplit('/',1)[0])
S='/private/tmp/claude-501/-Users-sakimi/fcab29d7-e2b5-4a24-be4c-0897b7fa57ff/scratchpad/'
H=json.load(open(S+'hist.json'))
MED={i: st.median([v[1][str(i)]['tb'] for v in H.values() if str(i) in v[1]]) for i in range(1,13)}

def norm(tc):
    s=0
    for i in range(1,13):
        tk=tc[str(i)]['tk']; tb=MED[i]
        s+=int(100*tb/(tb+tk))
    return s, s/12.0

if __name__=='__main__':
    from cases import fetch
    from xpuoj_pow import Client
    c=Client(); rows=[]
    for sid in sys.argv[1:]:
        if sid in H: ds,tc = H[sid][0], H[sid][1]
        else: ds,tc = fetch(int(sid), c); tc={str(k):v for k,v in tc.items()}
        pts,nr = norm(tc)
        rows.append((sid, ds, sum(v['tk'] for v in tc.values()), pts, nr))
    for sid,ds,tk,pts,nr in rows:
        print(f"SID{sid}  实得={ds:>6}  Σtk={tk:7.3f}  归一分={nr:6.3f} ({pts})")
