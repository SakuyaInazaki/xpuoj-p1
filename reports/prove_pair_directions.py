"""CPU integer oracles for proposed EP2 placement and unordered-pair DN grouping.

These checks do not import the submission, execute CUDA, or establish SQNR/speed.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import random

OUT = Path(__file__).resolve().parent


def ep2_check(T, mode, seed):
    E, k = 256, 8
    rng = random.Random(seed)
    pool = list(range(E)) if mode == "random" else list(range(0,128) if mode == "lower" else range(128,256))
    routes = [[rng.sample(pool,k) for _ in range(T)] for _ in range(4)]
    buckets = [[] for _ in range(4)]
    for source in range(4):
        for t, ids in enumerate(routes[source]):
            for j,e in enumerate(ids):
                dest = (source//2)*2 + e//128
                code = ((source%2)*T+t)*k+j
                buckets[dest].append((e%128,source%2,code,source,t,j,e))
    observed = set()
    maximum = 0
    for dest, records in enumerate(buckets):
        records.sort(key=lambda v:(v[0],v[1],v[2]))
        assert len(records)<=2*T*k
        maximum=max(maximum,len(records))
        counts=Counter(r[0] for r in records)
        tiles=sum((n+127)//128 for n in counts.values())
        assert tiles <= (2*T*k+127)//128 + 128
        for local_e,slot,code,source,t,j,e in records:
            assert e == (dest%2)*128 + local_e
            assert source//2 == dest//2
            assert code//k == slot*T+t and code%k == j
            key=(source,t,j)
            assert key not in observed
            observed.add(key)
    assert len(observed)==4*T*k
    if mode in ("lower","upper"):
        assert maximum==2*T*k, "Must cover worst case, not only uniform mean"
    return dict(T=T,mode=mode,branches_checked=len(observed),max_owner_rows=maximum,capacity=2*T*k)


def pair_key(a,b,E):
    lo,hi=sorted((a,b))
    assert lo<hi
    return lo*(2*E-lo-1)//2+hi-lo-1


def dn_check(T,E,mode,seed):
    rng=random.Random(seed)
    pairs=[(a,b) for a in range(E) for b in range(a+1,E)]
    assert sorted(pair_key(a,b,E) for a,b in pairs)==list(range(len(pairs)))
    routes=[rng.sample(range(E),2) if mode=="random" else [0,E-1] for _ in range(T)]
    expert_order=sorted(range(2*T),key=lambda b:routes[b//2][b%2])
    inv={b:p for p,b in enumerate(expert_order)}
    groups=defaultdict(list)
    for t,(a,b) in enumerate(routes): groups[pair_key(a,b,E)].append(t)
    seen=set()
    for key,tokens in groups.items():
        lo,hi=pairs[key]
        for t in tokens:
            assert t not in seen
            seen.add(t)
            jlo=0 if routes[t][0]==lo else 1
            jhi=1-jlo
            assert expert_order[inv[2*t+jlo]]==2*t+jlo
            assert routes[t][jlo]==lo and routes[t][jhi]==hi
    assert len(seen)==T
    return dict(T=T,E=E,mode=mode,groups_used=len(groups),tokens_checked=T,
                padded_pair_rows={str(bm):sum((len(v)+bm-1)//bm*bm for v in groups.values()) for bm in (32,64,128)},
                padded_expert_branch_rows=sum((n+127)//128*128 for n in Counter(e for ids in routes for e in ids).values()))


def main():
    ep=[ep2_check(t,mode,100+i) for i,t in enumerate((1,17,4096)) for mode in ("random","lower","upper")]
    dn=[dn_check(t,e,mode,200+i) for i,(t,e) in enumerate(((17,8),(16384,8),(65536,32))) for mode in ("random","single_pair")]
    result=dict(scope="CPU integer placement/capacity only; no GPU, floating-point or performance validation",
                ep2=ep,pair_dn=dn,all_passed=True)
    (OUT/"2026-09-27-pair-directions-proof.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
