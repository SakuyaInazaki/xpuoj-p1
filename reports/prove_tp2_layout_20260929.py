"""CPU checks for TP2 tiled channel slicing and real-arithmetic MLP identity.

Not a BF16/FP8 oracle and not a GPU performance test.
"""
from pathlib import Path
import json
import math
import random

ROOT = Path(__file__).resolve().parents[1]


def pack(e, n, k, N, K, BN):
    return ((e*(N//BN)+n//BN)*(K//128)+k//128)*BN+n%BN, k%128


def unpack(row, col, N, K, BN):
    n_inner = row % BN
    q = row // BN
    kt = q % (K//128)
    q //= K//128
    return q//(N//BN), (q%(N//BN))*BN+n_inner, kt*128+col


def main():
    rng = random.Random(20260929)
    checks = []
    for I in (2048, 1536):
        H, E, Is = 4096, 256, I//2
        for shard in (0, 1):
            for _ in range(12000):
                e, h, j = rng.randrange(E), rng.randrange(H), rng.randrange(Is)
                for gu in (0, 1):
                    local_n = gu*Is+j
                    source_n = gu*I+shard*Is+j
                    r,c = pack(e, local_n, h, 2*Is, H, 128)
                    assert unpack(r,c,2*Is,H,128) == (e,local_n,h)
                    sr,sc = pack(e, source_n, h, 2*I, H, 128)
                    assert unpack(sr,sc,2*I,H,128) == (e,source_n,h)
                r,c = pack(e,h,j,H,Is,256)
                assert unpack(r,c,H,Is,256) == (e,h,j)
                sr,sc = pack(e,h,shard*Is+j,H,I,256)
                assert unpack(sr,sc,H,I,256) == (e,h,shard*Is+j)
            checks.append({"I":I,"shard":shard,"GU_and_D_coordinates":36000})
    max_abs = 0.
    for _ in range(50):
        H,I = 7,10
        x=[rng.uniform(-1,1) for _ in range(H)]
        g=[[rng.uniform(-.4,.4) for _ in range(H)] for _ in range(I)]
        u=[[rng.uniform(-.4,.4) for _ in range(H)] for _ in range(I)]
        d=[[rng.uniform(-.4,.4) for _ in range(I)] for _ in range(H)]
        a=[]
        for j in range(I):
            gj=sum(x[h]*g[j][h] for h in range(H))
            uj=sum(x[h]*u[j][h] for h in range(H))
            a.append(gj/(1+math.exp(-gj))*uj)
        for h in range(H):
            full=sum(a[j]*d[h][j] for j in range(I))
            shards=sum(sum(a[j]*d[h][j] for j in range(s*I//2,(s+1)*I//2)) for s in (0,1))
            max_abs=max(max_abs,abs(full-shards))
            assert abs(full-shards)<1e-12
    T,E,k=4096,256,8
    sources=[]
    for _ in range(2):
        counts=[0]*E
        for _ in range(T):
            for e in rng.sample(range(E),k):
                counts[e]+=1
        sources.append(counts)
    baseline=sum((n+127)//128*128 for c in sources for n in c)/2
    merged=[a+b for a,b in zip(*sources)]
    ep_rows=[sum((n+127)//128*128 for n in merged[s*128:(s+1)*128]) for s in (0,1)]
    tp_rows=sum((n+127)//128*128 for n in merged)
    result={"scope":"CPU coordinate checks + Python float64 identity only; no quantized accuracy/GPU performance claim.",
            "layout_checks":checks,"checked_coordinates":sum(x['GU_and_D_coordinates'] for x in checks),
            "float64_examples":50,"float64_max_abs_difference":max_abs,
            "synthetic_legal_topk_padding":{"T":T,"E":E,"k":k,
                "baseline_mean_padded_branch_rows":baseline,"ep2_padded_rows_by_owner":ep_rows,
                "tp2_padded_rows":tp_rows,"tp2_half_I_equivalent_rows":tp_rows/2,
                "work_ratio_ep2_mean":sum(ep_rows)/2/baseline,
                "work_ratio_tp2":tp_rows/2/baseline,
                "warning":"Uniform random unique top-k model, not measured OJ routing."}}
    (ROOT/'reports/2026-09-29-tp2-layout-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
