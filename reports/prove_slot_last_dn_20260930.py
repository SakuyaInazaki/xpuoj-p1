"""CPU layout/coverage and conditional epilogue proof. No GPU performance claim.

The FP32 arithmetic check assumes identical input q/scales and separate rounded
multiply/add operations. It does NOT prove GPU MMA equality, FMA contraction,
TMA synchronization, the oracle SQNR, or arbitrary-call cache correctness.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bf16_bits(x):
    bits = np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def layout(ids, E, bm=128):
    T, k = ids.shape
    branches = np.arange(T*k,dtype=np.int64)
    keys = (branches % k)*E + ids.ravel()
    order = np.argsort(keys,kind="stable")
    inv = np.empty(T*k,dtype=np.int64)
    inv[order] = np.arange(T*k)
    counts = np.bincount(keys,minlength=k*E)
    starts = np.cumsum(np.r_[0,counts[:-1]])
    tiles = (counts+bm-1)//bm
    tile_starts = np.cumsum(np.r_[0,tiles[:-1]])
    padded_starts = tile_starts*bm
    invpad = np.empty(T*k,dtype=np.int64)
    coverage = np.zeros(T*k,dtype=np.int16)
    slot_writers = np.zeros((k,T),dtype=np.int16)
    for group in range(k*E):
        begin, n = int(starts[group]), int(counts[group])
        for local_m in range(int(tiles[group])):
            compact = begin+local_m*bm+np.arange(min(bm,n-local_m*bm))
            branch = order[compact]
            assert np.all(ids.ravel()[branch] == group % E)
            assert np.all(branch % k == group // E)
            coverage[branch] += 1
            slot_writers[group//E,branch//k] += 1
            invpad[branch] = padded_starts[group]+local_m*bm+np.arange(len(branch))
    assert np.all(coverage == 1) and np.all(slot_writers == 1)
    assert np.array_equal(order[inv],branches)
    assert len(np.unique(invpad)) == T*k
    cap_per_slot = ((T+E*(bm-1)+bm-1)//bm)*bm
    per_slot_padded = tiles.reshape(k,E).sum(axis=1)*bm
    assert np.all(per_slot_padded <= cap_per_slot)
    base_counts = np.bincount(ids.ravel(),minlength=E)
    base_tiles = int(((base_counts+bm-1)//bm).sum())
    assert int(tiles.sum()) <= base_tiles+E*(k-1)
    return order, inv, counts, starts, tiles, invpad, cap_per_slot, base_tiles


def run():
    rng=np.random.default_rng(20260930)
    runs=[]
    checked_branches=0
    for E,T,kind in [(e,t,"random") for e in (4,8,32) for t in (1,63,64,127,128,129,255,256,257,4096)] + [
            (32,65536,"random"),(32,65536,"all_one_pair"),
            (32,65536,"reverse_slots"),(32,4096,"cyclic"),(8,16384,"random")]:
        e0=rng.integers(0,E,size=T,dtype=np.int64)
        e1=rng.integers(0,E-1,size=T,dtype=np.int64)
        e1 += e1 >= e0
        ids=np.stack((e0,e1),axis=1)
        if kind=="all_one_pair": ids[:]=[0,E-1]
        if kind=="reverse_slots":
            ids[:]=[0,E-1]; ids[1::2]=[E-1,0]
        if kind=="cyclic":
            ids[:,0]=np.arange(T)%E; ids[:,1]=(ids[:,0]+1)%E
        order,inv,counts,starts,tiles,invpad,capacity,base_tiles=layout(ids,E)
        # Conditional identity of the split gather/epilogue with arbitrary
        # representable E4M3 values, including zeros and subnormals.
        H=8
        codes=rng.integers(0,127,size=(2*T,H),dtype=np.uint8)
        exp=(codes>>3).astype(np.int32)
        mant=(codes&7).astype(np.float32)
        q=np.where(exp==0,mant*np.float32(2**-9),
                   (1+mant/8)*np.exp2(exp-7)).astype(np.float32)
        q *= rng.choice(np.array([-1.,1.],np.float32),size=q.shape)
        scales=np.exp2(rng.integers(-20,20,size=(2*T,1))).astype(np.float32)
        contrib=q*scales
        baseline=bf16_bits((np.zeros((T,H),np.float32)+contrib[0::2])+contrib[1::2])
        first_branches=order[:T]
        assert np.all(first_branches%2==0)
        first_q=q[first_branches]
        first_s=scales[first_branches]
        result=np.empty((T,H),np.uint16)
        for group in range(E,2*E):
            begin=int(starts[group]); end=begin+int(counts[group])
            branches=order[begin:end]; token=branches//2
            first_row=inv[token*2]
            total=(np.zeros((len(token),H),np.float32)+first_q[first_row]*first_s[first_row])+q[branches]*scales[branches]
            result[token]=bf16_bits(total)
        assert np.array_equal(baseline,result)
        checked_branches += 2*T
        lo=ids.min(axis=1); hi=ids.max(axis=1)
        pair_id=lo*(2*E-lo-1)//2+hi-lo-1
        pair_counts=np.bincount(pair_id,minlength=E*(E-1)//2)
        pair_tiles=int(((pair_counts+63)//64).sum())
        runs.append({"E":E,"T":T,"distribution":kind,"branches":2*T,
                     "base_padded_rows":base_tiles*128,
                     "slot_padded_rows":int(tiles.sum())*128,
                     "slot_padding_ratio":float(tiles.sum()/base_tiles),
                     "slot0_capacity":capacity,
                     "pair_bm64_equivalent_branch_rows":pair_tiles*64*2,
                     "pair_compute_ratio":pair_tiles*64*2/(base_tiles*128),
                     "pair_weight_tile_visit_ratio":2*pair_tiles/base_tiles,
                     "status":"pass"})
    T,H=65536,1024
    byte_ledger={"T":T,"H":H,"base_useful_scratch_write_read_and_output":6*T*H,
                 "slot_last_useful_scratch_write_read_and_output":4*T*H,
                 "naive_bf16_two_pass_write_read_and_output":6*T*H,
                 "pair_direct_output":2*T*H,
                 "slot_last_saved_useful_bytes":2*T*H,
                 "extra_full_ACT_reorder_read_write_c11":4*T*1024,
                 "scope":"Logical bytes; excludes padding, scales, metadata, caches and bus transactions."}
    result={"kind":"CPU-only mathematical/layout proof; not GPU correctness or performance",
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scenarios":len(runs),"branches_checked":checked_branches,
            "epilogue_assumption":"Same q/scales and FP32 multiply/add rounding, no unmodelled FMA contraction",
            "runs":runs,"byte_ledger":byte_ledger}
    (ROOT/"reports/2026-09-30-slot-last-proof.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="runs"},ensure_ascii=False,indent=2))
    print(json.dumps([r for r in runs if r["T"]==65536],indent=2))


if __name__=="__main__":
    run()
