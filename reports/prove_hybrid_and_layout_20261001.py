"""CPU index/FP32 contracts for proposed hybrid GQ and padded ACT.

Does not import or execute a submitted kernel. GPU correctness, FP8 conversion,
FMA contraction, compiler behavior, or speed remain unverified.
"""
import ast
import hashlib
import itertools
import json
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def to_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def main():
    rng = random.Random(20261001)
    branches_checked = 0
    layouts_checked = 0
    for token_count, experts, k in [(257,32,4),(193,64,8),(251,96,3),(73,8,2)]:
        ids = [e for _ in range(token_count) for e in rng.sample(range(experts), k)]
        weights = [f32(0 if b%73 == 0 else rng.random()) for b in range(len(ids))]
        scales = [f32(2**rng.uniform(-28, -3)) for _ in range(token_count)]
        norms = [f32(2**rng.uniform(-3, 5)) for _ in range(experts)]
        order = sorted(range(len(ids)), key=lambda b: ids[b])
        inv = [None]*len(ids)
        for compact, branch in enumerate(order):
            inv[branch] = compact
        produced = [None]*len(ids)
        writes = [0]*len(ids)
        for t in range(token_count):
            a = scales[t]
            for slot in range(k):
                b = t*k+slot
                n, w = norms[ids[b]], weights[b]
                v = f32(f32(f32(f32(a*a)*n)*n)*abs(w))
                exp = (to_bits(max(v,f32(1e-30))) >> 23) & 255
                s = from_bits((exp+10)<<23)
                ri = from_bits((244-exp)<<23)
                produced[inv[b]] = (f32(a*.5), f32(f32(w*a)*ri), s)
                writes[inv[b]] += 1
        assert all(n == 1 for n in writes)
        for r,b in enumerate(order):
            # Independent consumer-order oracle, based on ORDER and expert id.
            t,e = b//k,ids[b]
            a,w,n = scales[t],weights[b],norms[e]
            bound = f32(a*a)
            bound = f32(bound*n)
            bound = f32(bound*n)
            bound = f32(bound*abs(w))
            exp = (to_bits(max(bound,f32(1e-30)))>>23)&255
            expected = (f32(a*.5), f32(f32(w*a)*from_bits((244-exp)<<23)), from_bits((exp+10)<<23))
            assert tuple(map(to_bits,produced[r])) == tuple(map(to_bits,expected))
            # Token-only Q row and replicated/sorted Q row encode the same token.
            assert order[inv[b]]//k == t
            branches_checked += 1
    # Include empty experts, all-in-one experts, both sides of every tile tail.
    vectors = [[0]*32,[0,1,127,128,129,255,256,257],[1024]+[0]*31,[0]*31+[1024]]
    vectors += [[rng.randrange(600) for _ in range(e)] for e in [8,32,64,96,256]]
    full_tiles_checked = valid_rows_checked = 0
    for counts in vectors:
        row_prefix = [0]+list(itertools.accumulate(counts))
        ntiles=[(n+127)//128 for n in counts]
        tile_prefix=[0]+list(itertools.accumulate(ntiles))
        capacity=128*((sum(counts)+127*len(counts)+127)//128)
        assert 128*tile_prefix[-1] <= capacity
        occupied=set()
        for e,n in enumerate(counts):
            for lm in range(ntiles[e]):
                padded=(tile_prefix[e]+lm)*128
                compact=row_prefix[e]+lm*128
                target=set(range(padded,padded+128))
                assert not target.intersection(occupied)
                occupied.update(target)
                for lane in range(min(128,n-lm*128)):
                    r=compact+lane
                    p=padded+lane
                    assert p-r == 128*tile_prefix[e]-row_prefix[e]
                    assert row_prefix[e] <= r < row_prefix[e+1]
                    valid_rows_checked += 1
                full_tiles_checked += 1
        layouts_checked += 1
    fs={n.name:n for n in ast.parse((ROOT/'p1/kernel.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    annotations={name:{a.arg:ast.unparse(a.annotation) if a.annotation else None for a in fs[name].args.args if a.arg in ('I','K')}
                 for name in ['_fgs_t1i_mdq_kernel_g','_fgs_t1i_mdq_tma_pre_kernel','_fgs_t1i_mdq_tma_pre_nf_kernel','_fgs_t1i_mdq_pre_kernel']}
    out={'production_sha256':hashlib.sha256((ROOT/'p1/kernel.py').read_bytes()).hexdigest(),
         'hybrid_branch_parameter_bit_comparisons':branches_checked*3,'hybrid_branch_mappings':branches_checked,
         'layout_scenarios':layouts_checked,'nonoverlapping_full_tile_writes':full_tiles_checked,
         'valid_compact_padded_row_pairs':valid_rows_checked,'current_dimension_annotations':annotations,
         'limits':['CPU index and explicitly rounded FP32 operation checks only.',
                   'No GPU test, compiler proof, FP8 SQNR, or measured speedup.',
                   'Existing A_SCALE exponent/floor semantics retained; this does not validate arbitrary extreme inputs.']}
    (ROOT/'reports/2026-10-01-direction-proofs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
