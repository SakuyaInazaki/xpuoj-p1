"""CPU contracts for proposed P1 changes, not a GPU correctness/speed test.

Checks (1) independent persistent counters, including expert-tail swizzles,
(2) compact-to-padded ACT ownership and inverse maps, and (3) factored static
DN scales with unchanged FP32 rounding/floor. No dependencies beyond Python.
"""
import bisect
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.pack('<f', x).hex()


def swizzle(m, n, count_m, count_n, group_m):
    linear = m*count_n+n
    group = linear//(group_m*count_n)
    row_base = group*group_m
    group_rows = min(count_m-row_base, group_m)
    offset = linear%(group_m*count_n)
    return row_base+offset%group_rows, offset//group_rows


def prefix(xs):
    return [0]+list(itertools.accumulate(xs))


def check_layout(counts, ncols, group_m, grid):
    rows, tiles = prefix(counts), [(n+127)//128 for n in counts]
    tile_prefix = prefix(tiles)
    total = tile_prefix[-1]*ncols
    def decode(t):
        pid_m, pid_n = divmod(t, ncols)
        e = bisect.bisect_right(tile_prefix, pid_m)-1
        local = pid_m-tile_prefix[e]
        lm, pn = swizzle(local,pid_n,tiles[e],ncols,group_m)
        return e, lm, pn, rows[e]+lm*128, (tile_prefix[e]+lm)*128
    seen = set()
    for pid in range(grid):
        write_counter = pid-grid
        for read_counter in range(pid,total,grid):
            write_counter += grid
            assert write_counter == read_counter
            address = decode(write_counter)
            assert address == decode(read_counter)
            assert address[:3] not in seen
            seen.add(address[:3])
    assert len(seen) == total
    compact_to_padded = {}
    padding = set()
    owners = set()
    for e, n in enumerate(counts):
        for lm in range(tiles[e]):
            pr = (tile_prefix[e]+lm)*128
            for lane in range(128):
                p = pr+lane
                assert p not in owners
                owners.add(p)
                if lm*128+lane < n:
                    compact_to_padded[rows[e]+lm*128+lane] = p
                else:
                    padding.add(p)
    assert len(compact_to_padded) == sum(counts)
    assert set(compact_to_padded.values()).isdisjoint(padding)
    cap = ((sum(counts)+len(counts)*127+127)//128)*128
    assert tile_prefix[-1]*128 <= cap
    # A full TMA write is private to one expert block; padded DN reads the
    # same physical row while scales can remain in compact row order.
    for r,p in compact_to_padded.items():
        e = bisect.bisect_right(rows,r)-1
        assert p == r + 128*tile_prefix[e]-rows[e]
    return total, len(compact_to_padded)


def main():
    rng = random.Random(20260930)
    distributions = [[0]*8, [0,1,127,128,129,255,256,257],
                     [4096]+[0]*31, [0]*31+[4096], [4096]*32]
    distributions += [[rng.randrange(0,600) for _ in range(e)] for e in (8,32,64,96,256)]
    cases = tiles = rows = 0
    for counts in distributions:
        for nc,gm,grid in itertools.product((4,8,16),(8,32),(1,132)):
            nt,nr = check_layout(counts,nc,gm,grid)
            cases += 1; tiles += nt; rows += nr
    # Independent producer/consumer index paths: materialise the original
    # DSCL in expert order, then reconstruct it in unsorted branch order.
    # Leave two experts empty and cover padding plus dynamic floor cases.
    branch_experts = [rng.randrange(1,31) for _ in range(10000)]
    counts = [branch_experts.count(e) for e in range(32)]
    row_prefix = prefix(counts)
    tile_prefix = prefix([(n+127)//128 for n in counts])
    order = sorted(range(len(branch_experts)), key=branch_experts.__getitem__)
    inv = [0]*len(order)
    inv_pad = [0]*len(order)
    for compact,branch in enumerate(order):
        e = branch_experts[branch]
        inv[branch] = compact
        inv_pad[branch] = 128*tile_prefix[e]+compact-row_prefix[e]
    a_scales = [f32(0.0 if row%97 == 0 else 2**rng.uniform(-60,10))
                for row in range(len(order))]
    constants = [[f32(2**rng.uniform(-10,20)) for _ in range(4)] for e in range(32)]
    old_dscl = {}
    for e,n in enumerate(counts):
        for offset in range(n):
            row = row_prefix[e]+offset
            for chunk,c in enumerate(constants[e]):
                old_dscl[128*tile_prefix[e]+offset,chunk] = bits(
                    max(f32(a_scales[row]*c),f32(1e-12)))
    comparisons = 0
    for branch,e in enumerate(branch_experts):
        a = a_scales[inv[branch]]
        for chunk,c in enumerate(constants[e]):
            in_final = f32(max(f32(a*c),f32(1e-12)))
            assert bits(in_final) == old_dscl[inv_pad[branch],chunk]
            comparisons += 1
    out = {'production_sha256': hashlib.sha256((ROOT/'p1/kernel.py').read_bytes()).hexdigest(),
           'persistent_schedule_scenarios': cases, 'tile_visits_checked': tiles,
           'valid_row_mappings_checked_across_scenarios': rows,
           'fp32_scale_bit_comparisons': comparisons,
           'limitations': ['CPU contracts only; no generated Triton, PTX, GPU timings or SQNR.',
                          'Scale test covers producer-order DSCL vs branch-order INV/INV_PAD/expert reconstruction; assumes identical FP32 operation order.',
                          'Does not prove FMA contraction, TMA ordering, compiler lowering or physical cache-key behavior.',
                          'Retains existing scale floor semantics, including its known tiny-input algebra boundary.']}
    (ROOT/'reports/2026-09-30-sprint-direction-proofs.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
