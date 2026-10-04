"""V721 ragged tail-block: coverage proof + cost/benefit model.

Part 1  -- exact numpy mirror of the new metadata kernel (_rag_meta_kernel) and
           of the two consuming kernels' tile->row maps (including the real
           tl.swizzle2d permutation used by the BM=128 main kernel).
           Asserts every (row, n-block) pair is computed exactly once over
           200 random expert-count vectors plus hand-built edge cases.

Part 2  -- Monte-Carlo cost/benefit for the 12 judged shapes: padded-row
           saving, tile-count delta, fixed-cost penalty, net.

Run:  python3 sandbox/verify_v721.py
"""
import numpy as np

BM = 128           # GROUP_GEMM_BLOCK_SIZE_M
BT = 64            # _RAG_BM_TAIL


# ---------------------------------------------------------------------------
# Part 1: metadata mirror + coverage proof
# ---------------------------------------------------------------------------
def build_ragged_meta(rows, bm=BM, bt=BT):
    """Mirror of _rag_meta_kernel.  rows: int array [E]."""
    rows = np.asarray(rows, dtype=np.int64)
    q = rows // bm
    r = rows - q * bm
    has_tail = (r > 0) & (r <= bt)
    # exact form used by _rag_meta_kernel after V721.1:
    main_t = -(-rows // bm) - has_tail.astype(np.int64)
    assert (main_t == q + (r > bt).astype(np.int64)).all()   # both forms agree
    assert (main_t >= 0).all()
    rcum = np.cumsum(rows)
    roff = rcum - rows                              # expert row offset
    tcum = np.cumsum(main_t)                        # inclusive
    total = int(tcum[-1]) if len(tcum) else 0

    blk_e = np.empty(total, dtype=np.int64)
    blk_ro = np.empty(total, dtype=np.int64)
    blk_tn = np.empty(total, dtype=np.int64)
    blk_tc = np.empty(total, dtype=np.int64)
    for p in range(total):
        # kernel does: e = #{ j : tcum[j] <= p }
        e = int(np.sum(tcum <= p))
        blk_e[p] = e
        blk_ro[p] = roff[e]
        blk_tn[p] = main_t[e]
        blk_tc[p] = tcum[e]

    sel = np.nonzero(has_tail)[0]
    tail_e = sel.astype(np.int64)
    tail_b = (roff[sel] + q[sel] * bm).astype(np.int64)
    tail_r = r[sel].astype(np.int64)
    return (blk_e, blk_ro, blk_tn, blk_tc, total), (tail_e, tail_b, tail_r, len(sel))


def build_base_meta(rows, bm=BM):
    """Mirror of the stock build_block_row_idx_info_kernel (baseline)."""
    rows = np.asarray(rows, dtype=np.int64)
    tsplit = -(-rows // bm)                          # cdiv
    rcum = np.cumsum(rows)
    roff = rcum - rows
    tcum = np.cumsum(tsplit)
    total = int(tcum[-1]) if len(tcum) else 0
    blk_e = np.empty(total, dtype=np.int64)
    for p in range(total):
        blk_e[p] = int(np.sum(tcum <= p))
    return blk_e, roff[blk_e], tsplit[blk_e], tcum[blk_e], total


def swizzle2d(i, j, size_i, size_j, size_g):
    """Faithful port of tl.swizzle2d."""
    ij = i * size_j + j
    size_gj = size_g * size_j
    group_id = ij // size_gj
    off_i = group_id * size_g
    size_g = min(size_i - off_i, size_g)
    new_i = off_i + (ij % size_g)
    new_j = (ij % size_gj) // size_g
    return new_i, new_j


def coverage(rows, nbn, group_m, ragged=True):
    """Return a [total_rows, nbn] int8 count of how often each output element
    is written, replaying the exact index math of the kernels."""
    rows = np.asarray(rows, dtype=np.int64)
    total_rows = int(rows.sum())
    cov = np.zeros((max(total_rows, 1), nbn), dtype=np.int32)

    if ragged:
        (blk_e, blk_ro, blk_tn, blk_tc, total), (te, tb, tr, ntail) = \
            build_ragged_meta(rows)
    else:
        blk_e, blk_ro, blk_tn, blk_tc, total = build_base_meta(rows)
        te = tb = tr = np.zeros(0, dtype=np.int64)
        ntail = 0

    # --- BM=128 main kernel (unchanged source; persistent over total*nbn) ---
    for tile_id in range(total * nbn):
        pid_m = tile_id // nbn
        pid_n = tile_id % nbn
        e = blk_e[pid_m]
        n_rows = rows[e]
        row_begin = blk_ro[pid_m]
        t_num = blk_tn[pid_m]
        t_cum = blk_tc[pid_m]
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = swizzle2d(local_m, pid_n, t_num, nbn, group_m)
        offs_m = row_begin + local_m * BM + np.arange(BM)
        m = offs_m < row_begin + n_rows
        cov[offs_m[m], pid_n] += 1

    # --- BM=64 tail kernel (new) ---
    for tile_id in range(ntail * nbn):
        pid_m = tile_id // nbn
        pid_n = tile_id % nbn
        row_begin = tb[pid_m]
        n_rows = tr[pid_m]
        offs_m = row_begin + np.arange(BT)
        m = offs_m < row_begin + n_rows
        cov[offs_m[m], pid_n] += 1

    return cov[:total_rows], total, ntail


def run_coverage_tests():
    rng = np.random.default_rng(20260904)
    edge = [
        [0], [1], [63], [64], [65], [127], [128], [129], [191], [192], [193],
        [255], [256], [257],
        [0, 0, 0], [0, 1, 0], [128, 0, 128], [64, 64, 64], [65, 65, 65],
        [0, 127, 1, 128, 129, 64, 65, 63, 192, 193, 255, 256],
        [1] * 16, [127] * 8, [128] * 8, [129] * 8,
        [512, 511, 513, 448, 576, 500, 524],
        list(range(0, 300)),                       # 300 experts, rows 0..299
        # c9/c10 regime: 256 experts, rows ~ 128 +- 11 -> many experts with
        # main tile_num == 0 (rows <= 64) or 1.  Gated OFF in v721, verified
        # anyway as defence in depth (coordinator review point 1).
        list(map(int, np.random.default_rng(9).multinomial(32768, [1/256]*256))),
        list(map(int, np.random.default_rng(10).multinomial(32768, [1/256]*256))),
        [64] * 32, [63] * 32, [0, 64, 0, 64, 0, 64],
    ]
    cases = list(edge)
    for _ in range(200):
        E = int(rng.integers(1, 129))
        style = rng.integers(0, 4)
        if style == 0:
            r = rng.integers(0, 300, size=E)
        elif style == 1:                            # near-multiple-of-128 (real workload)
            base = int(rng.choice([128, 256, 512, 1024, 2048, 4096]))
            r = np.maximum(0, base + rng.normal(0, 40, size=E).astype(int))
        elif style == 2:
            r = rng.integers(0, 3, size=E)
        else:
            r = rng.integers(0, 5000, size=E)
        cases.append(list(map(int, r)))

    bad = 0
    tiles_base_all = tiles_rag_all = 0
    for ci, rows in enumerate(cases):
        nbn = int(np.random.default_rng(ci).integers(1, 5))
        gm = int(np.random.default_rng(ci + 7).choice([4, 8, 16, 32]))
        cov_r, tot_r, ntail = coverage(rows, nbn, gm, ragged=True)
        cov_b, tot_b, _ = coverage(rows, nbn, gm, ragged=False)
        ok = cov_r.size == 0 or (cov_r == 1).all()
        same_tiles = (tot_r + ntail) == tot_b
        if not ok or not same_tiles or not (cov_b.size == 0 or (cov_b == 1).all()):
            bad += 1
            print("FAIL case", ci, "rows=", rows[:12], "...",
                  "min/max cover=", cov_r.min() if cov_r.size else "-",
                  cov_r.max() if cov_r.size else "-",
                  "tiles base/rag+tail=", tot_b, tot_r + ntail)
        tiles_base_all += tot_b
        tiles_rag_all += tot_r + ntail

        # tail-table invariants (what the BM=64 kernel relies on)
        a = np.asarray(rows, dtype=np.int64)
        roff = np.cumsum(a) - a
        (_, _, _, _, _), (te, tb, tr, nt) = build_ragged_meta(a)
        assert nt == len(te) <= len(a)
        assert (np.diff(te) > 0).all() if nt > 1 else True      # compacted, ascending
        assert ((tr >= 1) & (tr <= BT)).all() if nt else True   # 1..64 rows
        # each tail block ends exactly at its expert's last row
        assert (tb + tr == roff[te] + a[te]).all() if nt else True
        # each tail block starts on a 128-multiple offset inside its expert
        assert ((tb - roff[te]) % BM == 0).all() if nt else True

    print(f"[coverage] {len(cases)} cases, failures = {bad}")
    print(f"[coverage] every (row, n-block) written exactly once in all cases")
    print(f"[coverage] tail-table invariants (compacted / ascending / 1<=rows<=64 /"
          f" ends at expert end / starts on 128 boundary): OK")
    print(f"[coverage] total m-tiles  baseline = {tiles_base_all}, "
          f"ragged(main+tail) = {tiles_rag_all}  "
          f"(delta = {tiles_rag_all - tiles_base_all}; must be 0)")
    assert bad == 0
    assert tiles_base_all == tiles_rag_all
    return bad


def run_bm_tradeoff():
    """BM=32 vs BM=64 for the capped scheme.

    A BM=32 tail tile cannot use wgmma (Hopper wgmma has m>=64), and sm_90
    mma.sync has no FP8 path at all, so Triton must up-convert to f16 -> the
    tile costs `1/eff` times its proportional (32/128) share of a full tile.
    A BM=64 tile keeps wgmma (m64nNk32 is the native shape).
    Break-even: the eff at which cap32 stops beating cap64.
    """
    print("\n=== BM=32 vs BM=64 (capped scheme), MMA-effective padded rows saved ===")
    print(f"{'case':5s} {'cap64':>7s} {'cap32@eff=1':>12s} {'cap32@0.75':>11s} "
          f"{'cap32@0.5':>10s} {'break-even eff':>14s}")
    for name, T, H, E, I, k, waste in CASES:
        if name not in ENABLED:
            continue
        o = mc(T, H, E, I, k, seed=hash(name) % 2**31)
        pad_b, _, _ = o["base"]
        pad64, _, n64 = o[("cap", 64)]
        pad32, _, n32 = o[("cap", 32)]
        s64 = pad_b - pad64
        # cap32 with efficiency loss: the 32-row tile behaves like 32/eff rows.
        def s32(eff):
            return pad_b - (pad32 + n32 * 32 * (1.0 / eff - 1.0))
        lo, hi = 0.05, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if s32(mid) < s64:
                lo = mid
            else:
                hi = mid
        print(f"{name:5s} {s64:7.0f} {s32(1.0):12.0f} {s32(0.75):11.0f} "
              f"{s32(0.5):10.0f} {lo:14.2f}")
    print("  -> cap32 only wins if the BM=32 (MMAv2, fp8->f16) tail tile keeps")
    print("     >= ~0.75 of proportional throughput.  It cannot.  => BM=64.")


# ---------------------------------------------------------------------------
# Part 2: cost / benefit
# ---------------------------------------------------------------------------
CASES = [
    # name, T, H, E, I, k, waste_ms(md+dn, measured by user)
    ("c1", 16384, 4096, 8, 8192, 2, 0.052),
    ("c2", 16384, 4096, 8, 14336, 2, 0.090),
    ("c3", 16384, 2048, 32, 2048, 4, 0.026),
    ("c4", 16384, 2048, 32, 1024, 4, 0.013),
    ("c5", 8192, 3584, 64, 2560, 8, 0.113),
    ("c6", 8192, 3584, 64, 1024, 8, 0.045),
    ("c7", 16384, 4096, 96, 2048, 3, 0.153),
    ("c8", 16384, 4096, 96, 1024, 3, 0.077),
    ("c9", 4096, 4096, 256, 2048, 8, 0.402),
    ("c10", 4096, 4096, 256, 1536, 8, 0.301),
    ("c11", 65536, 1024, 32, 1024, 2, 0.006),
    ("c12", 65536, 1024, 32, 2048, 2, 0.013),
]

ENABLED = {"c3", "c4", "c5", "c6", "c7", "c8"}      # v721 gate: _GA[0] and 16<=E<=96


def mc(T, H, E, I, k, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = T * k
    out = {}
    rows = rng.multinomial(n, np.full(E, 1.0 / E), size=trials)     # [trials,E]
    q = rows // BM
    r = rows - q * BM

    pad_base = (-(-rows // BM) * BM - rows).sum(axis=1)
    tiles_base = (-(-rows // BM)).sum(axis=1)

    for bt in (32, 64):
        has_tail = (r > 0) & (r <= bt)
        pad = np.where(has_tail, bt - r, np.where(r > 0, BM - r, 0)).sum(axis=1)
        tiles = (-(-rows // BM)).sum(axis=1)                # capped: identical
        ntail = has_tail.sum(axis=1)
        out[("cap", bt)] = (pad.mean(), tiles.mean(), ntail.mean())

    for bt in (32, 64):
        # full ragged: every leftover row split into bt-sized tiles
        ntail = -(-r // bt)
        pad = (ntail * bt - r).sum(axis=1)
        tiles = (q + ntail).sum(axis=1)
        out[("full", bt)] = (pad.mean(), tiles.mean(), (ntail.sum(axis=1)).mean())

    out["base"] = (pad_base.mean(), tiles_base.mean(), 0.0)
    out["rows"] = float(n)
    return out


def run_cost_model():
    # Fixed cost per (m,n) tile expressed in "padded rows"; F/(F+C) is the
    # fraction of a tile's time that does not scale with BLOCK_M.
    for FRAC, label in ((0.077, "md-like F/(F+C)=7.7%"), (0.241, "dn-like F/(F+C)=24.1%")):
        F_rows = BM * FRAC
        print(f"\n=== fixed cost per extra tile = {F_rows:.1f} padded-row equivalents "
              f"({label}) ===")
        hdr = (f"{'case':5s} {'mean rows/E':>11s} {'pad/E base':>10s} "
               f"{'cap64 save':>10s} {'cap32 save':>10s} "
               f"{'full64 net':>10s} {'full32 net':>10s}")
        print(hdr)
        tot = {"cap64": 0.0, "cap32": 0.0, "full64": 0.0, "full32": 0.0}
        for name, T, H, E, I, k, waste in CASES:
            o = mc(T, H, E, I, k, seed=hash(name) % 2**31)
            pad_b, tiles_b, _ = o["base"]
            per_e = o["rows"] / E
            row_scale = waste / pad_b if pad_b > 0 else 0.0   # ms per padded row
            line = f"{name:5s} {per_e:11.1f} {pad_b / E:10.1f} "
            vals = {}
            for tag, key in (("cap64", ("cap", 64)), ("cap32", ("cap", 32)),
                             ("full64", ("full", 64)), ("full32", ("full", 32))):
                pad, tiles, ntail = o[key]
                saved = pad_b - pad
                extra = tiles - tiles_b
                net_rows = saved - extra * F_rows
                vals[tag] = net_rows * row_scale
            line += (f"{vals['cap64'] * 1000:9.1f}u {vals['cap32'] * 1000:9.1f}u "
                     f"{vals['full64'] * 1000:9.1f}u {vals['full32'] * 1000:9.1f}u")
            print(line + ("" if name in ENABLED else "   [off]"))
            if name in ENABLED:
                for t in tot:
                    tot[t] += vals[t]
        print(f"{'SUM(enabled c3-c8)':>17s}  "
              + "  ".join(f"{t}={tot[t] * 1000:.0f}us" for t in
                          ("cap64", "cap32", "full64", "full32")))


def run_detail_table():
    print("\n=== v721 (cap, BT=64) per-case detail, MMA-side only ===")
    print(f"{'case':5s} {'E':>4s} {'nbn_md':>6s} {'nbn_dn':>6s} "
          f"{'pad rows base':>13s} {'pad rows v721':>13s} {'saved':>7s} "
          f"{'tail tiles/E':>12s} {'extra tiles':>11s} {'est. gain':>10s}")
    total = 0.0
    for name, T, H, E, I, k, waste in CASES:
        o = mc(T, H, E, I, k, seed=hash(name) % 2**31)
        pad_b, tiles_b, _ = o["base"]
        pad, tiles, ntail = o[("cap", 64)]
        saved = pad_b - pad
        gain = waste * saved / pad_b if pad_b else 0.0
        on = name in ENABLED
        print(f"{name:5s} {E:4d} {I // 128:6d} {H // 256:6d} "
              f"{pad_b:13.0f} {pad:13.0f} {saved:7.0f} "
              f"{ntail:12.1f} {tiles - tiles_b:11.0f} "
              f"{gain * 1000:9.1f}u" + ("" if on else "   [off]"))
        if on:
            total += gain
    print(f"  -> v721 enabled set (c3-c8) total expected gain = {total * 1000:.0f} us "
          f"= {total:.3f} ms")


if __name__ == "__main__":
    run_coverage_tests()
    run_detail_table()
    run_bm_tradeoff()
    run_cost_model()
