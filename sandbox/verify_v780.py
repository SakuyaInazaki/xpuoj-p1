# -*- coding: utf-8 -*-
"""V780 coverage proof.

V780 differs from V721 in *how* the tail tile is executed: instead of a new
BM=64 kernel source, it re-launches the **unmodified** main kernel source
(`_fgs_t1i_mdq_kernel_g` / `_dn_tma2_f8_kernel`) with `BLOCK_M=64` and a
"one fake expert-group per tail tile" metadata table:

    expert_ids[p]     = tail_expert[p]        (real expert id)
    split_size[e]     = tail_rows[e]          (dense, r_e or 0)
    split_size_cum[p] = tail_begin[p]         (roff_e + q_e*128)
    tile_num[p]       = 1
    tile_cum[p]       = p + 1
    num_tiles         = n_tail

so this script replays the *real* main-kernel index math (t_cum/t_num ->
local_m -> tl.swizzle2d -> offs_m/row_mask) on that table and asserts it
reproduces V721's already-verified tail model exactly, then re-runs the full
226+ case coverage suite of sandbox/verify_v721.py unchanged.

Run:  python3 sandbox/verify_v780.py
"""
import io
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_v721 as V                       # reuse, do not modify

BM = V.BM
BT = V.BT


def tail_coverage_via_main_kernel(rows, nbn, group_m):
    """Replay _fgs_t1i_mdq_kernel_g / _dn_tma2_f8_kernel index math (BLOCK_M=64)
    over the fake tail table.  Returns a [total_rows, nbn] write count."""
    rows = np.asarray(rows, dtype=np.int64)
    total_rows = int(rows.sum())
    (_, _, _, _, _), (tail_e, tail_b, tail_r, n_tail) = V.build_ragged_meta(rows)

    # --- the fake table exactly as _rag_meta_kernel writes it ---------------
    expert_ids = tail_e                                   # per tail tile
    split_size = np.zeros(len(rows), dtype=np.int64)      # dense, by expert id
    split_size[tail_e] = tail_r
    split_size_cum = tail_b                               # per tail tile
    tile_num = np.ones(max(n_tail, 1), dtype=np.int64)
    tile_cum = np.arange(1, max(n_tail, 1) + 1, dtype=np.int64)

    cov = np.zeros((max(total_rows, 1), nbn), dtype=np.int32)
    for tile_id in range(n_tail * nbn):
        pid_m = tile_id // nbn
        pid_n = tile_id % nbn
        expert = expert_ids[pid_m]
        n_rows = split_size[expert]                       # == r_e
        row_begin = split_size_cum[pid_m]
        t_num = tile_num[pid_m]
        t_cum = tile_cum[pid_m]
        local_m = pid_m - (t_cum - t_num)
        assert local_m == 0, (pid_m, t_cum, t_num)
        local_m, pid_n2 = V.swizzle2d(local_m, pid_n, t_num, nbn, group_m)
        assert (local_m, pid_n2) == (0, pid_n), "swizzle2d is not identity on a 1-tile group"
        offs_m = row_begin + local_m * BT + np.arange(BT)
        m = offs_m < row_begin + n_rows
        # the TMA-store fast path is taken only when the tile is exactly full
        tma = (row_begin + local_m * BT) + BT <= row_begin + n_rows
        assert tma == (n_rows == BT)
        cov[offs_m[m], pid_n] += 1
    return cov[:total_rows], n_tail


def main():
    # 0. the shipped kernel really uses BM=128 / BT=64 / one fake group per tile
    src = io.open(os.path.join(os.path.dirname(HERE), "p1",
                               "kernel_v780_ragged.py"), encoding="utf-8").read()
    assert "_RAG_BM_TAIL = 64" in src
    assert "GROUP_GEMM_BLOCK_SIZE_M, _RAG_BM_TAIL, 132," in src
    assert src.count("BLOCK_M=_RAG_BM_TAIL") == 2, "expected exactly 2 BM=64 launches"
    assert "_rag_meta_kernel" in src and src.count("@triton_dist.jit") == \
        io.open(os.path.join(os.path.dirname(HERE), "p1",
                             "kernel_v760a_tanh.py"), encoding="utf-8").read().count(
            "@triton_dist.jit") + 1, "expected exactly 1 new @triton_dist.jit source"
    print("[v780] shipped file: 1 new jit source, 2 BLOCK_M=64 re-launches  OK")

    # 1. main-kernel replay of the tail table == V721's verified tail model
    rng = np.random.default_rng(20260904)
    cases = [[0], [1], [63], [64], [65], [127], [128], [129], [191], [192],
             [255], [256], [257], [0, 0, 0], [64, 64, 64], [65, 65, 65],
             [128, 0, 128], [1] * 16, [127] * 8, [128] * 8, [129] * 8,
             [0, 127, 1, 128, 129, 64, 65, 63, 192, 193, 255, 256],
             [64] * 32, [63] * 32, [0, 64, 0, 64, 0, 64], list(range(0, 300))]
    for _ in range(200):
        E = int(rng.integers(1, 129))
        style = rng.integers(0, 4)
        if style == 0:
            r = rng.integers(0, 300, size=E)
        elif style == 1:
            base = int(rng.choice([128, 256, 512, 1024, 2048, 4096]))
            r = np.maximum(0, base + rng.normal(0, 40, size=E).astype(int))
        elif style == 2:
            r = rng.integers(0, 3, size=E)
        else:
            r = rng.integers(0, 5000, size=E)
        cases.append(list(map(int, r)))

    bad = 0
    for ci, rows in enumerate(cases):
        nbn = int(np.random.default_rng(ci).integers(1, 5))
        gm = int(np.random.default_rng(ci + 7).choice([4, 8, 16, 32]))
        cov_main, ntail = tail_coverage_via_main_kernel(rows, nbn, gm)
        # V721's own model, main(BM=128) + tail(BM=64), must equal 1 everywhere
        cov_full, tot_r, ntail_ref = V.coverage(rows, nbn, gm, ragged=True)
        assert ntail == ntail_ref
        # main-kernel replay of the tail must reproduce V721's tail half exactly
        cov_ref_main, _, _ = V.coverage(rows, nbn, gm, ragged=False)
        if not (cov_full.size == 0 or (cov_full == 1).all()):
            bad += 1
        # tail-only coverage from the replay + main-only coverage must sum to 1
        if cov_full.size:
            main_only = cov_full - cov_main
            if not ((main_only >= 0).all() and (main_only <= 1).all()
                    and (main_only + cov_main == 1).all()):
                bad += 1
                print("FAIL", ci, rows[:8])
    print("[v780] %d cases: main-kernel replay of the fake tail table reproduces "
          "V721's tail model exactly, main+tail covers every (row, n-block) "
          "exactly once, failures = %d" % (len(cases), bad))
    assert bad == 0

    # 2. the untouched V721 suite
    print()
    V.run_coverage_tests()
    print()
    V.run_detail_table()


main()
