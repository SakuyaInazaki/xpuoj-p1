"""GeneratedWorkload-only H800 diagnostic for c10 sorted-A TMA.

This is not a P1 submission candidate.  It builds one fixed c10 workload and
compares the active token-only FP8 A plus pointer-gather MD path against the
historical V659 sorted FP8 A plus dual-TMA MD path.  Allocation, route metadata,
GU quantization, BNORM, and descriptor construction are outside every timing.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


T = 4096
H = 4096
E = 256
I = 1536
KTOP = 8
M = 32768
GRID = 132
BM = 128
BN = 128
BK = 128
GM = 32
SQNR_THRESHOLD_DB = 35.0


@triton.jit
def setup_route_metadata_kernel(
    ORDER,
    INV_ORDER,
    W,
    COUNTS,
    EXPERT_IDS,
    SPLIT_CUM,
    TILE_NUM,
    TILE_CUM,
    NUM_TILES,
    BLOCK_R: tl.constexpr,
):
    # Thirty-two groups of eight experts.  Group sizes alternate 112/144,
    # producing real BM128 tail tiles while preserving eight experts/token.
    expert = tl.program_id(0)
    group = expert // 8
    lane = expert % 8
    parity = group % 2
    n_rows = 112 + 32 * parity
    token_start = (group // 2) * 256 + parity * 112
    offs_r = tl.arange(0, BLOCK_R)
    row_begin = 8 * token_start + lane * n_rows
    src = (token_start + offs_r) * 8 + lane
    row = row_begin + offs_r
    row_mask = offs_r < n_rows
    tl.store(ORDER + row, src, mask=row_mask)
    tl.store(INV_ORDER + src, row, mask=row_mask)
    tl.store(W + src, 0.125, mask=row_mask)
    tl.store(COUNTS + expert, n_rows)

    n_tiles = 1 + parity
    tile_begin = (group // 2) * 24 + parity * 8 + lane * n_tiles
    tile_cum = tile_begin + n_tiles
    tile_offsets = tl.arange(0, 2)
    tile_mask = tile_offsets < n_tiles
    tile_index = tile_begin + tile_offsets
    tl.store(EXPERT_IDS + tile_index, expert, mask=tile_mask)
    tl.store(SPLIT_CUM + tile_index, row_begin, mask=tile_mask)
    tl.store(TILE_NUM + tile_index, n_tiles, mask=tile_mask)
    tl.store(TILE_CUM + tile_index, tile_cum, mask=tile_mask)
    if expert == 0:
        tl.store(NUM_TILES, 384)


@triton.jit
def gq1p_tok_kernel(
    X,
    Q,
    SCALE,
    T_RUNTIME,
    H_RUNTIME,
    stride_xm,
    stride_xh,
    stride_qm,
    stride_qh,
    BLOCK_H: tl.constexpr,
):
    # Exact active p1/kernel.py _gq1p_tok_kernel math and launch shape.
    t = tl.program_id(axis=0)
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H_RUNTIME
    vals = tl.load(
        X + t * stride_xm + offs_h * stride_xh,
        mask=h_mask,
        other=0.0,
    )
    amax = tl.max(tl.abs(vals)).to(tl.float32)
    scale = tl.maximum(amax / 448.0, 1e-12)
    q = (vals.to(tl.float32) * (1.0 / scale)).to(tl.float8e4nv)
    tl.store(Q + t * stride_qm + offs_h * stride_qh, q, mask=h_mask)
    tl.store(SCALE + t, scale)


@triton.jit
def gq1p_sorted_kernel(
    X,
    INV_ORDER,
    Q,
    SCALE,
    T_RUNTIME,
    H_RUNTIME,
    stride_xm,
    stride_xh,
    stride_qm,
    stride_qh,
    K_BRANCH: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Exact active/V659 _gq1p_tm_kernel math: load and quantize each token once,
    # then scatter its identical FP8 row and scale to all sorted destinations.
    t = tl.program_id(axis=0)
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H_RUNTIME
    vals = tl.load(
        X + t * stride_xm + offs_h * stride_xh,
        mask=h_mask,
        other=0.0,
    )
    amax = tl.max(tl.abs(vals)).to(tl.float32)
    scale = tl.maximum(amax / 448.0, 1e-12)
    q = (vals.to(tl.float32) * (1.0 / scale)).to(tl.float8e4nv)
    for j in tl.static_range(K_BRANCH):
        dest = tl.load(INV_ORDER + t * K_BRANCH + j)
        tl.store(Q + dest * stride_qm + offs_h * stride_qh, q, mask=h_mask)
        tl.store(SCALE + dest, scale)


@triton.jit
def quantize_gu_kernel(
    X,
    Q,
    SCALE,
    ROW_NORM,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    row_base = row64 * K
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(X + row_base + offs_k, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.0, 1.0e-12)
    q = (x / scale).to(tl.float8e4nv)
    tl.store(Q + row_base + offs_k, q, mask=mask)
    tl.store(SCALE + row, scale)
    real = q.to(tl.float32) * scale
    tl.store(ROW_NORM + row, tl.sqrt(tl.sum(real * real, axis=0)))


@triton.jit
def expert_bnorm_kernel(
    ROW_NORM,
    BNORM,
    ROWS_PER_EXPERT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    expert = tl.program_id(0)
    offs_r = tl.arange(0, BLOCK_R)
    mask = offs_r < ROWS_PER_EXPERT
    base = expert * ROWS_PER_EXPERT
    norm = tl.load(ROW_NORM + base + offs_r, mask=mask, other=0.0)
    tl.store(BNORM + expert, tl.max(norm, axis=0))


@triton.jit
def pointer_gather_md_kernel(
    A,
    A_SCALE,
    BNORM,
    B_DESC,
    B_SCALE,
    W,
    ORDER,
    ACT,
    ROWSCALE,
    EXPERT_IDS,
    SPLIT_SIZE,
    SPLIT_CUM,
    TILE_NUM,
    TILE_CUM,
    NUM_TILES,
    M_RUNTIME,
    I_RUNTIME,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    KTOP_CONST: tl.constexpr,
):
    # Exact active c10 q8 MD structure: pointer-gather A, descriptor-load B.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I_RUNTIME, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)

    for tile_id in tl.range(
        pid,
        total_tiles * num_block_n,
        num_pid,
        num_stages=2,
    ):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n
        expert = tl.load(EXPERT_IDS + pid_m)
        n_rows = tl.load(SPLIT_SIZE + expert)
        row_begin = tl.load(SPLIT_CUM + pid_m)
        t_num = tl.load(TILE_NUM + pid_m)
        t_cum = tl.load(TILE_CUM + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(
            local_m,
            pid_n,
            t_num,
            num_block_n,
            GROUP_M,
        )

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = offs_m < row_begin + n_rows
        rows = tl.load(ORDER + offs_m, mask=row_mask, other=0) // KTOP_CONST
        a_ptrs = A + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_row_g = expert * (2 * I_RUNTIME) + pid_n * BLOCK_N
        b_row_u = b_row_g + I_RUNTIME
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            bg = B_DESC.load([b_row_g, k * BLOCK_K])
            bu = B_DESC.load([b_row_u, k * BLOCK_K])
            acc_g = tl.dot(a, bg.T, acc_g)
            acc_u = tl.dot(a, bu.T, acc_u)
            a_ptrs += BLOCK_K * stride_ak

        a_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)
        g_scale = a_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I_RUNTIME) + offs_n
        )[None, :]
        u_scale = a_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I_RUNTIME) + I_RUNTIME + offs_n
        )[None, :]
        g = acc_g * g_scale
        u = acc_u * u_scale
        src = tl.load(ORDER + offs_m, mask=row_mask, other=0)
        w = tl.load(W + src, mask=row_mask, other=0.0).to(tl.float32)
        silu = tl.fdiv(
            g,
            1.0 + tl.exp2(-g * 1.4426950408889634),
            ieee_rounding=False,
        )
        act = silu * u * w[:, None]
        bn = tl.load(BNORM + expert)
        bound = a_scale * a_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        c_ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None], eviction_policy="evict_first")
        tl.store(ROWSCALE + offs_m, s, mask=row_mask)


@triton.jit
def sorted_tma_md_kernel(
    A_DESC,
    A_SCALE,
    BNORM,
    B_DESC,
    B_SCALE,
    W,
    ORDER,
    ACT,
    ROWSCALE,
    EXPERT_IDS,
    SPLIT_SIZE,
    SPLIT_CUM,
    TILE_NUM,
    TILE_CUM,
    NUM_TILES,
    M_RUNTIME,
    I_RUNTIME,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    KTOP_CONST: tl.constexpr,
):
    # V659 structure: sorted A and B both use TensorDescriptor loads.  ORDER
    # still identifies the original branch for W; A_SCALE is in sorted rows.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I_RUNTIME, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)

    for tile_id in tl.range(
        pid,
        total_tiles * num_block_n,
        num_pid,
        num_stages=2,
    ):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n
        expert = tl.load(EXPERT_IDS + pid_m)
        n_rows = tl.load(SPLIT_SIZE + expert)
        row_begin = tl.load(SPLIT_CUM + pid_m)
        t_num = tl.load(TILE_NUM + pid_m)
        t_cum = tl.load(TILE_CUM + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(
            local_m,
            pid_n,
            t_num,
            num_block_n,
            GROUP_M,
        )

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < row_begin + n_rows
        a_row = row_begin + local_m * BLOCK_M
        b_row_g = expert * (2 * I_RUNTIME) + pid_n * BLOCK_N
        b_row_u = b_row_g + I_RUNTIME
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = A_DESC.load([a_row, k * BLOCK_K])
            bg = B_DESC.load([b_row_g, k * BLOCK_K])
            bu = B_DESC.load([b_row_u, k * BLOCK_K])
            acc_g = tl.dot(a, bg.T, acc_g)
            acc_u = tl.dot(a, bu.T, acc_u)

        a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        g_scale = a_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I_RUNTIME) + offs_n
        )[None, :]
        u_scale = a_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I_RUNTIME) + I_RUNTIME + offs_n
        )[None, :]
        g = acc_g * g_scale
        u = acc_u * u_scale
        src = tl.load(ORDER + offs_m, mask=row_mask, other=0)
        w = tl.load(W + src, mask=row_mask, other=0.0).to(tl.float32)
        silu = tl.fdiv(
            g,
            1.0 + tl.exp2(-g * 1.4426950408889634),
            ieee_rounding=False,
        )
        act = silu * u * w[:, None]
        bn = tl.load(BNORM + expert)
        bound = a_scale * a_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        c_ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None], eviction_policy="evict_first")
        tl.store(ROWSCALE + offs_m, s, mask=row_mask)


@triton.jit
def md_math_check_kernel(
    A_TOKEN,
    SCALE_TOKEN,
    A_SORTED,
    SCALE_SORTED,
    INV_ORDER,
    BNORM,
    B,
    B_SCALE,
    W,
    ORDER,
    ACT_BASE,
    ROWSCALE_BASE,
    ACT_SORTED,
    ROWSCALE_SORTED,
    METRICS,
    STATUS,
    M_CONST: tl.constexpr,
    T_CONST: tl.constexpr,
    E_CONST: tl.constexpr,
    I_CONST: tl.constexpr,
    K_CONST: tl.constexpr,
    KTOP_CONST: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Three sorted rows x one N128 segment.  Each program performs an
    # independent FP32 scalar K4096 reduction and never calls tl.dot.
    pid = tl.program_id(0)
    check_index = pid // 128
    col_in_segment = pid % 128
    branch_row = tl.where(
        check_index == 0,
        0,
        tl.where(check_index == 1, M_CONST // 2, M_CONST - 1),
    )
    expert = tl.where(
        check_index == 0,
        0,
        tl.where(check_index == 1, E_CONST // 2, E_CONST - 1),
    )
    n_start = tl.where(
        check_index == 0,
        0,
        tl.where(
            check_index == 1,
            (I_CONST // (2 * 128)) * 128,
            I_CONST - 128,
        ),
    )
    n = n_start + col_in_segment
    src = tl.load(ORDER + branch_row)
    token = src // KTOP_CONST
    offs_k = tl.arange(0, BLOCK_K)
    token64 = token.to(tl.int64)
    branch64 = branch_row.to(tl.int64)
    expert64 = expert.to(tl.int64)
    bg_row64 = expert64 * (2 * I_CONST) + n
    bu_row64 = expert64 * (2 * I_CONST) + I_CONST + n
    a = tl.load(A_TOKEN + token64 * K_CONST + offs_k).to(tl.float32)
    bg = tl.load(B + bg_row64 * K_CONST + offs_k).to(tl.float32)
    bu = tl.load(B + bu_row64 * K_CONST + offs_k).to(tl.float32)
    acc_g = tl.sum(a * bg, axis=0)
    acc_u = tl.sum(a * bu, axis=0)
    a_scale = tl.load(SCALE_TOKEN + token)
    # Reuse the same 384 programs for the prep hard gate.  Each representative
    # row covers a distinct K128 segment while the oracle below covers N128.
    prep_k_start = tl.where(
        check_index == 0,
        0,
        tl.where(check_index == 1, K_CONST // 2, K_CONST - 128),
    )
    prep_k = prep_k_start + col_in_segment
    token_q_check = tl.load(A_TOKEN + token64 * K_CONST + prep_k)
    sorted_q_check = tl.load(A_SORTED + branch64 * K_CONST + prep_k)
    sorted_a_scale = tl.load(SCALE_SORTED + branch_row)
    inverse_row = tl.load(INV_ORDER + src)
    tl.atomic_or(
        STATUS,
        32,
        mask=(token_q_check.to(tl.uint8, bitcast=True)
              != sorted_q_check.to(tl.uint8, bitcast=True)),
    )
    tl.atomic_or(STATUS, 64, mask=a_scale != sorted_a_scale)
    tl.atomic_or(STATUS, 128, mask=inverse_row != branch_row)
    g_scale = a_scale * tl.load(B_SCALE + expert * (2 * I_CONST) + n)
    u_scale = a_scale * tl.load(
        B_SCALE + expert * (2 * I_CONST) + I_CONST + n
    )
    g = acc_g * g_scale
    u = acc_u * u_scale
    w = tl.load(W + src).to(tl.float32)
    silu = tl.fdiv(
        g,
        1.0 + tl.exp2(-g * 1.4426950408889634),
        ieee_rounding=False,
    )
    raw_reference = silu * u * w

    bn = tl.load(BNORM + expert)
    bound = a_scale * a_scale * bn * bn * tl.abs(w)
    bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
    bexp = (bbits >> 23) & 0xFF
    expected_scale = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
    inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
    reference_q = (raw_reference * inv).to(tl.float8e4nv)
    reference = reference_q.to(tl.float32) * expected_scale

    actual_scale_base = tl.load(ROWSCALE_BASE + branch_row)
    actual_scale_sorted = tl.load(ROWSCALE_SORTED + branch_row)
    base_q = tl.load(ACT_BASE + branch_row * I_CONST + n)
    sorted_q = tl.load(ACT_SORTED + branch_row * I_CONST + n)
    actual_base = base_q.to(tl.float32) * actual_scale_base
    actual_sorted = sorted_q.to(tl.float32) * actual_scale_sorted

    ref2 = reference * reference
    err_base = (actual_base - reference) * (actual_base - reference)
    err_sorted = (actual_sorted - reference) * (actual_sorted - reference)
    base2 = actual_base * actual_base
    err_pair = (actual_sorted - actual_base) * (actual_sorted - actual_base)
    tl.atomic_add(METRICS + 0, ref2)
    tl.atomic_add(METRICS + 1, err_base)
    tl.atomic_add(METRICS + 2, err_sorted)
    tl.atomic_add(METRICS + 3, base2)
    tl.atomic_add(METRICS + 4, err_pair)
    reference_bits = reference_q.to(tl.uint8, bitcast=True)
    base_bits = base_q.to(tl.uint8, bitcast=True)
    sorted_bits = sorted_q.to(tl.uint8, bitcast=True)
    tl.atomic_add(METRICS + 5, (base_bits != reference_bits).to(tl.float32))
    tl.atomic_add(METRICS + 6, (sorted_bits != reference_bits).to(tl.float32))
    tl.atomic_add(METRICS + 7, (sorted_bits != base_bits).to(tl.float32))

    tl.atomic_or(STATUS, 1, mask=actual_scale_base != expected_scale)
    tl.atomic_or(STATUS, 2, mask=actual_scale_sorted != expected_scale)
    ref_finite = (reference == reference) & (tl.abs(reference) <= 3.402823466e38)
    base_finite = (actual_base == actual_base) & (tl.abs(actual_base) <= 3.402823466e38)
    sorted_finite = (actual_sorted == actual_sorted) & (tl.abs(actual_sorted) <= 3.402823466e38)
    tl.atomic_or(STATUS, 4, mask=ref_finite == 0)
    tl.atomic_or(STATUS, 8, mask=base_finite == 0)
    tl.atomic_or(STATUS, 16, mask=sorted_finite == 0)


@triton.jit
def finalize_metrics_kernel(METRICS, SUMMARY, CHECK_COUNT: tl.constexpr):
    ref2 = tl.load(METRICS + 0)
    err_base = tl.load(METRICS + 1)
    err_sorted = tl.load(METRICS + 2)
    base2 = tl.load(METRICS + 3)
    err_pair = tl.load(METRICS + 4)
    mismatch_base = tl.load(METRICS + 5)
    mismatch_sorted = tl.load(METRICS + 6)
    mismatch_pair = tl.load(METRICS + 7)
    safe_ref = tl.maximum(ref2, 1.0e-30)
    safe_base = tl.maximum(base2, 1.0e-30)
    # Report floor: relative error >=1e-6 and SQNR <=120 dB.
    safe_base_err = tl.maximum(err_base, safe_ref * 1.0e-12)
    safe_sorted_err = tl.maximum(err_sorted, safe_ref * 1.0e-12)
    safe_pair_err = tl.maximum(err_pair, safe_base * 1.0e-12)
    tl.store(SUMMARY + 0, tl.sqrt(safe_base_err / safe_ref))
    tl.store(SUMMARY + 1, 3.010299956639812 * tl.log2(safe_ref / safe_base_err))
    tl.store(SUMMARY + 2, tl.sqrt(safe_sorted_err / safe_ref))
    tl.store(SUMMARY + 3, 3.010299956639812 * tl.log2(safe_ref / safe_sorted_err))
    tl.store(SUMMARY + 4, tl.sqrt(safe_pair_err / safe_base))
    tl.store(SUMMARY + 5, 3.010299956639812 * tl.log2(safe_base / safe_pair_err))
    tl.store(SUMMARY + 6, 1.0 - mismatch_base / CHECK_COUNT)
    tl.store(SUMMARY + 7, 1.0 - mismatch_sorted / CHECK_COUNT)
    tl.store(SUMMARY + 8, 1.0 - mismatch_pair / CHECK_COUNT)


def launch_token_prep(x, q, scale):
    return gq1p_tok_kernel[(T,)](
        x,
        q,
        scale,
        T,
        H,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        BLOCK_H=H,
        num_warps=4,
        num_stages=1,
    )


def launch_sorted_prep(x, inv_order, q, scale):
    return gq1p_sorted_kernel[(T,)](
        x,
        inv_order,
        q,
        scale,
        T,
        H,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        K_BRANCH=KTOP,
        BLOCK_H=H,
        num_warps=4,
        num_stages=1,
    )


def launch_pointer_md(
    a_q,
    a_s,
    b_desc,
    b_s,
    bnorm,
    weights,
    order,
    out,
    rowscale,
    expert_ids,
    counts,
    split_cum,
    tile_num,
    tile_cum,
    num_tiles,
):
    return pointer_gather_md_kernel[(GRID,)](
        a_q,
        a_s,
        bnorm,
        b_desc,
        b_s,
        weights,
        order,
        out,
        rowscale,
        expert_ids,
        counts,
        split_cum,
        tile_num,
        tile_cum,
        num_tiles,
        M_RUNTIME=M,
        I_RUNTIME=I,
        K=H,
        stride_am=a_q.stride(0),
        stride_ak=a_q.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GM,
        KTOP_CONST=KTOP,
        num_warps=8,
        num_stages=4,
    )


def launch_sorted_md(
    a_desc,
    a_q,
    a_s,
    b_desc,
    b_s,
    bnorm,
    weights,
    order,
    out,
    rowscale,
    expert_ids,
    counts,
    split_cum,
    tile_num,
    tile_cum,
    num_tiles,
):
    return sorted_tma_md_kernel[(GRID,)](
        a_desc,
        a_s,
        bnorm,
        b_desc,
        b_s,
        weights,
        order,
        out,
        rowscale,
        expert_ids,
        counts,
        split_cum,
        tile_num,
        tile_cum,
        num_tiles,
        M_RUNTIME=M,
        I_RUNTIME=I,
        K=H,
        stride_am=a_q.stride(0),
        stride_ak=a_q.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GM,
        KTOP_CONST=KTOP,
        num_warps=8,
        num_stages=4,
    )


def run_diagnostic():
    device = "cuda"
    weights = torch.empty((M,), dtype=torch.float32, device=device)
    # Active P1 counting sort already produces both tensors as int64.  Their
    # construction is therefore common setup, rather than a sorted-A cost.
    order = torch.empty((M,), dtype=torch.int64, device=device)
    inv_order = torch.empty((M,), dtype=torch.int64, device=device)
    counts = torch.empty((E,), dtype=torch.int32, device=device)
    metadata_size = triton.cdiv(M, BM) + E
    expert_ids = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    split_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_num = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    num_tiles = torch.empty((1,), dtype=torch.int32, device=device)
    setup_route_metadata_kernel[(E,)](
        order,
        inv_order,
        weights,
        counts,
        expert_ids,
        split_cum,
        tile_num,
        tile_cum,
        num_tiles,
        BLOCK_R=256,
        num_warps=4,
        num_stages=1,
    )

    x = torch.randn((T, H), dtype=torch.bfloat16, device=device)
    gu = torch.randn((E, 2 * I, H), dtype=torch.bfloat16, device=device)
    b_q = torch.empty((E * 2 * I, H), dtype=torch.float8_e4m3fn, device=device)
    b_s_flat = torch.empty((E * 2 * I,), dtype=torch.float32, device=device)
    b_row_norm = torch.empty((E * 2 * I,), dtype=torch.float32, device=device)
    bnorm = torch.empty((E,), dtype=torch.float32, device=device)
    quantize_gu_kernel[(E * 2 * I,)](
        gu,
        b_q,
        b_s_flat,
        b_row_norm,
        K=H,
        BLOCK_K=H,
        num_warps=8,
        num_stages=1,
    )
    expert_bnorm_kernel[(E,)](
        b_row_norm,
        bnorm,
        ROWS_PER_EXPERT=2 * I,
        BLOCK_R=4096,
        num_warps=8,
        num_stages=1,
    )
    del gu
    del b_row_norm
    b_s = b_s_flat.view(E, 2 * I)
    b_desc = TensorDescriptor(b_q, b_q.shape, b_q.stride(), [BN, BK])

    a_token = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=device)
    scale_token = torch.empty((T,), dtype=torch.float32, device=device)
    a_sorted = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=device)
    scale_sorted = torch.empty((M,), dtype=torch.float32, device=device)
    a_sorted_desc = TensorDescriptor(
        a_sorted,
        a_sorted.shape,
        a_sorted.stride(),
        [BM, BK],
    )
    out_base = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    out_sorted = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    rowscale_base = torch.empty((M,), dtype=torch.float32, device=device)
    rowscale_sorted = torch.empty((M,), dtype=torch.float32, device=device)

    def call_base_prep():
        return launch_token_prep(x, a_token, scale_token)

    def call_sorted_prep():
        return launch_sorted_prep(x, inv_order, a_sorted, scale_sorted)

    def call_base_md():
        return launch_pointer_md(
            a_token,
            scale_token,
            b_desc,
            b_s,
            bnorm,
            weights,
            order,
            out_base,
            rowscale_base,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
        )

    def call_sorted_md():
        return launch_sorted_md(
            a_sorted_desc,
            a_sorted,
            scale_sorted,
            b_desc,
            b_s,
            bnorm,
            weights,
            order,
            out_sorted,
            rowscale_sorted,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
        )

    def call_base_total():
        launch_token_prep(x, a_token, scale_token)
        return call_base_md()

    def call_sorted_total():
        launch_sorted_prep(x, inv_order, a_sorted, scale_sorted)
        return call_sorted_md()

    compiled_base_prep = call_base_prep()
    compiled_sorted_prep = call_sorted_prep()
    compiled_base_md = call_base_md()
    compiled_sorted_md = call_sorted_md()

    status = torch.zeros((1,), dtype=torch.int32, device=device)
    metrics = torch.zeros((8,), dtype=torch.float32, device=device)
    summary = torch.empty((9,), dtype=torch.float32, device=device)
    md_math_check_kernel[(3 * BN,)](
        a_token,
        scale_token,
        a_sorted,
        scale_sorted,
        inv_order,
        bnorm,
        b_q,
        b_s,
        weights,
        order,
        out_base,
        rowscale_base,
        out_sorted,
        rowscale_sorted,
        metrics,
        status,
        M_CONST=M,
        T_CONST=T,
        E_CONST=E,
        I_CONST=I,
        K_CONST=H,
        KTOP_CONST=KTOP,
        BLOCK_K=H,
        num_warps=8,
        num_stages=1,
    )
    finalize_metrics_kernel[(1,)](
        metrics,
        summary,
        CHECK_COUNT=3 * BN,
        num_warps=1,
        num_stages=1,
    )
    status_code = int(status[0].item())
    base_rel = float(summary[0].item())
    base_sqnr = float(summary[1].item())
    sorted_rel = float(summary[2].item())
    sorted_sqnr = float(summary[3].item())
    pair_rel = float(summary[4].item())
    pair_sqnr = float(summary[5].item())
    base_exact = float(summary[6].item())
    sorted_exact = float(summary[7].item())
    pair_exact = float(summary[8].item())
    math_passed = (
        status_code == 0
        and base_sqnr >= SQNR_THRESHOLD_DB
        and sorted_sqnr >= SQNR_THRESHOLD_DB
    )
    if not math_passed:
        print(
            'P1MD {"math_check_passed":false,"status":'
            + str(status_code)
            + ',"base_sqnr_db":' + str(base_sqnr)
            + ',"sorted_sqnr_db":' + str(sorted_sqnr)
            + ',"base_rel":' + str(base_rel)
            + ',"sorted_rel":' + str(sorted_rel)
            + ',"pair_rel":' + str(pair_rel)
            + ',"pair_sqnr_db":' + str(pair_sqnr)
            + ',"base_exact_fraction":' + str(base_exact)
            + ',"sorted_exact_fraction":' + str(sorted_exact)
            + ',"pair_exact_fraction":' + str(pair_exact)
            + ',"threshold_db":35.0}'
        )
        raise RuntimeError("P1MD sorted-A math check failed")

    groups_text = []
    for group in range(3):
        if group % 2 == 0:
            base_prep_ms = float(triton.testing.do_bench(call_base_prep, warmup=5, rep=10))
            sorted_prep_ms = float(triton.testing.do_bench(call_sorted_prep, warmup=5, rep=10))
            base_md_ms = float(triton.testing.do_bench(call_base_md, warmup=5, rep=10))
            sorted_md_ms = float(triton.testing.do_bench(call_sorted_md, warmup=5, rep=10))
            base_total_ms = float(triton.testing.do_bench(call_base_total, warmup=5, rep=10))
            sorted_total_ms = float(triton.testing.do_bench(call_sorted_total, warmup=5, rep=10))
            order_name = "base_then_sorted"
        else:
            sorted_prep_ms = float(triton.testing.do_bench(call_sorted_prep, warmup=5, rep=10))
            base_prep_ms = float(triton.testing.do_bench(call_base_prep, warmup=5, rep=10))
            sorted_md_ms = float(triton.testing.do_bench(call_sorted_md, warmup=5, rep=10))
            base_md_ms = float(triton.testing.do_bench(call_base_md, warmup=5, rep=10))
            sorted_total_ms = float(triton.testing.do_bench(call_sorted_total, warmup=5, rep=10))
            base_total_ms = float(triton.testing.do_bench(call_base_total, warmup=5, rep=10))
            order_name = "sorted_then_base"
        groups_text.append(
            '{"group":' + str(group)
            + ',"order":"' + order_name
            + '","base_prep_ms":' + str(base_prep_ms)
            + ',"sorted_prep_ms":' + str(sorted_prep_ms)
            + ',"sorted_over_base_prep":' + str(sorted_prep_ms / base_prep_ms)
            + ',"base_md_ms":' + str(base_md_ms)
            + ',"sorted_md_ms":' + str(sorted_md_ms)
            + ',"sorted_over_base_md":' + str(sorted_md_ms / base_md_ms)
            + ',"base_prep_md_ms":' + str(base_total_ms)
            + ',"sorted_prep_md_ms":' + str(sorted_total_ms)
            + ',"sorted_over_base_prep_md":' + str(sorted_total_ms / base_total_ms)
            + '}'
        )

    report = (
        '{"kind":"fp8_c10_sorted_a_custom","shape":{"T":4096,"H":4096,'
        '"E":256,"I":1536,"topk":8,"M":32768},'
        '"geometry":{"grid":132,"BM":128,"BN":128,"BK":128,"GM":32,'
        '"warps":8,"stages":4,"persistent_range_stages":2},'
        '"route":{"kind":"analytic_112_144_alternating","tiles":384,'
        '"inv_order":"shared_existing_P1_route_artifact_setup_excluded"},'
        '"paths":{"base":"token_FP8_A_pointer_gather_plus_TMA_B",'
        '"sorted":"sorted_FP8_A_dual_TMA_A_B"},'
        '"prep":{"base":"active_gq1p_tok_preallocated",'
        '"sorted":"V659_gq1p_tm_preallocated_load_quantize_once_scatter8"},'
        '"math_check":{"comparisons":384,"passed":true,"status":'
        + str(status_code)
        + ',"threshold_db":35.0,"a_prep_fp8_and_scale_bitwise":true,'
        + '"order_inverse_exact":true,"base_rel":' + str(base_rel)
        + ',"base_sqnr_db":' + str(base_sqnr)
        + ',"sorted_rel":' + str(sorted_rel)
        + ',"sorted_sqnr_db":' + str(sorted_sqnr)
        + ',"sorted_vs_base_rel":' + str(pair_rel)
        + ',"sorted_vs_base_sqnr_db":' + str(pair_sqnr)
        + ',"base_exact_fraction":' + str(base_exact)
        + ',"sorted_exact_fraction":' + str(sorted_exact)
        + ',"sorted_vs_base_exact_fraction":' + str(pair_exact)
        + ',"rowscale_exact":{"base":true,"sorted":true},'
        + '"finite":{"reference":true,"base":true,"sorted":true}},'
        '"resources":{"base_prep":{"n_regs":' + str(compiled_base_prep.n_regs)
        + ',"n_spills":' + str(compiled_base_prep.n_spills)
        + ',"shared":' + str(compiled_base_prep.metadata.shared)
        + '},"sorted_prep":{"n_regs":' + str(compiled_sorted_prep.n_regs)
        + ',"n_spills":' + str(compiled_sorted_prep.n_spills)
        + ',"shared":' + str(compiled_sorted_prep.metadata.shared)
        + '},"base_md":{"n_regs":' + str(compiled_base_md.n_regs)
        + ',"n_spills":' + str(compiled_base_md.n_spills)
        + ',"shared":' + str(compiled_base_md.metadata.shared)
        + '},"sorted_md":{"n_regs":' + str(compiled_sorted_md.n_regs)
        + ',"n_spills":' + str(compiled_sorted_md.n_spills)
        + ',"shared":' + str(compiled_sorted_md.metadata.shared) + '}},'
        '"timing":{"method":"do_bench_warmup5_rep10","allocation_excluded":true,'
        '"route_GU_quant_BNORM_descriptors_excluded":true,'
        '"prep_and_MD_measured_separately_and_combined":true,"decision_metric":'
        '"prep_plus_MD","groups":[' + ",".join(groups_text) + ']},'
        '"limitations":"custom_single_GPU_c10_only_no_DN_no_P1_integration"}'
    )
    print("P1MD " + report)
    raise RuntimeError("P1MD done")


def run_kernel(x):
    return x


run_diagnostic()
