"""GeneratedWorkload-only H800 diagnostic for c10 FP16 accumulation.

This is deliberately not a P1 submission candidate.  It builds one fixed c10
workload and compares the existing row-amax FP8/F32 MD path with row-L2 FP8/F32
and row-L2 FP8/FP16-accumulator paths.  Setup is outside all timings.
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
BM_HALF = 256
BN = 128
BK = 128
GM = 32
FP8_MAX = 448.0
L2_TARGET = 128.0
SQNR_THRESHOLD_DB = 35.0


@triton.jit
def setup_route_metadata_kernel(
    ORDER,
    W,
    COUNTS,
    EXPERT_IDS,
    SPLIT_CUM,
    TILE_NUM,
    TILE_CUM,
    NUM_TILES,
    EXPERT_IDS_256,
    SPLIT_CUM_256,
    TILE_NUM_256,
    TILE_CUM_256,
    NUM_TILES_256,
    BLOCK_R: tl.constexpr,
):
    # Thirty-two groups of eight experts.  Group sizes alternate 112/144,
    # producing real tail tiles while preserving eight unique experts/token.
    expert = tl.program_id(0)
    group = expert // 8
    lane = expert % 8
    parity = group % 2
    n_rows = 112 + 32 * parity
    token_start = (group // 2) * 256 + parity * 112
    row_begin = 8 * token_start + lane * n_rows
    src = (token_start + tl.arange(0, BLOCK_R)) * 8 + lane
    row = row_begin + tl.arange(0, BLOCK_R)
    row_mask = tl.arange(0, BLOCK_R) < n_rows
    tl.store(ORDER + row, src, mask=row_mask)
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

    # Every 112/144-row expert is one complete BM256 tile.
    tl.store(EXPERT_IDS_256 + expert, expert)
    tl.store(SPLIT_CUM_256 + expert, row_begin)
    tl.store(TILE_NUM_256 + expert, 1)
    tl.store(TILE_CUM_256 + expert, expert + 1)
    if expert == 0:
        tl.store(NUM_TILES_256, 256)


@triton.jit
def quantize_rows_kernel(
    X,
    Q_AMAX,
    SCALE_AMAX,
    NORM_AMAX,
    Q_L2,
    SCALE_L2,
    NORM_L2,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    row_base = row64 * K
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(X + row_base + offs_k, mask=mask, other=0.0).to(tl.float32)
    source_l2 = tl.sqrt(tl.sum(x * x, axis=0))
    scale_amax = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.0, 1.0e-12)
    scale_l2 = tl.maximum(source_l2 / 128.0, 1.0e-12)
    q_amax = (x / scale_amax).to(tl.float8e4nv)
    q_l2 = (x / scale_l2).to(tl.float8e4nv)
    tl.store(Q_AMAX + row_base + offs_k, q_amax, mask=mask)
    tl.store(Q_L2 + row_base + offs_k, q_l2, mask=mask)
    tl.store(SCALE_AMAX + row, scale_amax)
    tl.store(SCALE_L2 + row, scale_l2)
    real_amax = q_amax.to(tl.float32) * scale_amax
    real_l2 = q_l2.to(tl.float32) * scale_l2
    tl.store(NORM_AMAX + row, tl.sqrt(tl.sum(real_amax * real_amax, axis=0)))
    tl.store(NORM_L2 + row, tl.sqrt(tl.sum(real_l2 * real_l2, axis=0)))


@triton.jit
def expert_bnorm_kernel(
    ROW_NORM_AMAX,
    ROW_NORM_L2,
    BNORM_AMAX,
    BNORM_L2,
    ROWS_PER_EXPERT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    expert = tl.program_id(0)
    offs_r = tl.arange(0, BLOCK_R)
    mask = offs_r < ROWS_PER_EXPERT
    base = expert * ROWS_PER_EXPERT
    norm_amax = tl.load(ROW_NORM_AMAX + base + offs_r, mask=mask, other=0.0)
    norm_l2 = tl.load(ROW_NORM_L2 + base + offs_r, mask=mask, other=0.0)
    tl.store(BNORM_AMAX + expert, tl.max(norm_amax, axis=0))
    tl.store(BNORM_L2 + expert, tl.max(norm_l2, axis=0))


@triton.jit
def fp8_md_kernel(
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
    M,
    I,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    KTOP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, num_stages=2):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n
        expert = tl.load(EXPERT_IDS + pid_m)
        n_rows = tl.load(SPLIT_SIZE + expert)
        row_begin = tl.load(SPLIT_CUM + pid_m)
        t_num = tl.load(TILE_NUM + pid_m)
        t_cum = tl.load(TILE_CUM + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = offs_m < row_begin + n_rows
        rows = tl.load(ORDER + offs_m, mask=row_mask, other=0) // KTOP

        a_ptrs = A + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_row_g = expert * (2 * I) + pid_n * BLOCK_N
        b_row_u = b_row_g + I
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
        g_scale = a_scale[:, None] * tl.load(B_SCALE + expert * (2 * I) + offs_n)[None, :]
        u_scale = a_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I) + I + offs_n
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
def l2_f32_md_kernel(
    A,
    A_SCALE,
    A_BOUND_SCALE,
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
    M,
    I,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    KTOP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, num_stages=2):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n
        expert = tl.load(EXPERT_IDS + pid_m)
        n_rows = tl.load(SPLIT_SIZE + expert)
        row_begin = tl.load(SPLIT_CUM + pid_m)
        t_num = tl.load(TILE_NUM + pid_m)
        t_cum = tl.load(TILE_CUM + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = offs_m < row_begin + n_rows
        rows = tl.load(ORDER + offs_m, mask=row_mask, other=0) // KTOP

        a_ptrs = A + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_row_g = expert * (2 * I) + pid_n * BLOCK_N
        b_row_u = b_row_g + I
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

        a_deq_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)
        a_bound_scale = tl.load(A_BOUND_SCALE + rows, mask=row_mask, other=1.0)
        g_scale = a_deq_scale[:, None] * tl.load(B_SCALE + expert * (2 * I) + offs_n)[None, :]
        u_scale = a_deq_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I) + I + offs_n
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
        bound = a_bound_scale * a_bound_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        c_ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None], eviction_policy="evict_first")
        tl.store(ROWSCALE + offs_m, s, mask=row_mask)


@triton.jit
def l2_f16_md_kernel(
    A,
    A_SCALE,
    A_BOUND_SCALE,
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
    M,
    I,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    KTOP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, num_stages=2):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n
        expert = tl.load(EXPERT_IDS + pid_m)
        n_rows = tl.load(SPLIT_SIZE + expert)
        row_begin = tl.load(SPLIT_CUM + pid_m)
        t_num = tl.load(TILE_NUM + pid_m)
        t_cum = tl.load(TILE_CUM + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = offs_m < row_begin + n_rows
        rows = tl.load(ORDER + offs_m, mask=row_mask, other=0) // KTOP

        a_ptrs = A + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_row_g = expert * (2 * I) + pid_n * BLOCK_N
        b_row_u = b_row_g + I
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            bg = B_DESC.load([b_row_g, k * BLOCK_K])
            bu = B_DESC.load([b_row_u, k * BLOCK_K])
            acc_g = tl.dot(a, bg.T, acc_g, out_dtype=tl.float16)
            acc_u = tl.dot(a, bu.T, acc_u, out_dtype=tl.float16)
            a_ptrs += BLOCK_K * stride_ak

        a_deq_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)
        a_bound_scale = tl.load(A_BOUND_SCALE + rows, mask=row_mask, other=1.0)
        g_scale = a_deq_scale[:, None] * tl.load(B_SCALE + expert * (2 * I) + offs_n)[None, :]
        u_scale = a_deq_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I) + I + offs_n
        )[None, :]
        g = acc_g.to(tl.float32) * g_scale
        u = acc_u.to(tl.float32) * u_scale
        src = tl.load(ORDER + offs_m, mask=row_mask, other=0)
        w = tl.load(W + src, mask=row_mask, other=0.0).to(tl.float32)
        silu = tl.fdiv(
            g,
            1.0 + tl.exp2(-g * 1.4426950408889634),
            ieee_rounding=False,
        )
        act = silu * u * w[:, None]

        bn = tl.load(BNORM + expert)
        bound = a_bound_scale * a_bound_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        c_ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None], eviction_policy="evict_first")
        tl.store(ROWSCALE + offs_m, s, mask=row_mask)


@triton.jit
def l2_math_check_kernel(
    A,
    A_SCALE,
    A_BOUND_SCALE,
    BNORM,
    B,
    B_SCALE,
    ACT_F32,
    ROWSCALE_F32,
    ACT_HALF,
    ROWSCALE_HALF,
    METRICS,
    STATUS,
    M: tl.constexpr,
    T: tl.constexpr,
    E: tl.constexpr,
    I: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Three routed rows x one corresponding N128 segment.  Each program uses
    # an independent scalar FP32 K4096 reduction, never tl.dot.
    pid = tl.program_id(0)
    check_index = pid // 128
    col_in_segment = pid % 128
    branch_row = tl.where(check_index == 0, 0, tl.where(check_index == 1, M // 2, M - 1))
    token = tl.where(check_index == 0, 0, tl.where(check_index == 1, T // 2, T - 1))
    expert = tl.where(check_index == 0, 0, tl.where(check_index == 1, E // 2, E - 1))
    n_start = tl.where(
        check_index == 0,
        0,
        tl.where(check_index == 1, (I // (2 * 128)) * 128, I - 128),
    )
    n = n_start + col_in_segment
    offs_k = tl.arange(0, BLOCK_K)
    token64 = token.to(tl.int64)
    expert64 = expert.to(tl.int64)
    bg_row64 = expert64 * (2 * I) + n
    bu_row64 = expert64 * (2 * I) + I + n
    a = tl.load(A + token64 * K + offs_k).to(tl.float32)
    bg = tl.load(B + bg_row64 * K + offs_k).to(tl.float32)
    bu = tl.load(B + bu_row64 * K + offs_k).to(tl.float32)
    acc_g = tl.sum(a * bg, axis=0)
    acc_u = tl.sum(a * bu, axis=0)
    a_scale = tl.load(A_SCALE + token)
    g_scale = a_scale * tl.load(B_SCALE + expert * (2 * I) + n)
    u_scale = a_scale * tl.load(B_SCALE + expert * (2 * I) + I + n)
    g = acc_g * g_scale
    u = acc_u * u_scale
    silu = tl.fdiv(
        g,
        1.0 + tl.exp2(-g * 1.4426950408889634),
        ieee_rounding=False,
    )
    raw_reference = silu * u * 0.125

    bound_a_scale = tl.load(A_BOUND_SCALE + token)
    bn = tl.load(BNORM + expert)
    bound = bound_a_scale * bound_a_scale * bn * bn * 0.125
    bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
    bexp = (bbits >> 23) & 0xFF
    expected_scale = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
    inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
    reference_q = (raw_reference * inv).to(tl.float8e4nv)
    reference = reference_q.to(tl.float32) * expected_scale

    actual_scale_f32 = tl.load(ROWSCALE_F32 + branch_row)
    actual_scale_half = tl.load(ROWSCALE_HALF + branch_row)
    actual_q_f32 = tl.load(ACT_F32 + branch_row * I + n)
    actual_q_half = tl.load(ACT_HALF + branch_row * I + n)
    actual_f32 = actual_q_f32.to(tl.float32) * actual_scale_f32
    actual_half = actual_q_half.to(tl.float32) * actual_scale_half

    ref2 = reference * reference
    err_f32 = (actual_f32 - reference) * (actual_f32 - reference)
    err_half = (actual_half - reference) * (actual_half - reference)
    norm2 = actual_f32 * actual_f32
    err_pair = (actual_half - actual_f32) * (actual_half - actual_f32)
    tl.atomic_add(METRICS + 0, ref2)
    tl.atomic_add(METRICS + 1, err_f32)
    tl.atomic_add(METRICS + 2, err_half)
    tl.atomic_add(METRICS + 3, norm2)
    tl.atomic_add(METRICS + 4, err_pair)
    reference_bits = reference_q.to(tl.uint8, bitcast=True)
    tl.atomic_add(
        METRICS + 5,
        (actual_q_f32.to(tl.uint8, bitcast=True) != reference_bits).to(tl.float32),
    )
    tl.atomic_add(
        METRICS + 6,
        (actual_q_half.to(tl.uint8, bitcast=True) != reference_bits).to(tl.float32),
    )

    tl.atomic_or(STATUS, 1, mask=actual_scale_f32 != expected_scale)
    tl.atomic_or(STATUS, 2, mask=actual_scale_half != expected_scale)
    ref_finite = (reference == reference) & (tl.abs(reference) <= 3.402823466e38)
    f32_finite = (actual_f32 == actual_f32) & (tl.abs(actual_f32) <= 3.402823466e38)
    half_finite = (actual_half == actual_half) & (tl.abs(actual_half) <= 3.402823466e38)
    tl.atomic_or(STATUS, 4, mask=ref_finite == 0)
    tl.atomic_or(STATUS, 8, mask=f32_finite == 0)
    tl.atomic_or(STATUS, 16, mask=half_finite == 0)


@triton.jit
def finalize_metrics_kernel(METRICS, SUMMARY, CHECK_COUNT: tl.constexpr):
    ref2 = tl.load(METRICS + 0)
    err_f32 = tl.load(METRICS + 1)
    err_half = tl.load(METRICS + 2)
    norm2 = tl.load(METRICS + 3)
    err_pair = tl.load(METRICS + 4)
    mismatch_f32 = tl.load(METRICS + 5)
    mismatch_half = tl.load(METRICS + 6)
    safe_ref = tl.maximum(ref2, 1.0e-30)
    safe_norm = tl.maximum(norm2, 1.0e-30)
    # Report floor: relative error >=1e-6 and SQNR <=120 dB.
    safe_f32_err = tl.maximum(err_f32, safe_ref * 1.0e-12)
    safe_half_err = tl.maximum(err_half, safe_ref * 1.0e-12)
    safe_pair_err = tl.maximum(err_pair, safe_norm * 1.0e-12)
    tl.store(SUMMARY + 0, tl.sqrt(safe_f32_err / safe_ref))
    tl.store(SUMMARY + 1, 3.010299956639812 * tl.log2(safe_ref / safe_f32_err))
    tl.store(SUMMARY + 2, tl.sqrt(safe_half_err / safe_ref))
    tl.store(SUMMARY + 3, 3.010299956639812 * tl.log2(safe_ref / safe_half_err))
    tl.store(SUMMARY + 4, tl.sqrt(safe_pair_err / safe_norm))
    tl.store(SUMMARY + 5, 3.010299956639812 * tl.log2(safe_norm / safe_pair_err))
    tl.store(SUMMARY + 6, 1.0 - mismatch_f32 / CHECK_COUNT)
    tl.store(SUMMARY + 7, 1.0 - mismatch_half / CHECK_COUNT)


def quantize_rows_dual(x):
    rows = x.shape[0]
    q_amax = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device="cuda")
    q_l2 = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device="cuda")
    scale_amax = torch.empty((rows,), dtype=torch.float32, device="cuda")
    scale_l2 = torch.empty((rows,), dtype=torch.float32, device="cuda")
    norm_amax = torch.empty((rows,), dtype=torch.float32, device="cuda")
    norm_l2 = torch.empty((rows,), dtype=torch.float32, device="cuda")
    quantize_rows_kernel[(rows,)](
        x,
        q_amax,
        scale_amax,
        norm_amax,
        q_l2,
        scale_l2,
        norm_l2,
        K=H,
        BLOCK_K=H,
        num_warps=8,
        num_stages=1,
    )
    return q_amax, scale_amax, norm_amax, q_l2, scale_l2, norm_l2


def reduce_expert_bnorm(norm_amax, norm_l2):
    bnorm_amax = torch.empty((E,), dtype=torch.float32, device="cuda")
    bnorm_l2 = torch.empty((E,), dtype=torch.float32, device="cuda")
    expert_bnorm_kernel[(E,)](
        norm_amax,
        norm_l2,
        bnorm_amax,
        bnorm_l2,
        ROWS_PER_EXPERT=2 * I,
        BLOCK_R=4096,
        num_warps=8,
        num_stages=1,
    )
    return bnorm_amax, bnorm_l2


def launch_baseline(
    a_q, a_s, b_q, b_s, bnorm, weights, order, out, rowscale,
    expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
):
    return fp8_md_kernel[(GRID,)](
        a_q, a_s, bnorm, b_q, b_s, weights, order, out, rowscale,
        expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M=M, I=I, K=H,
        stride_am=a_q.stride(0), stride_ak=a_q.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, KTOP=KTOP,
        num_warps=8, num_stages=4,
    )


def launch_l2_f32(
    a_q, a_s, a_bound_s, b_q, b_s, bnorm, weights, order, out, rowscale,
    expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
):
    return l2_f32_md_kernel[(GRID,)](
        a_q, a_s, a_bound_s, bnorm, b_q, b_s, weights, order, out, rowscale,
        expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M=M, I=I, K=H,
        stride_am=a_q.stride(0), stride_ak=a_q.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, KTOP=KTOP,
        num_warps=8, num_stages=4,
    )


def launch_l2_half(
    a_q, a_s, a_bound_s, b_q, b_s, bnorm, weights, order, out, rowscale,
    expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
):
    return l2_f16_md_kernel[(GRID,)](
        a_q, a_s, a_bound_s, bnorm, b_q, b_s, weights, order, out, rowscale,
        expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M=M, I=I, K=H,
        stride_am=a_q.stride(0), stride_ak=a_q.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_M=BM_HALF, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, KTOP=KTOP,
        num_warps=8, num_stages=3,
    )


def check_l2_math(
    a_q, a_s, a_bound_s, b_q, b_s, bnorm,
    out_f32, rowscale_f32, out_half, rowscale_half, metrics, status,
):
    return l2_math_check_kernel[(3 * BN,)](
        a_q, a_s, a_bound_s, bnorm, b_q, b_s,
        out_f32, rowscale_f32, out_half, rowscale_half, metrics, status,
        M=M, T=T, E=E, I=I, K=H, BLOCK_K=H,
        num_warps=8, num_stages=1,
    )


def run_diagnostic():
    device = "cuda"
    weights = torch.empty((M,), dtype=torch.float32, device=device)
    order = torch.empty((M,), dtype=torch.int32, device=device)
    counts = torch.empty((E,), dtype=torch.int32, device=device)

    metadata_size = triton.cdiv(M, BM) + E
    expert_ids = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    split_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_num = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    num_tiles = torch.empty((1,), dtype=torch.int32, device=device)

    expert_ids_256 = torch.empty((E,), dtype=torch.int32, device=device)
    split_cum_256 = torch.empty((E,), dtype=torch.int32, device=device)
    tile_num_256 = torch.empty((E,), dtype=torch.int32, device=device)
    tile_cum_256 = torch.empty((E,), dtype=torch.int32, device=device)
    num_tiles_256 = torch.empty((1,), dtype=torch.int32, device=device)
    setup_route_metadata_kernel[(E,)](
        order, weights, counts,
        expert_ids, split_cum, tile_num, tile_cum, num_tiles,
        expert_ids_256, split_cum_256, tile_num_256, tile_cum_256, num_tiles_256,
        BLOCK_R=256, num_warps=4, num_stages=1,
    )

    x = torch.randn((T, H), dtype=torch.bfloat16, device=device)
    gu = torch.randn((E, 2 * I, H), dtype=torch.bfloat16, device=device)
    a8, a8s, an8, al2, al2s, anl2 = quantize_rows_dual(x)
    b8, b8s_flat, bnorm_rows8, bl2, bl2s_flat, bnorm_rows_l2 = quantize_rows_dual(
        gu.view(E * 2 * I, H)
    )
    b8s = b8s_flat.view(E, 2 * I)
    bl2s = bl2s_flat.view(E, 2 * I)
    bn8, bnl2 = reduce_expert_bnorm(bnorm_rows8, bnorm_rows_l2)
    del x
    del gu
    del an8
    del anl2
    del bnorm_rows8
    del bnorm_rows_l2

    b8_flat = b8.view(E * 2 * I, H)
    bl2_flat = bl2.view(E * 2 * I, H)
    b8_desc = TensorDescriptor(b8_flat, b8_flat.shape, b8_flat.stride(), [BN, BK])
    bl2_desc = TensorDescriptor(bl2_flat, bl2_flat.shape, bl2_flat.stride(), [BN, BK])
    out_base = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    out_f32 = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    out_half = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    rowscale_base = torch.empty((M,), dtype=torch.float32, device=device)
    rowscale_f32 = torch.empty((M,), dtype=torch.float32, device=device)
    rowscale_half = torch.empty((M,), dtype=torch.float32, device=device)

    def call_base():
        return launch_baseline(
            a8, a8s, b8_desc, b8s, bn8, weights, order, out_base, rowscale_base,
            expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        )

    def call_f32():
        return launch_l2_f32(
            al2, al2s, a8s, bl2_desc, bl2s, bnl2, weights, order, out_f32, rowscale_f32,
            expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        )

    def call_half():
        return launch_l2_half(
            al2, al2s, a8s, bl2_desc, bl2s, bnl2, weights, order, out_half, rowscale_half,
            expert_ids_256, counts, split_cum_256, tile_num_256, tile_cum_256, num_tiles_256,
        )

    compiled_base = call_base()
    compiled_f32 = call_f32()
    compiled_half = call_half()

    metrics = torch.zeros((7,), dtype=torch.float32, device=device)
    summary = torch.empty((8,), dtype=torch.float32, device=device)
    status = torch.zeros((1,), dtype=torch.int32, device=device)
    check_l2_math(
        al2, al2s, a8s, bl2, bl2s, bnl2,
        out_f32, rowscale_f32, out_half, rowscale_half, metrics, status,
    )
    finalize_metrics_kernel[(1,)](
        metrics, summary, CHECK_COUNT=3 * BN, num_warps=1, num_stages=1
    )
    status_code = int(status[0].item())
    norm_rel = float(summary[0].item())
    norm_sqnr = float(summary[1].item())
    half_rel = float(summary[2].item())
    half_sqnr = float(summary[3].item())
    pair_rel = float(summary[4].item())
    pair_sqnr = float(summary[5].item())
    norm_exact_fraction = float(summary[6].item())
    half_exact_fraction = float(summary[7].item())
    rowscale_f32_ok = (status_code & 1) == 0
    rowscale_half_ok = (status_code & 2) == 0
    reference_finite = (status_code & 4) == 0
    norm_f32_finite = (status_code & 8) == 0
    half_finite = (status_code & 16) == 0
    math_passed = status_code == 0 and norm_sqnr >= SQNR_THRESHOLD_DB and half_sqnr >= SQNR_THRESHOLD_DB
    if not math_passed:
        print(
            'P1MD {"math_check_passed":false,"status":'
            + str(status_code)
            + ',"norm_f32_rel":' + str(norm_rel)
            + ',"norm_f32_sqnr_db":' + str(norm_sqnr)
            + ',"half_rel":' + str(half_rel)
            + ',"half_sqnr_db":' + str(half_sqnr)
            + ',"half_vs_norm_rel":' + str(pair_rel)
            + ',"half_vs_norm_sqnr_db":' + str(pair_sqnr)
            + ',"norm_f32_exact_fraction":' + str(norm_exact_fraction)
            + ',"half_exact_fraction":' + str(half_exact_fraction)
            + ',"rowscale_f32_ok":' + ("true" if rowscale_f32_ok else "false")
            + ',"rowscale_half_ok":' + ("true" if rowscale_half_ok else "false")
            + ',"reference_finite":' + ("true" if reference_finite else "false")
            + ',"norm_f32_finite":' + ("true" if norm_f32_finite else "false")
            + ',"half_finite":' + ("true" if half_finite else "false")
            + ',"threshold_db":35.0}'
        )
        raise RuntimeError("P1MD L2/FP16 math check failed")

    groups_text = []
    for group in range(3):
        if group == 0:
            base_ms = float(triton.testing.do_bench(call_base, warmup=5, rep=10))
            f32_ms = float(triton.testing.do_bench(call_f32, warmup=5, rep=10))
            half_ms = float(triton.testing.do_bench(call_half, warmup=5, rep=10))
            order_name = "ABC"
        elif group == 1:
            f32_ms = float(triton.testing.do_bench(call_f32, warmup=5, rep=10))
            half_ms = float(triton.testing.do_bench(call_half, warmup=5, rep=10))
            base_ms = float(triton.testing.do_bench(call_base, warmup=5, rep=10))
            order_name = "BCA"
        else:
            half_ms = float(triton.testing.do_bench(call_half, warmup=5, rep=10))
            base_ms = float(triton.testing.do_bench(call_base, warmup=5, rep=10))
            f32_ms = float(triton.testing.do_bench(call_f32, warmup=5, rep=10))
            order_name = "CAB"
        groups_text.append(
            '{"group":' + str(group) + ',"order":"' + order_name
            + '","baseline_ms":' + str(base_ms)
            + ',"l2_f32_ms":' + str(f32_ms)
            + ',"l2_half_ms":' + str(half_ms)
            + ',"l2_f32_over_baseline":' + str(f32_ms / base_ms)
            + ',"l2_half_over_baseline":' + str(half_ms / base_ms) + '}'
        )

    report = (
        '{"kind":"fp16acc_l2_c10_md_custom","shape":{"T":4096,"H":4096,'
        '"E":256,"I":1536,"topk":8,"M":32768},'
        '"geometry":{"grid":132,"BN":128,"BK":128,"GM":32,"warps":8,'
        '"baseline":{"BM":128,"stages":4},"l2_f32":{"BM":128,"stages":4},'
        '"l2_half":{"BM":256,"stages":3}},'
        '"route":{"kind":"analytic_112_144_alternating","BM128_tiles":384,'
        '"BM256_tiles":256},'
        '"quantization":{"baseline":"row_amax_over_448_FP8",'
        '"l2":"BF16_source_row_L2_over_128_FP8",'
        '"act_bound_A_scale":"original_row_amax_over_448",'
        '"l2_GU_BNORM":"max_true_L2_of_dequantized_l2_FP8_rows_per_expert"},'
        '"math_check":{"kind":"3_rows_x_128cols_independent_F32_K4096_reduction_then_common_ACT_FP8_quant",'
        '"comparisons":384,"passed":true,"status":' + str(status_code)
        + ',"threshold_db":35.0,"norm_f32_rel":' + str(norm_rel)
        + ',"norm_f32_sqnr_db":' + str(norm_sqnr)
        + ',"half_rel":' + str(half_rel)
        + ',"half_sqnr_db":' + str(half_sqnr)
        + ',"half_vs_norm_rel":' + str(pair_rel)
        + ',"half_vs_norm_sqnr_db":' + str(pair_sqnr)
        + ',"norm_f32_exact_fraction":' + str(norm_exact_fraction)
        + ',"half_exact_fraction":' + str(half_exact_fraction)
        + ',"rowscale_exact":{"norm_f32":true,"half":true},'
        + '"finite":{"reference":true,"norm_f32":true,"half":true}},'
        '"resources":{"baseline":{"n_regs":' + str(compiled_base.n_regs)
        + ',"n_spills":' + str(compiled_base.n_spills)
        + ',"shared":' + str(compiled_base.metadata.shared)
        + '},"l2_f32":{"n_regs":' + str(compiled_f32.n_regs)
        + ',"n_spills":' + str(compiled_f32.n_spills)
        + ',"shared":' + str(compiled_f32.metadata.shared)
        + '},"l2_half":{"n_regs":' + str(compiled_half.n_regs)
        + ',"n_spills":' + str(compiled_half.n_spills)
        + ',"shared":' + str(compiled_half.metadata.shared) + '}},'
        '"timing":{"method":"do_bench_warmup5_rep10","scope":"MD_kernel_only",'
        '"setup_excluded":true,"groups":[' + ",".join(groups_text) + ']},'
        '"limitations":"custom_single_GPU_c10_only_no_DN_no_P1_integration"}'
    )
    print("P1MD " + report)
    raise RuntimeError("P1MD done")


def run_kernel(x):
    return x


run_diagnostic()
