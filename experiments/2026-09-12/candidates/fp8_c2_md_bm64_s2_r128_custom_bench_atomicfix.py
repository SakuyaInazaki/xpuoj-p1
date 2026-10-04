"""GeneratedWorkload-only H800 c2 MD occupancy-configuration diagnostic.

This is not a P1 submission candidate.  It builds one deterministic private-
token c2 workload and times the active call-3+ MD baseline against one BM64,
two-stage, maxnreg=128 candidate.  Allocation, route/metadata, quantization,
BNORM, descriptor construction, validation, DN, and final gather are outside
timing.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


T = 16384
H = 4096
E = 8
I = 14336
KTOP = 2
M = 32768
GRID = 132
BM = 128
BM_CANDIDATE = 64
MD_BN = 128
BK = 128
CANDIDATE_SQNR_THRESHOLD_DB = 60.0
CANDIDATE_SCALE_REL_THRESHOLD = 1.0e-5


@triton.jit
def setup_route_metadata_kernel(
    ORDER,
    INV_ORDER,
    W_SORTED,
    COUNTS,
    EXPERT_IDS,
    SPLIT_CUM,
    TILE_NUM,
    TILE_CUM,
    NUM_TILES,
    EXPERT_IDS_CANDIDATE,
    SPLIT_CUM_CANDIDATE,
    TILE_NUM_CANDIDATE,
    TILE_CUM_CANDIDATE,
    NUM_TILES_CANDIDATE,
    M_CONST: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Alternating 4016/4176 expert counts sum to M and leave a real partial
    # BM128 tile for every expert.  The odd affine branch permutation is a
    # deterministic bijection, so each private token appears exactly twice.
    expert = tl.program_id(0)
    parity = expert % 2
    n_rows = 4016 + 160 * parity
    row_begin = (expert // 2) * 8192 + parity * 4016
    offs_r = tl.arange(0, BLOCK_R)
    mask = offs_r < n_rows
    row = row_begin + offs_r
    src = (row * 8191) & (M_CONST - 1)
    tl.store(ORDER + row, src, mask=mask)
    tl.store(INV_ORDER + src, row, mask=mask)
    tl.store(W_SORTED + row, 0.5, mask=mask)
    tl.store(COUNTS + expert, n_rows)

    n_tiles = 32 + parity
    tile_begin = (expert // 2) * 65 + parity * 32
    tile_cum = tile_begin + n_tiles
    toffs = tl.arange(0, 64)
    tmask = toffs < n_tiles
    tid = tile_begin + toffs
    tl.store(EXPERT_IDS + tid, expert, mask=tmask)
    tl.store(SPLIT_CUM + tid, row_begin, mask=tmask)
    tl.store(TILE_NUM + tid, n_tiles, mask=tmask)
    tl.store(TILE_CUM + tid, tile_cum, mask=tmask)
    if expert == 0:
        tl.store(NUM_TILES, 260)

    # BM64 needs 63 tiles for 4016 rows and 66 for 4176 rows: 516 total.
    n_tiles_candidate = 63 + 3 * parity
    tile_begin_candidate = (expert // 2) * 129 + parity * 63
    tile_cum_candidate = tile_begin_candidate + n_tiles_candidate
    toffs_candidate = tl.arange(0, 128)
    tmask_candidate = toffs_candidate < n_tiles_candidate
    tid_candidate = tile_begin_candidate + toffs_candidate
    tl.store(EXPERT_IDS_CANDIDATE + tid_candidate, expert, mask=tmask_candidate)
    tl.store(SPLIT_CUM_CANDIDATE + tid_candidate, row_begin, mask=tmask_candidate)
    tl.store(TILE_NUM_CANDIDATE + tid_candidate, n_tiles_candidate, mask=tmask_candidate)
    tl.store(TILE_CUM_CANDIDATE + tid_candidate, tile_cum_candidate, mask=tmask_candidate)
    if expert == 0:
        tl.store(NUM_TILES_CANDIDATE, 516)


@triton.jit
def gq1p_tm_kernel(
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
    # Exact active _gq1p_tm math: one private token load and quantization, then
    # identical stores to its two sorted branch destinations.
    token = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H_RUNTIME
    x = tl.load(
        X + token * stride_xm + offs_h * stride_xh,
        mask=mask,
        other=0.0,
    )
    amax = tl.max(tl.abs(x)).to(tl.float32)
    scale = tl.maximum(amax / 448.0, 1.0e-12)
    q = (x.to(tl.float32) * (1.0 / scale)).to(tl.float8e4nv)
    for j in tl.static_range(K_BRANCH):
        dest = tl.load(INV_ORDER + token * K_BRANCH + j)
        tl.store(Q + dest * stride_qm + offs_h * stride_qh, q, mask=mask)
        tl.store(SCALE + dest, scale)


@triton.jit
def quantize_weight_kernel(
    X,
    Q,
    SCALE,
    BNORM,
    K_CONST: tl.constexpr,
    ROWS_PER_EXPERT: tl.constexpr,
    WRITE_BNORM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The active rowwise FP8 quantizer for GU K=4096.  Large linear offsets
    # are promoted before K multiplication.
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    base = row64 * K_CONST
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K_CONST
    x = tl.load(X + base + offs_k, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.0, 1.0e-12)
    q = (x / scale).to(tl.float8e4nv)
    tl.store(Q + base + offs_k, q, mask=mask)
    tl.store(SCALE + row, scale)
    if WRITE_BNORM:
        real = q.to(tl.float32) * scale
        norm = tl.sqrt(tl.sum(real * real, axis=0))
        tl.atomic_max(BNORM + row // ROWS_PER_EXPERT, norm)


@triton.jit
def active_c2_md_kernel(
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
    M_CONST: tl.constexpr,
    I_CONST: tl.constexpr,
    K_CONST: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Source-equivalent _fgs_tma2_int_pm_q8_kernel.  B is the physical
    # [g128,u128,...] interleave created by active _get_int_gu(gran=128).
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I_CONST, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)
    for tile_id in range(pid, total_tiles * num_block_n, num_pid):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n
        expert = tl.load(EXPERT_IDS + pid_m)
        n_rows = tl.load(SPLIT_SIZE + expert)
        row_begin = tl.load(SPLIT_CUM + pid_m)
        t_num = tl.load(TILE_NUM + pid_m)
        t_cum = tl.load(TILE_CUM + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(
            local_m, pid_n, t_num, num_block_n, GROUP_M
        )
        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < row_begin + n_rows
        a_row = row_begin + local_m * BLOCK_M
        b_row_g = expert * (2 * I_CONST) + pid_n * (2 * BLOCK_N)
        b_row_u = b_row_g + BLOCK_N
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K_CONST, BLOCK_K)):
            a = A_DESC.load([a_row, k * BLOCK_K])
            bg = B_DESC.load([b_row_g, k * BLOCK_K])
            bu = B_DESC.load([b_row_u, k * BLOCK_K])
            acc_g = tl.dot(a, bg.T, acc_g)
            acc_u = tl.dot(a, bu.T, acc_u)

        a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        g_scale = a_scale[:, None] * tl.load(
            B_SCALE
            + expert * (2 * I_CONST)
            + pid_n * 2 * BLOCK_N
            + tl.arange(0, BLOCK_N)
        )[None, :]
        u_scale = a_scale[:, None] * tl.load(
            B_SCALE
            + expert * (2 * I_CONST)
            + pid_n * 2 * BLOCK_N
            + BLOCK_N
            + tl.arange(0, BLOCK_N)
        )[None, :]
        g = acc_g * g_scale
        u = acc_u * u_scale
        w = tl.load(W + offs_m, mask=row_mask, other=0.0).to(tl.float32)
        th = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [g * 0.5],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        act = (g * 0.5) * (1.0 + th) * u * w[:, None]
        bn = tl.load(BNORM + expert)
        bound = a_scale * a_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1.0e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(ptrs, q, mask=row_mask[:, None], eviction_policy="evict_first")
        tl.store(ROWSCALE + offs_m, s, mask=row_mask)


@triton.jit
def md_oracle_check_kernel(
    A,
    A_SCALE,
    BNORM,
    GU_WEIGHT,
    GU_SCALE,
    W_SORTED,
    BASE_OUT,
    BASE_ROWSCALE,
    CANDIDATE_OUT,
    CANDIDATE_ROWSCALE,
    METRICS,
    STATUS,
    M_CONST: tl.constexpr,
    I_CONST: tl.constexpr,
    H_CONST: tl.constexpr,
    BLOCK_K_MD: tl.constexpr,
):
    # Independent F32 scalar-reduction reference for three rows x 128 columns.
    pid = tl.program_id(0)
    check = pid // 128
    lane = pid % 128
    row = tl.where(check == 0, 4015, tl.where(check == 1, 4016, M_CONST - 1))
    expert = tl.where(check == 0, 0, tl.where(check == 1, 1, 7))
    md_start = tl.where(check == 0, 0, tl.where(check == 1, I_CONST // 2, I_CONST - 128))
    md_n = md_start + lane
    block = md_n // 128
    in_block = md_n % 128
    g_row = expert * (2 * I_CONST) + block * 256 + in_block
    u_row = g_row + 128
    offs_k = tl.arange(0, BLOCK_K_MD)
    row64 = row.to(tl.int64)
    g_row64 = g_row.to(tl.int64)
    u_row64 = u_row.to(tl.int64)
    a = tl.load(A + row64 * H_CONST + offs_k).to(tl.float32)
    bg = tl.load(GU_WEIGHT + g_row64 * H_CONST + offs_k).to(tl.float32)
    bu = tl.load(GU_WEIGHT + u_row64 * H_CONST + offs_k).to(tl.float32)
    acc_g = tl.sum(a * bg, axis=0)
    acc_u = tl.sum(a * bu, axis=0)
    a_scale = tl.load(A_SCALE + row)
    g = acc_g * a_scale * tl.load(GU_SCALE + g_row)
    u = acc_u * a_scale * tl.load(GU_SCALE + u_row)
    w = tl.load(W_SORTED + row).to(tl.float32)
    th = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [g * 0.5],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    raw = (g * 0.5) * (1.0 + th) * u * w
    bn = tl.load(BNORM + expert)
    bound = a_scale * a_scale * bn * bn * tl.abs(w)
    bbits = tl.maximum(bound, 1.0e-30).to(tl.int32, bitcast=True)
    bexp = (bbits >> 23) & 0xFF
    scale_ref = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
    inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
    q_ref = (raw * inv).to(tl.float8e4nv)
    ref = q_ref.to(tl.float32) * scale_ref
    offset = row64 * I_CONST + md_n
    q_base = tl.load(BASE_OUT + offset)
    q_candidate = tl.load(CANDIDATE_OUT + offset)
    scale_base = tl.load(BASE_ROWSCALE + row)
    scale_candidate = tl.load(CANDIDATE_ROWSCALE + row)
    base = q_base.to(tl.float32) * scale_base
    candidate = q_candidate.to(tl.float32) * scale_candidate

    tl.atomic_add(METRICS + 0, ref * ref)
    tl.atomic_add(METRICS + 1, (base - ref) * (base - ref))
    tl.atomic_add(METRICS + 2, (candidate - ref) * (candidate - ref))
    ref_bits = q_ref.to(tl.uint8, bitcast=True)
    base_bits = q_base.to(tl.uint8, bitcast=True)
    candidate_bits = q_candidate.to(tl.uint8, bitcast=True)
    tl.atomic_add(METRICS + 3, (base_bits != ref_bits).to(tl.float32))
    tl.atomic_add(METRICS + 4, (candidate_bits != ref_bits).to(tl.float32))
    tl.atomic_add(METRICS + 5, (scale_base != scale_ref).to(tl.float32))
    tl.atomic_add(METRICS + 6, (scale_candidate != scale_ref).to(tl.float32))

    ref_finite = (ref == ref) & (tl.abs(ref) <= 3.402823466e38)
    base_finite = (base == base) & (tl.abs(base) <= 3.402823466e38)
    candidate_finite = (candidate == candidate) & (tl.abs(candidate) <= 3.402823466e38)
    scale_finite = (
        (scale_base == scale_base)
        & (scale_candidate == scale_candidate)
        & (scale_base > 0.0)
        & (scale_candidate > 0.0)
    )
    tl.atomic_or(STATUS, 1, mask=ref_finite == 0)
    tl.atomic_or(STATUS, 2, mask=base_finite == 0)
    tl.atomic_or(STATUS, 4, mask=candidate_finite == 0)
    tl.atomic_or(STATUS, 8, mask=scale_finite == 0)


@triton.jit
def finalize_oracle_metrics_kernel(METRICS, SUMMARY, CHECK_COUNT: tl.constexpr):
    signal = tl.maximum(tl.load(METRICS + 0), 1.0e-30)
    base_err = tl.maximum(tl.load(METRICS + 1), signal * 1.0e-12)
    candidate_err = tl.maximum(tl.load(METRICS + 2), signal * 1.0e-12)
    tl.store(SUMMARY + 0, tl.sqrt(base_err / signal))
    tl.store(SUMMARY + 1, 3.010299956639812 * tl.log2(signal / base_err))
    tl.store(SUMMARY + 2, tl.sqrt(candidate_err / signal))
    tl.store(SUMMARY + 3, 3.010299956639812 * tl.log2(signal / candidate_err))
    tl.store(SUMMARY + 4, 1.0 - tl.load(METRICS + 3) / CHECK_COUNT)
    tl.store(SUMMARY + 5, 1.0 - tl.load(METRICS + 4) / CHECK_COUNT)
    tl.store(SUMMARY + 6, 1.0 - tl.load(METRICS + 5) / CHECK_COUNT)
    tl.store(SUMMARY + 7, 1.0 - tl.load(METRICS + 6) / CHECK_COUNT)


@triton.jit
def full_compare_partial_kernel(
    BASE_OUT,
    BASE_ROWSCALE,
    CANDIDATE_OUT,
    CANDIDATE_ROWSCALE,
    PARTIAL,
    STATUS,
    TOTAL_CONST: tl.constexpr,
    I_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_CONST
    row = offs // I_CONST
    base_scale = tl.load(BASE_ROWSCALE + row, mask=mask, other=1.0)
    candidate_scale = tl.load(CANDIDATE_ROWSCALE + row, mask=mask, other=1.0)
    q_base = tl.load(BASE_OUT + offs, mask=mask, other=0.0)
    q_candidate = tl.load(CANDIDATE_OUT + offs, mask=mask, other=0.0)
    base = q_base.to(tl.float32) * base_scale
    candidate = q_candidate.to(tl.float32) * candidate_scale
    diff = candidate - base
    rel = tl.abs(diff) / tl.maximum(tl.abs(base), 1.0e-6)
    base_bits = q_base.to(tl.uint8, bitcast=True)
    candidate_bits = q_candidate.to(tl.uint8, bitcast=True)
    first_col = (offs % I_CONST) == 0
    scale_mask = mask & first_col
    scale_diff = candidate_scale - base_scale
    scale_rel = tl.abs(scale_diff) / tl.maximum(tl.abs(base_scale), 1.0e-12)
    out = PARTIAL + pid * 7
    tl.store(out + 0, tl.sum(tl.where(mask, base * base, 0.0), axis=0))
    tl.store(out + 1, tl.sum(tl.where(mask, diff * diff, 0.0), axis=0))
    tl.store(out + 2, tl.max(tl.where(mask, rel, 0.0), axis=0))
    tl.store(
        out + 3,
        tl.sum(tl.where(mask, (base_bits != candidate_bits).to(tl.float32), 0.0), axis=0),
    )
    tl.store(out + 4, tl.sum(tl.where(scale_mask, base_scale * base_scale, 0.0), axis=0))
    tl.store(out + 5, tl.sum(tl.where(scale_mask, scale_diff * scale_diff, 0.0), axis=0))
    tl.store(out + 6, tl.max(tl.where(scale_mask, scale_rel, 0.0), axis=0))
    value_finite = (
        (base == base)
        & (candidate == candidate)
        & (tl.abs(base) <= 3.402823466e38)
        & (tl.abs(candidate) <= 3.402823466e38)
    )
    scale_finite = (
        (base_scale == base_scale)
        & (candidate_scale == candidate_scale)
        & (base_scale > 0.0)
        & (candidate_scale > 0.0)
    )
    value_invalid = tl.sum(tl.where(mask & (value_finite == 0), 1, 0), axis=0) > 0
    scale_invalid = tl.sum(tl.where(scale_mask & (scale_finite == 0), 1, 0), axis=0) > 0
    tl.atomic_or(STATUS, 1, mask=value_invalid)
    tl.atomic_or(STATUS, 2, mask=scale_invalid)


@triton.jit
def reduce_compare_partials_kernel(
    PARTIAL_IN,
    PARTIAL_OUT,
    N_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_CONST
    out = PARTIAL_OUT + pid * 7
    for metric in tl.static_range(7):
        x = tl.load(PARTIAL_IN + offs * 7 + metric, mask=mask, other=0.0)
        if metric == 2 or metric == 6:
            tl.store(out + metric, tl.max(x, axis=0))
        else:
            tl.store(out + metric, tl.sum(x, axis=0))


@triton.jit
def finalize_compare_metrics_kernel(
    PARTIAL,
    SUMMARY,
    N_CONST: tl.constexpr,
    TOTAL_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N_CONST
    signal = tl.sum(tl.load(PARTIAL + offs * 7 + 0, mask=mask, other=0.0), axis=0)
    err = tl.sum(tl.load(PARTIAL + offs * 7 + 1, mask=mask, other=0.0), axis=0)
    max_rel = tl.max(tl.load(PARTIAL + offs * 7 + 2, mask=mask, other=0.0), axis=0)
    mismatches = tl.sum(tl.load(PARTIAL + offs * 7 + 3, mask=mask, other=0.0), axis=0)
    scale_signal = tl.sum(tl.load(PARTIAL + offs * 7 + 4, mask=mask, other=0.0), axis=0)
    scale_err = tl.sum(tl.load(PARTIAL + offs * 7 + 5, mask=mask, other=0.0), axis=0)
    max_scale_rel = tl.max(tl.load(PARTIAL + offs * 7 + 6, mask=mask, other=0.0), axis=0)
    signal = tl.maximum(signal, 1.0e-30)
    err = tl.maximum(err, signal * 1.0e-12)
    scale_signal = tl.maximum(scale_signal, 1.0e-30)
    scale_err = tl.maximum(scale_err, scale_signal * 1.0e-12)
    tl.store(SUMMARY + 0, tl.sqrt(err / signal))
    tl.store(SUMMARY + 1, 3.010299956639812 * tl.log2(signal / err))
    tl.store(SUMMARY + 2, max_rel)
    tl.store(SUMMARY + 3, 1.0 - mismatches / TOTAL_CONST)
    tl.store(SUMMARY + 4, tl.sqrt(scale_err / scale_signal))
    tl.store(SUMMARY + 5, max_scale_rel)

def launch_md(
    a_desc,
    a_scale,
    bnorm,
    gu_desc,
    gu_scale,
    weights,
    order,
    act,
    rowscale,
    expert_ids,
    counts,
    split_cum,
    tile_num,
    tile_cum,
    num_tiles,
    block_m,
    stages,
    register_cap,
):
    return active_c2_md_kernel[(GRID,)](
        a_desc,
        a_scale,
        bnorm,
        gu_desc,
        gu_scale,
        weights,
        order,
        act,
        rowscale,
        expert_ids,
        counts,
        split_cum,
        tile_num,
        tile_cum,
        num_tiles,
        M_CONST=M,
        I_CONST=I,
        K_CONST=H,
        stride_cm=act.stride(0),
        stride_cn=act.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=MD_BN,
        BLOCK_K=BK,
        GROUP_M=8,
        num_warps=8,
        num_stages=stages,
        maxnreg=register_cap,
        num_ctas=1,
    )


def run_diagnostic():
    device = "cuda"
    order = torch.empty((M,), dtype=torch.int64, device=device)
    inv_order = torch.empty((M,), dtype=torch.int64, device=device)
    weights = torch.empty((M,), dtype=torch.float32, device=device)
    counts = torch.empty((E,), dtype=torch.int32, device=device)
    metadata_size = triton.cdiv(M, BM) + E
    expert_ids = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    split_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_num = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    num_tiles = torch.empty((1,), dtype=torch.int32, device=device)
    metadata_size_candidate = triton.cdiv(M, BM_CANDIDATE) + E
    expert_ids_candidate = torch.empty((metadata_size_candidate,), dtype=torch.int32, device=device)
    split_cum_candidate = torch.empty((metadata_size_candidate,), dtype=torch.int32, device=device)
    tile_num_candidate = torch.empty((metadata_size_candidate,), dtype=torch.int32, device=device)
    tile_cum_candidate = torch.empty((metadata_size_candidate,), dtype=torch.int32, device=device)
    num_tiles_candidate = torch.empty((1,), dtype=torch.int32, device=device)
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
        expert_ids_candidate,
        split_cum_candidate,
        tile_num_candidate,
        tile_cum_candidate,
        num_tiles_candidate,
        M_CONST=M,
        BLOCK_R=8192,
        num_warps=8,
        num_stages=1,
    )

    x = torch.randn((T, H), dtype=torch.bfloat16, device=device)
    a_q = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=device)
    a_scale = torch.empty((M,), dtype=torch.float32, device=device)
    gq1p_tm_kernel[(T,)](
        x,
        inv_order,
        a_q,
        a_scale,
        T,
        H,
        x.stride(0),
        x.stride(1),
        a_q.stride(0),
        a_q.stride(1),
        K_BRANCH=KTOP,
        BLOCK_H=H,
        num_warps=4,
        num_stages=1,
    )
    del x
    a_desc = TensorDescriptor(a_q, a_q.shape, a_q.stride(), [BM, BK])
    a_desc_candidate = TensorDescriptor(
        a_q, a_q.shape, a_q.stride(), [BM_CANDIDATE, BK]
    )

    gu_bf16 = torch.randn((E * 2 * I, H), dtype=torch.bfloat16, device=device)
    gu_q = torch.empty((E * 2 * I, H), dtype=torch.float8_e4m3fn, device=device)
    gu_scale = torch.empty((E * 2 * I,), dtype=torch.float32, device=device)
    bnorm = torch.zeros((E,), dtype=torch.float32, device=device)
    quantize_weight_kernel[(E * 2 * I,)](
        gu_bf16,
        gu_q,
        gu_scale,
        bnorm,
        K_CONST=H,
        ROWS_PER_EXPERT=2 * I,
        WRITE_BNORM=True,
        BLOCK_K=H,
        num_warps=8,
        num_stages=1,
    )
    del gu_bf16
    gu_desc = TensorDescriptor(gu_q, gu_q.shape, gu_q.stride(), [MD_BN, BK])

    base_out = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    base_rowscale = torch.empty((M,), dtype=torch.float32, device=device)
    candidate_out = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    candidate_rowscale = torch.empty((M,), dtype=torch.float32, device=device)

    def call_base():
        return launch_md(
            a_desc,
            a_scale,
            bnorm,
            gu_desc,
            gu_scale,
            weights,
            order,
            base_out,
            base_rowscale,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
            BM,
            4,
            168,
        )

    def call_candidate():
        return launch_md(
            a_desc_candidate,
            a_scale,
            bnorm,
            gu_desc,
            gu_scale,
            weights,
            order,
            candidate_out,
            candidate_rowscale,
            expert_ids_candidate,
            counts,
            split_cum_candidate,
            tile_num_candidate,
            tile_cum_candidate,
            num_tiles_candidate,
            BM_CANDIDATE,
            2,
            128,
        )

    compiled_base = call_base()
    compiled_candidate = call_candidate()

    oracle_metrics = torch.zeros((7,), dtype=torch.float32, device=device)
    oracle_summary = torch.empty((8,), dtype=torch.float32, device=device)
    oracle_status = torch.zeros((1,), dtype=torch.int32, device=device)
    md_oracle_check_kernel[(3 * MD_BN,)](
        a_q,
        a_scale,
        bnorm,
        gu_q,
        gu_scale,
        weights,
        base_out,
        base_rowscale,
        candidate_out,
        candidate_rowscale,
        oracle_metrics,
        oracle_status,
        M_CONST=M,
        I_CONST=I,
        H_CONST=H,
        BLOCK_K_MD=H,
        num_warps=8,
        num_stages=1,
    )
    finalize_oracle_metrics_kernel[(1,)](
        oracle_metrics,
        oracle_summary,
        CHECK_COUNT=3 * MD_BN,
        num_warps=1,
        num_stages=1,
    )

    total_values = M * I
    compare_block = 4096
    partial_count = triton.cdiv(total_values, compare_block)
    reduced_count = triton.cdiv(partial_count, 1024)
    partial = torch.empty((partial_count, 7), dtype=torch.float32, device=device)
    reduced = torch.empty((reduced_count, 7), dtype=torch.float32, device=device)
    compare_summary = torch.empty((6,), dtype=torch.float32, device=device)
    compare_status = torch.zeros((1,), dtype=torch.int32, device=device)
    full_compare_partial_kernel[(partial_count,)](
        base_out,
        base_rowscale,
        candidate_out,
        candidate_rowscale,
        partial,
        compare_status,
        TOTAL_CONST=total_values,
        I_CONST=I,
        BLOCK=compare_block,
        num_warps=8,
        num_stages=1,
    )
    reduce_compare_partials_kernel[(reduced_count,)](
        partial,
        reduced,
        N_CONST=partial_count,
        BLOCK=1024,
        num_warps=8,
        num_stages=1,
    )
    finalize_compare_metrics_kernel[(1,)](
        reduced,
        compare_summary,
        N_CONST=reduced_count,
        TOTAL_CONST=total_values,
        BLOCK=128,
        num_warps=1,
        num_stages=1,
    )

    oracle_status_code = int(oracle_status[0].item())
    compare_status_code = int(compare_status[0].item())
    base_oracle_rel = float(oracle_summary[0].item())
    base_oracle_sqnr = float(oracle_summary[1].item())
    candidate_oracle_rel = float(oracle_summary[2].item())
    candidate_oracle_sqnr = float(oracle_summary[3].item())
    base_oracle_exact = float(oracle_summary[4].item())
    candidate_oracle_exact = float(oracle_summary[5].item())
    base_oracle_scale_exact = float(oracle_summary[6].item())
    candidate_oracle_scale_exact = float(oracle_summary[7].item())
    candidate_vs_base_rel = float(compare_summary[0].item())
    candidate_vs_base_sqnr = float(compare_summary[1].item())
    candidate_vs_base_max_rel = float(compare_summary[2].item())
    candidate_vs_base_exact = float(compare_summary[3].item())
    candidate_scale_rel = float(compare_summary[4].item())
    candidate_scale_max_rel = float(compare_summary[5].item())
    comparison_passed = (
        oracle_status_code == 0
        and compare_status_code == 0
        and candidate_vs_base_sqnr >= CANDIDATE_SQNR_THRESHOLD_DB
        and candidate_scale_max_rel <= CANDIDATE_SCALE_REL_THRESHOLD
    )
    if not comparison_passed:
        print(
            'P1MD {"candidate_vs_baseline_passed":false,"oracle_status":'
            + str(oracle_status_code)
            + ',"compare_status":' + str(compare_status_code)
            + ',"candidate_vs_base_sqnr_db":' + str(candidate_vs_base_sqnr)
            + ',"candidate_vs_base_rel":' + str(candidate_vs_base_rel)
            + ',"candidate_vs_base_max_rel":' + str(candidate_vs_base_max_rel)
            + ',"candidate_scale_rel":' + str(candidate_scale_rel)
            + ',"candidate_scale_max_rel":' + str(candidate_scale_max_rel)
            + ',"sqnr_threshold_db":60.0,"scale_max_rel_threshold":0.00001}'
        )
        raise RuntimeError("P1MD c2 MD occupancy candidate comparison failed")

    groups_text = []
    for group in range(3):
        if group == 1:
            candidate_ms = float(triton.testing.do_bench(call_candidate, warmup=5, rep=10))
            base_ms = float(triton.testing.do_bench(call_base, warmup=5, rep=10))
            order_name = "CANDIDATE_BASE"
        else:
            base_ms = float(triton.testing.do_bench(call_base, warmup=5, rep=10))
            candidate_ms = float(triton.testing.do_bench(call_candidate, warmup=5, rep=10))
            order_name = "BASE_CANDIDATE"
        groups_text.append(
            '{"group":' + str(group)
            + ',"order":"' + order_name
            + '","base_ms":' + str(base_ms)
            + ',"candidate_ms":' + str(candidate_ms)
            + ',"candidate_over_base":' + str(candidate_ms / base_ms)
            + '}'
        )

    report = (
        '{"kind":"fp8_c2_md_bm64_s2_r128_custom",'
        '"purpose":"single_GPU_current_c2_MD_occupancy_configuration_comparison_not_P1_candidate",'
        '"shape":{"T":16384,"H":4096,"E":8,"I":14336,"topk":2,"M":32768},'
        '"route":{"counts":[4016,4176,4016,4176,4016,4176,4016,4176],'
        '"baseline_metadata_tiles":260,"candidate_metadata_tiles":516,'
        '"all_experts_have_partial_tail_in_both_paths":true},'
        '"geometry":{"grid_programs":132,"BN":128,"BK":128,"GM":8,"warps":8,'
        '"baseline":{"BM":128,"stages":4,"maxnreg":168,"num_ctas":1},'
        '"candidate":{"BM":64,"stages":2,"maxnreg":128,"num_ctas":1},'
        '"candidate_changes":["BM","stages","maxnreg"]},'
        '"candidate_vs_baseline":{"passed":true,"scope":"full_M_times_I_GPU_reduction",'
        '"values":469762048,"finite":true,"sqnr_threshold_db":60.0,'
        '"scale_max_rel_threshold":0.00001,"relative_error_floor":0.000001,'
        '"rel":' + str(candidate_vs_base_rel)
        + ',"sqnr_db":' + str(candidate_vs_base_sqnr)
        + ',"max_rel":' + str(candidate_vs_base_max_rel)
        + ',"fp8_exact_fraction":' + str(candidate_vs_base_exact)
        + ',"scale_rel":' + str(candidate_scale_rel)
        + ',"scale_max_rel":' + str(candidate_scale_max_rel) + '},'
        '"independent_F32_reference":{"gate":false,"status":' + str(oracle_status_code)
        + ',"rows":[4015,4016,32767],"comparisons_per_path":384,'
        '"base_rel":' + str(base_oracle_rel)
        + ',"base_sqnr_db":' + str(base_oracle_sqnr)
        + ',"candidate_rel":' + str(candidate_oracle_rel)
        + ',"candidate_sqnr_db":' + str(candidate_oracle_sqnr)
        + ',"base_fp8_exact_fraction":' + str(base_oracle_exact)
        + ',"candidate_fp8_exact_fraction":' + str(candidate_oracle_exact)
        + ',"base_rowscale_exact_fraction":' + str(base_oracle_scale_exact)
        + ',"candidate_rowscale_exact_fraction":' + str(candidate_oracle_scale_exact) + '},'
        '"resources":{"base":{"num_ctas":1,"n_regs":' + str(compiled_base.n_regs)
        + ',"n_spills":' + str(compiled_base.n_spills)
        + ',"shared":' + str(compiled_base.metadata.shared)
        + '},"candidate":{"num_ctas":1,"n_regs":' + str(compiled_candidate.n_regs)
        + ',"n_spills":' + str(compiled_candidate.n_spills)
        + ',"shared":' + str(compiled_candidate.metadata.shared) + '}},'
        '"timing":{"method":"do_bench_warmup5_rep10","scope":"MD_kernel_only",'
        '"groups":[' + ",".join(groups_text) + ']},'
        '"excluded":{"allocation":true,"route_and_metadata":true,'
        '"A_and_GU_quantization":true,"BNORM_and_descriptors":true,'
        '"validation":true,"DN_and_final_gather":true},'
        '"limitations":"single_GPU_analytic_route_single_combined_configuration_no_parameter_sweep_or_P1_integration;resource_reduction_does_not_imply_speedup"}'
    )
    print("P1MD " + report)
    raise RuntimeError("P1MD done")

def run_kernel(x):
    return x


run_diagnostic()
