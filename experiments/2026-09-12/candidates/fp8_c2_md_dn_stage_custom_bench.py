"""GeneratedWorkload-only H800 stage diagnostic for active c2 MD and DN.

This is not a P1 submission candidate.  It builds one deterministic private-
token c2 workload and times the active call-3+ MD, DN, and consecutive MD+DN
launches.  Allocation, route/metadata, weight and activation quantization,
BNORM, descriptor construction, and final branch gather are outside timing.
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
MD_BN = 128
DN_BN = 256
BK = 128
SQNR_THRESHOLD_DB = 35.0
DN_SCALE_REL_THRESHOLD = 1.0e-3


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
    # The same rowwise FP8 quantizer is specialized once for GU K=4096 and
    # once for down K=14336.  Large linear offsets are promoted before K mul.
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
def active_c2_dn_kernel(
    A_DESC,
    A_SCALE,
    B_DESC,
    B_SCALE,
    C,
    CSCL,
    EXPERT_IDS,
    SPLIT_SIZE,
    SPLIT_CUM,
    TILE_NUM,
    TILE_CUM,
    NUM_TILES,
    M_CONST: tl.constexpr,
    N_CONST: tl.constexpr,
    K_CONST: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FLAT: tl.constexpr,
):
    # Source-equivalent _dn_tma2_f8_kernel for c2: FLAT=False and s3.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(N_CONST, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)
    for tile_id in tl.range(
        pid, total_tiles * num_block_n, num_pid, flatten=FLAT
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
            local_m, pid_n, t_num, num_block_n, GROUP_M
        )
        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < row_begin + n_rows
        a_row = row_begin + local_m * BLOCK_M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_row = expert * N_CONST + pid_n * BLOCK_N
        for k in range(0, tl.cdiv(K_CONST, BLOCK_K)):
            a = A_DESC.load([a_row, k * BLOCK_K])
            b = B_DESC.load([b_row, k * BLOCK_K])
            acc = tl.dot(a, b.T, acc)
        a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        b_scale = tl.load(B_SCALE + expert * N_CONST + offs_n)
        acc = acc * b_scale[None, :]
        row_max = tl.max(tl.abs(acc), axis=1)
        s = tl.maximum(a_scale * row_max * (1.0 / 448.0), 1.0e-12)
        q = (acc * (a_scale / s)[:, None]).to(tl.float8e4nv)
        tl.store(CSCL + offs_m * num_block_n + pid_n, s, mask=row_mask)
        ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(ptrs, q, mask=row_mask[:, None])


@triton.jit
def dn_chunk_oracle_kernel(
    ACT,
    DN_WEIGHT,
    DN_SCALE,
    RAW,
    M_CONST: tl.constexpr,
    H_CONST: tl.constexpr,
    I_CONST: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Three representative rows, one complete 256-column chunk each.  The
    # complete chunk is required because DN quantization uses its true max.
    pid = tl.program_id(0)
    check = pid // 256
    col = pid % 256
    row = tl.where(check == 0, 4015, tl.where(check == 1, 4016, M_CONST - 1))
    expert = tl.where(check == 0, 0, tl.where(check == 1, 1, 7))
    chunk = tl.where(check == 0, 0, tl.where(check == 1, 8, 15))
    n = chunk * 256 + col
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < I_CONST
    row64 = row.to(tl.int64)
    b_row64 = expert.to(tl.int64) * H_CONST + n
    a = tl.load(
        ACT + row64 * I_CONST + offs_k, mask=mask, other=0.0
    ).to(tl.float32)
    b = tl.load(
        DN_WEIGHT + b_row64 * I_CONST + offs_k, mask=mask, other=0.0
    ).to(tl.float32)
    acc = tl.sum(a * b, axis=0)
    raw = acc * tl.load(DN_SCALE + expert * H_CONST + n)
    tl.store(RAW + check * 256 + col, raw)


@triton.jit
def math_check_kernel(
    A,
    A_SCALE,
    BNORM,
    GU_WEIGHT,
    GU_SCALE,
    W_SORTED,
    MD_OUT,
    MD_ROWSCALE,
    DN_RAW,
    DN_OUT,
    DN_ROWSCALE,
    METRICS,
    STATUS,
    M_CONST: tl.constexpr,
    I_CONST: tl.constexpr,
    H_CONST: tl.constexpr,
    BLOCK_K_MD: tl.constexpr,
):
    # 3 rows x 128 columns. Rows cover expert-0 tail, expert-1 boundary, and
    # expert-7 final tail.  DN_RAW already covers all 256 columns needed by
    # each selected DN chunk's scale.
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
    md_raw = (g * 0.5) * (1.0 + th) * u * w
    bn = tl.load(BNORM + expert)
    bound = a_scale * a_scale * bn * bn * tl.abs(w)
    bbits = tl.maximum(bound, 1.0e-30).to(tl.int32, bitcast=True)
    bexp = (bbits >> 23) & 0xFF
    md_scale_ref = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
    md_inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
    md_q_ref = (md_raw * md_inv).to(tl.float8e4nv)
    md_ref = md_q_ref.to(tl.float32) * md_scale_ref
    md_q = tl.load(MD_OUT + row64 * I_CONST + md_n)
    md_scale = tl.load(MD_ROWSCALE + row)
    md_actual = md_q.to(tl.float32) * md_scale

    chunk = tl.where(check == 0, 0, tl.where(check == 1, 8, 15))
    dn_offset = tl.where(check == 2, 128 + lane, lane)
    dn_n = chunk * 256 + dn_offset
    raw_chunk = tl.load(DN_RAW + check * 256 + tl.arange(0, 256))
    dn_raw = tl.load(DN_RAW + check * 256 + dn_offset)
    dn_scale_ref = tl.maximum(
        md_scale * tl.max(tl.abs(raw_chunk), axis=0) * (1.0 / 448.0),
        1.0e-12,
    )
    dn_q_ref = (dn_raw * (md_scale / dn_scale_ref)).to(tl.float8e4nv)
    dn_ref = dn_q_ref.to(tl.float32) * dn_scale_ref
    dn_q = tl.load(DN_OUT + row64 * H_CONST + dn_n)
    dn_scale = tl.load(DN_ROWSCALE + row * (H_CONST // 256) + chunk)
    dn_actual = dn_q.to(tl.float32) * dn_scale

    tl.atomic_add(METRICS + 0, md_ref * md_ref)
    tl.atomic_add(METRICS + 1, (md_actual - md_ref) * (md_actual - md_ref))
    tl.atomic_add(METRICS + 2, dn_ref * dn_ref)
    tl.atomic_add(METRICS + 3, (dn_actual - dn_ref) * (dn_actual - dn_ref))
    md_ref_bits = md_q_ref.to(tl.uint8, bitcast=True)
    md_bits = md_q.to(tl.uint8, bitcast=True)
    dn_ref_bits = dn_q_ref.to(tl.uint8, bitcast=True)
    dn_bits = dn_q.to(tl.uint8, bitcast=True)
    tl.atomic_add(METRICS + 4, (md_bits != md_ref_bits).to(tl.float32))
    tl.atomic_add(METRICS + 5, (dn_bits != dn_ref_bits).to(tl.float32))
    tl.atomic_add(METRICS + 6, dn_scale_ref * dn_scale_ref)
    tl.atomic_add(
        METRICS + 7, (dn_scale - dn_scale_ref) * (dn_scale - dn_scale_ref)
    )

    tl.atomic_or(STATUS, 1, mask=md_scale != md_scale_ref)
    md_ref_finite = (md_ref == md_ref) & (tl.abs(md_ref) <= 3.402823466e38)
    md_actual_finite = (md_actual == md_actual) & (tl.abs(md_actual) <= 3.402823466e38)
    dn_ref_finite = (dn_ref == dn_ref) & (tl.abs(dn_ref) <= 3.402823466e38)
    dn_actual_finite = (dn_actual == dn_actual) & (tl.abs(dn_actual) <= 3.402823466e38)
    scale_finite = (dn_scale == dn_scale) & (dn_scale > 0.0) & (tl.abs(dn_scale) <= 3.402823466e38)
    tl.atomic_or(STATUS, 2, mask=md_ref_finite == 0)
    tl.atomic_or(STATUS, 4, mask=md_actual_finite == 0)
    tl.atomic_or(STATUS, 8, mask=dn_ref_finite == 0)
    tl.atomic_or(STATUS, 16, mask=dn_actual_finite == 0)
    tl.atomic_or(STATUS, 32, mask=scale_finite == 0)


@triton.jit
def finalize_metrics_kernel(METRICS, SUMMARY, CHECK_COUNT: tl.constexpr):
    md_signal = tl.maximum(tl.load(METRICS + 0), 1.0e-30)
    dn_signal = tl.maximum(tl.load(METRICS + 2), 1.0e-30)
    scale_signal = tl.maximum(tl.load(METRICS + 6), 1.0e-30)
    md_err = tl.maximum(tl.load(METRICS + 1), md_signal * 1.0e-12)
    dn_err = tl.maximum(tl.load(METRICS + 3), dn_signal * 1.0e-12)
    scale_err = tl.maximum(tl.load(METRICS + 7), scale_signal * 1.0e-12)
    tl.store(SUMMARY + 0, tl.sqrt(md_err / md_signal))
    tl.store(SUMMARY + 1, 3.010299956639812 * tl.log2(md_signal / md_err))
    tl.store(SUMMARY + 2, tl.sqrt(dn_err / dn_signal))
    tl.store(SUMMARY + 3, 3.010299956639812 * tl.log2(dn_signal / dn_err))
    tl.store(SUMMARY + 4, 1.0 - tl.load(METRICS + 4) / CHECK_COUNT)
    tl.store(SUMMARY + 5, 1.0 - tl.load(METRICS + 5) / CHECK_COUNT)
    tl.store(SUMMARY + 6, tl.sqrt(scale_err / scale_signal))


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
        BLOCK_M=BM,
        BLOCK_N=MD_BN,
        BLOCK_K=BK,
        GROUP_M=8,
        num_warps=8,
        num_stages=4,
        maxnreg=168,
    )


def launch_dn(
    act_desc,
    act_scale,
    dn_desc,
    dn_scale,
    down,
    down_scale,
    expert_ids,
    counts,
    split_cum,
    tile_num,
    tile_cum,
    num_tiles,
):
    return active_c2_dn_kernel[(GRID,)](
        act_desc,
        act_scale,
        dn_desc,
        dn_scale,
        down,
        down_scale,
        expert_ids,
        counts,
        split_cum,
        tile_num,
        tile_cum,
        num_tiles,
        M_CONST=M,
        N_CONST=H,
        K_CONST=I,
        stride_cm=down.stride(0),
        stride_cn=down.stride(1),
        BLOCK_M=BM,
        BLOCK_N=DN_BN,
        BLOCK_K=BK,
        GROUP_M=32,
        FLAT=False,
        num_warps=8,
        num_stages=3,
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

    # Generate physical interleaved GU directly.  Max row norm is invariant to
    # the active logical-to-physical row permutation used by _get_int_gu.
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

    dn_bf16 = torch.randn((E * H, I), dtype=torch.bfloat16, device=device)
    dn_q = torch.empty((E * H, I), dtype=torch.float8_e4m3fn, device=device)
    dn_scale = torch.empty((E * H,), dtype=torch.float32, device=device)
    quantize_weight_kernel[(E * H,)](
        dn_bf16,
        dn_q,
        dn_scale,
        bnorm,
        K_CONST=I,
        ROWS_PER_EXPERT=H,
        WRITE_BNORM=False,
        BLOCK_K=16384,
        num_warps=8,
        num_stages=1,
    )
    del dn_bf16
    dn_desc = TensorDescriptor(dn_q, dn_q.shape, dn_q.stride(), [DN_BN, BK])

    act = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    act_scale = torch.empty((M,), dtype=torch.float32, device=device)
    down = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=device)
    down_scale = torch.empty((M, H // DN_BN), dtype=torch.float32, device=device)
    act_desc = TensorDescriptor(act, act.shape, act.stride(), [BM, BK])

    def call_md():
        return launch_md(
            a_desc,
            a_scale,
            bnorm,
            gu_desc,
            gu_scale,
            weights,
            order,
            act,
            act_scale,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
        )

    def call_dn():
        return launch_dn(
            act_desc,
            act_scale,
            dn_desc,
            dn_scale,
            down,
            down_scale,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
        )

    def call_md_dn():
        launch_md(
            a_desc,
            a_scale,
            bnorm,
            gu_desc,
            gu_scale,
            weights,
            order,
            act,
            act_scale,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
        )
        return call_dn()

    compiled_md = call_md()
    compiled_dn = call_dn()

    dn_raw = torch.empty((3, DN_BN), dtype=torch.float32, device=device)
    dn_chunk_oracle_kernel[(3 * DN_BN,)](
        act,
        dn_q,
        dn_scale,
        dn_raw,
        M_CONST=M,
        H_CONST=H,
        I_CONST=I,
        BLOCK_K=16384,
        num_warps=8,
        num_stages=1,
    )
    metrics = torch.zeros((8,), dtype=torch.float32, device=device)
    status = torch.zeros((1,), dtype=torch.int32, device=device)
    summary = torch.empty((7,), dtype=torch.float32, device=device)
    math_check_kernel[(3 * MD_BN,)](
        a_q,
        a_scale,
        bnorm,
        gu_q,
        gu_scale,
        weights,
        act,
        act_scale,
        dn_raw,
        down,
        down_scale,
        metrics,
        status,
        M_CONST=M,
        I_CONST=I,
        H_CONST=H,
        BLOCK_K_MD=H,
        num_warps=8,
        num_stages=1,
    )
    finalize_metrics_kernel[(1,)](
        metrics,
        summary,
        CHECK_COUNT=3 * MD_BN,
        num_warps=1,
        num_stages=1,
    )
    status_code = int(status[0].item())
    md_rel = float(summary[0].item())
    md_sqnr = float(summary[1].item())
    dn_rel = float(summary[2].item())
    dn_sqnr = float(summary[3].item())
    md_exact = float(summary[4].item())
    dn_exact = float(summary[5].item())
    dn_scale_rel = float(summary[6].item())
    math_passed = (
        status_code == 0
        and md_sqnr >= SQNR_THRESHOLD_DB
        and dn_sqnr >= SQNR_THRESHOLD_DB
        and dn_scale_rel <= DN_SCALE_REL_THRESHOLD
    )
    if not math_passed:
        print(
            'P1MD {"math_check_passed":false,"status":'
            + str(status_code)
            + ',"md_sqnr_db":' + str(md_sqnr)
            + ',"dn_sqnr_db":' + str(dn_sqnr)
            + ',"md_rel":' + str(md_rel)
            + ',"dn_rel":' + str(dn_rel)
            + ',"dn_scale_rel":' + str(dn_scale_rel)
            + ',"threshold_db":35.0}'
        )
        raise RuntimeError("P1MD c2 stage math check failed")

    groups_text = []
    for group in range(3):
        if group == 0:
            md_ms = float(triton.testing.do_bench(call_md, warmup=5, rep=10))
            dn_ms = float(triton.testing.do_bench(call_dn, warmup=5, rep=10))
            combo_ms = float(triton.testing.do_bench(call_md_dn, warmup=5, rep=10))
            order_name = "MD_DN_COMBINED"
        elif group == 1:
            dn_ms = float(triton.testing.do_bench(call_dn, warmup=5, rep=10))
            combo_ms = float(triton.testing.do_bench(call_md_dn, warmup=5, rep=10))
            md_ms = float(triton.testing.do_bench(call_md, warmup=5, rep=10))
            order_name = "DN_COMBINED_MD"
        else:
            combo_ms = float(triton.testing.do_bench(call_md_dn, warmup=5, rep=10))
            md_ms = float(triton.testing.do_bench(call_md, warmup=5, rep=10))
            dn_ms = float(triton.testing.do_bench(call_dn, warmup=5, rep=10))
            order_name = "COMBINED_MD_DN"
        groups_text.append(
            '{"group":' + str(group)
            + ',"order":"' + order_name
            + '","md_ms":' + str(md_ms)
            + ',"dn_ms":' + str(dn_ms)
            + ',"md_plus_dn_sum_ms":' + str(md_ms + dn_ms)
            + ',"combined_ms":' + str(combo_ms)
            + ',"combined_over_separate_sum":' + str(combo_ms / (md_ms + dn_ms))
            + '}'
        )

    report = (
        '{"kind":"fp8_c2_md_dn_stage_custom","shape":{"T":16384,'
        '"H":4096,"E":8,"I":14336,"topk":2,"M":32768},'
        '"rank_semantics":"private_T16384_input_no_activation_collective",'
        '"route":{"kind":"deterministic_affine_bijection",'
        '"counts":[4016,4176,4016,4176,4016,4176,4016,4176],'
        '"metadata_tiles":260,"all_experts_have_partial_BM128_tail":true},'
        '"geometry":{"md":{"grid":132,"BM":128,"BN":128,"BK":128,'
        '"GM":8,"warps":8,"stages":4,"maxnreg":168},'
        '"dn":{"grid":132,"BM":128,"BN":256,"BK":128,"GM":32,'
        '"warps":8,"stages":3,"flatten":false}},'
        '"math_check":{"passed":true,"status":' + str(status_code)
        + ',"rows":[4015,4016,32767],"comparisons_per_stage":384,'
        '"md_reference":"independent_F32_K4096_then_same_FP8_contract",'
        '"dn_reference":"independent_F32_K14336_full_256col_chunk_max_then_same_FP8_contract",'
        '"threshold_db":35.0,"md_rel":' + str(md_rel)
        + ',"md_sqnr_db":' + str(md_sqnr)
        + ',"dn_rel":' + str(dn_rel)
        + ',"dn_sqnr_db":' + str(dn_sqnr)
        + ',"md_exact_fraction":' + str(md_exact)
        + ',"dn_exact_fraction":' + str(dn_exact)
        + ',"md_rowscale_bitwise":true,"dn_scale_rel":' + str(dn_scale_rel)
        + ',"dn_scale_rel_threshold":0.001,"finite":true},'
        '"resources":{"md":{"n_regs":' + str(compiled_md.n_regs)
        + ',"n_spills":' + str(compiled_md.n_spills)
        + ',"shared":' + str(compiled_md.metadata.shared)
        + '},"dn":{"n_regs":' + str(compiled_dn.n_regs)
        + ',"n_spills":' + str(compiled_dn.n_spills)
        + ',"shared":' + str(compiled_dn.metadata.shared) + '}},'
        '"timing":{"method":"do_bench_warmup5_rep10",'
        '"scope":"MD_only_DN_only_and_two_launch_MD_plus_DN",'
        '"groups":[' + ",".join(groups_text) + ']},'
        '"excluded":{"allocation":true,"route_and_metadata":true,'
        '"A_GU_DN_quantization":true,"BNORM_and_descriptors":true,'
        '"weight_cache_and_collectives":true,"final_gather":true},'
        '"limitations":"single_GPU_c2_stage_diagnostic_not_P1_end_to_end_or_candidate"}'
    )
    print("P1MD " + report)
    raise RuntimeError("P1MD done")


def run_kernel(x):
    return x


run_diagnostic()
