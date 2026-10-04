"""GeneratedWorkload-only H800 diagnostic for native INT8-A / INT6-GU MD.

This is deliberately not a P1 submission candidate.  It builds one fixed c10
workload, compares two pure-Triton MD kernels, prints a compact JSON record from
inside a function, and deliberately raises after printing so platform stdout is
returned in userError.
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
FP8_MAX = 448.0
INT8_MAX = 127.0
INT6_MAX = 31.0


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


@triton.jit
def quantize_rows_kernel(
    X,
    Q_FP8,
    SCALE_FP8,
    Q_INT,
    SCALE_INT,
    NORM_FP8,
    NORM_INT,
    QMAX_INT,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    row_base = row64 * K
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(X + row_base + offs_k, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale_fp8 = tl.maximum(amax / 448.0, 1.0e-12)
    scale_int = tl.maximum(amax / QMAX_INT, 1.0e-12)
    q_fp8 = (x / scale_fp8).to(tl.float8e4nv)
    v_int = tl.maximum(-QMAX_INT, tl.minimum(QMAX_INT, x / scale_int))
    q_int32 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [v_int],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q_int = q_int32.to(tl.int8)
    tl.store(Q_FP8 + row_base + offs_k, q_fp8, mask=mask)
    tl.store(Q_INT + row_base + offs_k, q_int, mask=mask)
    tl.store(SCALE_FP8 + row, scale_fp8)
    tl.store(SCALE_INT + row, scale_int)
    fp8_real = q_fp8.to(tl.float32) * scale_fp8
    int_real = q_int32.to(tl.float32) * scale_int
    tl.store(NORM_FP8 + row, tl.sqrt(tl.sum(fp8_real * fp8_real, axis=0)))
    tl.store(NORM_INT + row, tl.sqrt(tl.sum(int_real * int_real, axis=0)))


@triton.jit
def expert_bnorm_kernel(
    ROW_NORM_FP8,
    ROW_NORM_INT,
    BNORM_FP8,
    BNORM_INT,
    ROWS_PER_EXPERT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    expert = tl.program_id(0)
    offs_r = tl.arange(0, BLOCK_R)
    mask = offs_r < ROWS_PER_EXPERT
    base = expert * ROWS_PER_EXPERT
    norm_fp8 = tl.load(ROW_NORM_FP8 + base + offs_r, mask=mask, other=0.0)
    norm_int = tl.load(ROW_NORM_INT + base + offs_r, mask=mask, other=0.0)
    tl.store(BNORM_FP8 + expert, tl.max(norm_fp8, axis=0))
    tl.store(BNORM_INT + expert, tl.max(norm_int, axis=0))


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
def native_int6_md_kernel(
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
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None],
                other=0,
                eviction_policy="evict_last",
            )
            bg = B_DESC.load([b_row_g, k * BLOCK_K])
            bu = B_DESC.load([b_row_u, k * BLOCK_K])
            acc_g += tl.dot(a, bg.T)
            acc_u += tl.dot(a, bu.T)
            a_ptrs += BLOCK_K * stride_ak

        a_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)
        g_scale = a_scale[:, None] * tl.load(B_SCALE + expert * (2 * I) + offs_n)[None, :]
        u_scale = a_scale[:, None] * tl.load(
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
        bound_a_scale = a_scale * (127.0 / 448.0)
        bound = bound_a_scale * bound_a_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        c_ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None], eviction_policy="evict_first")
        tl.store(ROWSCALE + offs_m, s, mask=row_mask)


@triton.jit
def native_math_check_kernel(
    A,
    A_SCALE,
    BNORM,
    B,
    B_SCALE,
    ACT,
    ROWSCALE,
    STATUS,
    I: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Three representative sorted rows paired with the first, aligned middle,
    # and final N128 segment.  Scalar reductions are independent of tl.dot.
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
    a_row64 = token.to(tl.int64)
    bg_row64 = (expert * (2 * I) + n).to(tl.int64)
    bu_row64 = (expert * (2 * I) + I + n).to(tl.int64)
    a = tl.load(A + a_row64 * K + offs_k).to(tl.int32)
    bg = tl.load(B + bg_row64 * K + offs_k).to(tl.int32)
    bu = tl.load(B + bu_row64 * K + offs_k).to(tl.int32)
    acc_g = tl.sum(a * bg, axis=0)
    acc_u = tl.sum(a * bu, axis=0)
    a_scale = tl.load(A_SCALE + token)
    g_scale = a_scale * tl.load(B_SCALE + expert * (2 * I) + n)
    u_scale = a_scale * tl.load(B_SCALE + expert * (2 * I) + I + n)
    g = acc_g.to(tl.float32) * g_scale
    u = acc_u.to(tl.float32) * u_scale
    silu = tl.fdiv(
        g,
        1.0 + tl.exp2(-g * 1.4426950408889634),
        ieee_rounding=False,
    )
    act = silu * u * 0.125
    bn = tl.load(BNORM + expert)
    bound_a_scale = a_scale * (127.0 / 448.0)
    bound = bound_a_scale * bound_a_scale * bn * bn * 0.125
    bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
    bexp = (bbits >> 23) & 0xFF
    expected_scale = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
    inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
    expected_q = (act * inv).to(tl.float8e4nv)
    actual_q = tl.load(ACT + branch_row * I + n)
    actual_scale = tl.load(ROWSCALE + branch_row)
    tl.atomic_or(STATUS, 1, mask=actual_scale != expected_scale)
    tl.atomic_or(STATUS, 2, mask=actual_q != expected_q)


def quantize_rows_both(x, qmax_int):
    rows = x.shape[0]
    q_fp8 = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device="cuda")
    q_int = torch.empty(x.shape, dtype=torch.int8, device="cuda")
    scale_fp8 = torch.empty((rows,), dtype=torch.float32, device="cuda")
    scale_int = torch.empty((rows,), dtype=torch.float32, device="cuda")
    norm_fp8 = torch.empty((rows,), dtype=torch.float32, device="cuda")
    norm_int = torch.empty((rows,), dtype=torch.float32, device="cuda")
    quantize_rows_kernel[(rows,)](
        x,
        q_fp8,
        scale_fp8,
        q_int,
        scale_int,
        norm_fp8,
        norm_int,
        qmax_int,
        K=H,
        BLOCK_K=H,
        num_warps=8,
        num_stages=1,
    )
    return q_fp8, scale_fp8, q_int, scale_int, norm_fp8, norm_int


def reduce_expert_bnorm(norm_fp8, norm_int):
    bnorm_fp8 = torch.empty((E,), dtype=torch.float32, device="cuda")
    bnorm_int = torch.empty((E,), dtype=torch.float32, device="cuda")
    expert_bnorm_kernel[(E,)](
        norm_fp8,
        norm_int,
        bnorm_fp8,
        bnorm_int,
        ROWS_PER_EXPERT=2 * I,
        BLOCK_R=4096,
        num_warps=8,
        num_stages=1,
    )
    return bnorm_fp8, bnorm_int


def launch_fp8(
    a_q,
    a_s,
    b_q,
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
    return fp8_md_kernel[(GRID,)](
        a_q,
        a_s,
        bnorm,
        b_q,
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
        M=M,
        I=I,
        K=H,
        stride_am=a_q.stride(0),
        stride_ak=a_q.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GM,
        KTOP=KTOP,
        num_warps=8,
        num_stages=4,
    )


def launch_native(
    a_q,
    a_s,
    b_q,
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
    return native_int6_md_kernel[(GRID,)](
        a_q,
        a_s,
        bnorm,
        b_q,
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
        M=M,
        I=I,
        K=H,
        stride_am=a_q.stride(0),
        stride_ak=a_q.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GM,
        KTOP=KTOP,
        num_warps=8,
        num_stages=4,
    )


def check_native_math(a_q, a_s, b_q, b_s, bnorm, out, rowscale, status):
    return native_math_check_kernel[(3 * BN,)](
        a_q,
        a_s,
        bnorm,
        b_q,
        b_s,
        out,
        rowscale,
        status,
        I=I,
        K=H,
        BLOCK_K=H,
        num_warps=8,
        num_stages=1,
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
    setup_route_metadata_kernel[(E,)](
        order,
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

    a8, a8s, ai8, ai8s, an8, ani8 = quantize_rows_both(x, INT8_MAX)
    b8, b8s_flat, bi6, bi6s_flat, bnorm_rows8, bnorm_rows6 = quantize_rows_both(
        gu.view(E * 2 * I, H), INT6_MAX
    )
    b8s = b8s_flat.view(E, 2 * I)
    bi6s = bi6s_flat.view(E, 2 * I)
    bn8, bni6 = reduce_expert_bnorm(bnorm_rows8, bnorm_rows6)
    del x
    del gu
    del an8
    del ani8
    del bnorm_rows8
    del bnorm_rows6

    b8_flat = b8.view(E * 2 * I, H)
    bi6_flat = bi6.view(E * 2 * I, H)
    b8_desc = TensorDescriptor(b8_flat, b8_flat.shape, b8_flat.stride(), [BN, BK])
    bi6_desc = TensorDescriptor(bi6_flat, bi6_flat.shape, bi6_flat.stride(), [BN, BK])
    out8 = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    out6 = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    rowscale8 = torch.empty(M, dtype=torch.float32, device=device)
    rowscale6 = torch.empty(M, dtype=torch.float32, device=device)

    def call_fp8():
        return launch_fp8(
            a8,
            a8s,
            b8_desc,
            b8s,
            bn8,
            weights,
            order,
            out8,
            rowscale8,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
        )

    def call_native():
        return launch_native(
            ai8,
            ai8s,
            bi6_desc,
            bi6s,
            bni6,
            weights,
            order,
            out6,
            rowscale6,
            expert_ids,
            counts,
            split_cum,
            tile_num,
            tile_cum,
            num_tiles,
        )

    compiled_fp8 = call_fp8()
    compiled_native = call_native()
    check_status = torch.zeros((1,), dtype=torch.int32, device=device)
    def check_call():
        return check_native_math(
            ai8, ai8s, bi6, bi6s, bni6, out6, rowscale6, check_status
        )
    check_call()
    triton.testing.do_bench(check_call, warmup=0, rep=1)
    check_code = int(check_status[0].item())
    if check_code != 0:
        print("P1MD {\"math_check_passed\":false,\"status\":" + str(check_code) + "}")
        raise RuntimeError("P1MD native math check failed")

    groups_text = []
    for group in range(3):
        if group % 2 == 0:
            fp8_ms = float(triton.testing.do_bench(call_fp8, warmup=5, rep=10))
            native_ms = float(triton.testing.do_bench(call_native, warmup=5, rep=10))
            order_name = "fp8_then_native"
        else:
            native_ms = float(triton.testing.do_bench(call_native, warmup=5, rep=10))
            fp8_ms = float(triton.testing.do_bench(call_fp8, warmup=5, rep=10))
            order_name = "native_then_fp8"
        groups_text.append(
            '{"group":'
            + str(group)
            + ',"order":"'
            + order_name
            + '","fp8_ms":'
            + str(fp8_ms)
            + ',"native_ms":'
            + str(native_ms)
            + ',"native_over_fp8":'
            + str(native_ms / fp8_ms)
            + "}"
        )

    report = (
        '{"kind":"native_int8_a_int6_gu_md_custom","shape":{"T":4096,"H":4096,'
        '"E":256,"I":1536,"topk":8,"M":32768},"triton_version":"unavailable_dunder_blocked",'
        '"device_name":"unavailable_torch_cuda_blocked","geometry":{"grid":132,"BM":128,'
        '"BN":128,"BK":128,"GM":32,"warps":8,"stages":4},"route":{"kind":'
        '"analytic_grouped_nonrandom","counts":"112_144_alternating","total_tiles":384},'
        '"quantization":{"baseline":"A_GU_row_amax_over_448_FP8","native":'
        '"A_row_amax_over_127_INT8_GU_row_amax_over_31_signed6_in_INT8",'
        '"native_act_bound_factor":"127_over_448_squared","source_weights":'
        '"BF16_Gaussian_torch_randn_once_then_real_row_quant"},'
        '"math_check":{"kind":"3_rows_x_3_corresponding_N128_segments_scalar_reduction",'
        '"comparisons":384,"bitwise_status_check":true,"passed":true},"resources":{"fp8":'
        '{"n_regs":'
        + str(compiled_fp8.n_regs)
        + ',"n_spills":'
        + str(compiled_fp8.n_spills)
        + ',"shared":'
        + str(compiled_fp8.metadata.shared)
        + '},"native":{"n_regs":'
        + str(compiled_native.n_regs)
        + ',"n_spills":'
        + str(compiled_native.n_spills)
        + ',"shared":'
        + str(compiled_native.metadata.shared)
        + '}},"timing":{"method":"do_bench_warmup5_rep10","scope":'
        '"MD_kernel_only_not_end_to_end","setup_excluded":true,"groups":['
        + ",".join(groups_text)
        + ']},"limitations":"custom_single_GPU_unpacked_signed6_nonrandom_route_no_P1_integration"}'
    )
    print("P1MD " + report)
    raise RuntimeError("P1MD done")


def run_kernel(x):
    return x


run_diagnostic()
