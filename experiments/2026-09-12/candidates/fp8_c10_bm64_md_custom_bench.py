"""GeneratedWorkload-only H800 diagnostic for c10 MD BLOCK_M=64.

This is not a P1 submission candidate.  It builds one fixed c10 workload and
compares the active token-only FP8 pointer-gather MD at BM128 against the same
kernel at BM64.  Allocation, both metadata tables, A/GU quantization, BNORM,
and descriptor construction are outside every timing.
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
BM64 = 64
BN = 128
BK = 128
GM = 32
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
    EXPERT_IDS_64,
    SPLIT_CUM_64,
    TILE_NUM_64,
    TILE_CUM_64,
    NUM_TILES_64,
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

    # BM64 needs 2 tiles for 112 rows and 3 for 144 rows: 640 total.
    n_tiles_64 = 2 + parity
    tile_begin_64 = (group // 2) * 40 + parity * 16 + lane * n_tiles_64
    tile_cum_64 = tile_begin_64 + n_tiles_64
    tile_offsets_64 = tl.arange(0, 4)
    tile_mask_64 = tile_offsets_64 < n_tiles_64
    tile_index_64 = tile_begin_64 + tile_offsets_64
    tl.store(EXPERT_IDS_64 + tile_index_64, expert, mask=tile_mask_64)
    tl.store(SPLIT_CUM_64 + tile_index_64, row_begin, mask=tile_mask_64)
    tl.store(TILE_NUM_64 + tile_index_64, n_tiles_64, mask=tile_mask_64)
    tl.store(TILE_CUM_64 + tile_index_64, tile_cum_64, mask=tile_mask_64)
    if expert == 0:
        tl.store(NUM_TILES_64, 640)


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
def md_math_check_kernel(
    A,
    A_SCALE,
    BNORM,
    B,
    B_SCALE,
    W,
    ORDER,
    ACT_128,
    ROWSCALE_128,
    ACT_64,
    ROWSCALE_64,
    METRICS,
    STATUS,
    M_CONST: tl.constexpr,
    E_CONST: tl.constexpr,
    I_CONST: tl.constexpr,
    K_CONST: tl.constexpr,
    KTOP_CONST: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Three rows x N128.  row 64 starts the second BM64 tile of expert 0;
    # row 1024 starts the second BM128 / third BM64 tile of expert 8; the final
    # row is the last valid row of expert 255's 144-row tail.
    pid = tl.program_id(0)
    check_index = pid // 128
    col_in_segment = pid % 128
    branch_row = tl.where(
        check_index == 0,
        64,
        tl.where(check_index == 1, 1024, M_CONST - 1),
    )
    expert = tl.where(
        check_index == 0,
        0,
        tl.where(check_index == 1, 8, E_CONST - 1),
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
    expert64 = expert.to(tl.int64)
    bg_row64 = expert64 * (2 * I_CONST) + n
    bu_row64 = expert64 * (2 * I_CONST) + I_CONST + n
    a = tl.load(A + token64 * K_CONST + offs_k).to(tl.float32)
    bg = tl.load(B + bg_row64 * K_CONST + offs_k).to(tl.float32)
    bu = tl.load(B + bu_row64 * K_CONST + offs_k).to(tl.float32)
    acc_g = tl.sum(a * bg, axis=0)
    acc_u = tl.sum(a * bu, axis=0)
    a_scale = tl.load(A_SCALE + token)
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

    scale_128 = tl.load(ROWSCALE_128 + branch_row)
    scale_64 = tl.load(ROWSCALE_64 + branch_row)
    q_128 = tl.load(ACT_128 + branch_row * I_CONST + n)
    q_64 = tl.load(ACT_64 + branch_row * I_CONST + n)
    actual_128 = q_128.to(tl.float32) * scale_128
    actual_64 = q_64.to(tl.float32) * scale_64

    ref2 = reference * reference
    err_128 = (actual_128 - reference) * (actual_128 - reference)
    err_64 = (actual_64 - reference) * (actual_64 - reference)
    base2 = actual_128 * actual_128
    err_pair = (actual_64 - actual_128) * (actual_64 - actual_128)
    tl.atomic_add(METRICS + 0, ref2)
    tl.atomic_add(METRICS + 1, err_128)
    tl.atomic_add(METRICS + 2, err_64)
    tl.atomic_add(METRICS + 3, base2)
    tl.atomic_add(METRICS + 4, err_pair)
    reference_bits = reference_q.to(tl.uint8, bitcast=True)
    bits_128 = q_128.to(tl.uint8, bitcast=True)
    bits_64 = q_64.to(tl.uint8, bitcast=True)
    tl.atomic_add(METRICS + 5, (bits_128 != reference_bits).to(tl.float32))
    tl.atomic_add(METRICS + 6, (bits_64 != reference_bits).to(tl.float32))
    tl.atomic_add(METRICS + 7, (bits_64 != bits_128).to(tl.float32))

    tl.atomic_or(STATUS, 1, mask=scale_128 != expected_scale)
    tl.atomic_or(STATUS, 2, mask=scale_64 != expected_scale)
    ref_finite = (reference == reference) & (tl.abs(reference) <= 3.402823466e38)
    finite_128 = (actual_128 == actual_128) & (tl.abs(actual_128) <= 3.402823466e38)
    finite_64 = (actual_64 == actual_64) & (tl.abs(actual_64) <= 3.402823466e38)
    tl.atomic_or(STATUS, 4, mask=ref_finite == 0)
    tl.atomic_or(STATUS, 8, mask=finite_128 == 0)
    tl.atomic_or(STATUS, 16, mask=finite_64 == 0)


@triton.jit
def finalize_metrics_kernel(METRICS, SUMMARY, CHECK_COUNT: tl.constexpr):
    ref2 = tl.load(METRICS + 0)
    err_128 = tl.load(METRICS + 1)
    err_64 = tl.load(METRICS + 2)
    base2 = tl.load(METRICS + 3)
    err_pair = tl.load(METRICS + 4)
    mismatch_128 = tl.load(METRICS + 5)
    mismatch_64 = tl.load(METRICS + 6)
    mismatch_pair = tl.load(METRICS + 7)
    safe_ref = tl.maximum(ref2, 1.0e-30)
    safe_base = tl.maximum(base2, 1.0e-30)
    # Report floor: relative error >=1e-6 and SQNR <=120 dB.
    safe_128_err = tl.maximum(err_128, safe_ref * 1.0e-12)
    safe_64_err = tl.maximum(err_64, safe_ref * 1.0e-12)
    safe_pair_err = tl.maximum(err_pair, safe_base * 1.0e-12)
    tl.store(SUMMARY + 0, tl.sqrt(safe_128_err / safe_ref))
    tl.store(SUMMARY + 1, 3.010299956639812 * tl.log2(safe_ref / safe_128_err))
    tl.store(SUMMARY + 2, tl.sqrt(safe_64_err / safe_ref))
    tl.store(SUMMARY + 3, 3.010299956639812 * tl.log2(safe_ref / safe_64_err))
    tl.store(SUMMARY + 4, tl.sqrt(safe_pair_err / safe_base))
    tl.store(SUMMARY + 5, 3.010299956639812 * tl.log2(safe_base / safe_pair_err))
    tl.store(SUMMARY + 6, 1.0 - mismatch_128 / CHECK_COUNT)
    tl.store(SUMMARY + 7, 1.0 - mismatch_64 / CHECK_COUNT)
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


def launch_md(
    block_m,
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
        BLOCK_M=block_m,
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
    order = torch.empty((M,), dtype=torch.int64, device=device)
    counts = torch.empty((E,), dtype=torch.int32, device=device)

    metadata_size_128 = triton.cdiv(M, BM) + E
    expert_ids_128 = torch.empty((metadata_size_128,), dtype=torch.int32, device=device)
    split_cum_128 = torch.empty((metadata_size_128,), dtype=torch.int32, device=device)
    tile_num_128 = torch.empty((metadata_size_128,), dtype=torch.int32, device=device)
    tile_cum_128 = torch.empty((metadata_size_128,), dtype=torch.int32, device=device)
    num_tiles_128 = torch.empty((1,), dtype=torch.int32, device=device)

    metadata_size_64 = triton.cdiv(M, BM64) + E
    expert_ids_64 = torch.empty((metadata_size_64,), dtype=torch.int32, device=device)
    split_cum_64 = torch.empty((metadata_size_64,), dtype=torch.int32, device=device)
    tile_num_64 = torch.empty((metadata_size_64,), dtype=torch.int32, device=device)
    tile_cum_64 = torch.empty((metadata_size_64,), dtype=torch.int32, device=device)
    num_tiles_64 = torch.empty((1,), dtype=torch.int32, device=device)

    setup_route_metadata_kernel[(E,)](
        order,
        weights,
        counts,
        expert_ids_128,
        split_cum_128,
        tile_num_128,
        tile_cum_128,
        num_tiles_128,
        expert_ids_64,
        split_cum_64,
        tile_num_64,
        tile_cum_64,
        num_tiles_64,
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

    a_q = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=device)
    a_s = torch.empty((T,), dtype=torch.float32, device=device)
    launch_token_prep(x, a_q, a_s)
    del x
    out_128 = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    out_64 = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    rowscale_128 = torch.empty((M,), dtype=torch.float32, device=device)
    rowscale_64 = torch.empty((M,), dtype=torch.float32, device=device)

    def call_128():
        return launch_md(
            BM,
            a_q,
            a_s,
            b_desc,
            b_s,
            bnorm,
            weights,
            order,
            out_128,
            rowscale_128,
            expert_ids_128,
            counts,
            split_cum_128,
            tile_num_128,
            tile_cum_128,
            num_tiles_128,
        )

    def call_64():
        return launch_md(
            BM64,
            a_q,
            a_s,
            b_desc,
            b_s,
            bnorm,
            weights,
            order,
            out_64,
            rowscale_64,
            expert_ids_64,
            counts,
            split_cum_64,
            tile_num_64,
            tile_cum_64,
            num_tiles_64,
        )

    compiled_128 = call_128()
    compiled_64 = call_64()
    status = torch.zeros((1,), dtype=torch.int32, device=device)
    metrics = torch.zeros((8,), dtype=torch.float32, device=device)
    summary = torch.empty((9,), dtype=torch.float32, device=device)
    md_math_check_kernel[(3 * BN,)](
        a_q,
        a_s,
        bnorm,
        b_q,
        b_s,
        weights,
        order,
        out_128,
        rowscale_128,
        out_64,
        rowscale_64,
        metrics,
        status,
        M_CONST=M,
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
    rel_128 = float(summary[0].item())
    sqnr_128 = float(summary[1].item())
    rel_64 = float(summary[2].item())
    sqnr_64 = float(summary[3].item())
    pair_rel = float(summary[4].item())
    pair_sqnr = float(summary[5].item())
    exact_128 = float(summary[6].item())
    exact_64 = float(summary[7].item())
    pair_exact = float(summary[8].item())
    math_passed = (
        status_code == 0
        and sqnr_128 >= SQNR_THRESHOLD_DB
        and sqnr_64 >= SQNR_THRESHOLD_DB
    )
    if not math_passed:
        print(
            'P1MD {"math_check_passed":false,"status":'
            + str(status_code)
            + ',"bm128_sqnr_db":' + str(sqnr_128)
            + ',"bm64_sqnr_db":' + str(sqnr_64)
            + ',"bm128_rel":' + str(rel_128)
            + ',"bm64_rel":' + str(rel_64)
            + ',"bm64_vs_bm128_rel":' + str(pair_rel)
            + ',"bm64_vs_bm128_sqnr_db":' + str(pair_sqnr)
            + ',"threshold_db":35.0}'
        )
        raise RuntimeError("P1MD BM64 math check failed")

    groups_text = []
    for group in range(3):
        if group % 2 == 0:
            ms_128 = float(triton.testing.do_bench(call_128, warmup=5, rep=10))
            ms_64 = float(triton.testing.do_bench(call_64, warmup=5, rep=10))
            order_name = "BM128_then_BM64"
        else:
            ms_64 = float(triton.testing.do_bench(call_64, warmup=5, rep=10))
            ms_128 = float(triton.testing.do_bench(call_128, warmup=5, rep=10))
            order_name = "BM64_then_BM128"
        groups_text.append(
            '{"group":' + str(group)
            + ',"order":"' + order_name
            + '","bm128_ms":' + str(ms_128)
            + ',"bm64_ms":' + str(ms_64)
            + ',"bm64_over_bm128":' + str(ms_64 / ms_128)
            + '}'
        )

    report = (
        '{"kind":"fp8_c10_bm64_md_custom","shape":{"T":4096,"H":4096,'
        '"E":256,"I":1536,"topk":8,"M":32768},'
        '"geometry":{"grid":132,"BN":128,"BK":128,"GM":32,"warps":8,'
        '"stages":4,"persistent_range_stages":2,"bm128":{"BM":128,'
        '"tiles":384,"padded_rows":49152},"bm64":{"BM":64,"tiles":640,'
        '"padded_rows":40960}},'
        '"route":{"kind":"analytic_112_144_alternating","real_rows":32768},'
        '"paths":{"bm128":"active_token_FP8_A_pointer_gather_TMA_B",'
        '"bm64":"same_kernel_and_inputs_only_BLOCK_M_and_metadata_differ"},'
        '"math_check":{"kind":"3_boundary_rows_x_128cols_independent_F32_K4096_reduction",'
        '"rows":[64,1024,32767],"coverage":"BM64_boundary_BM128_boundary_expert_tail",'
        '"comparisons":384,"passed":true,"status":' + str(status_code)
        + ',"threshold_db":35.0,"bm128_rel":' + str(rel_128)
        + ',"bm128_sqnr_db":' + str(sqnr_128)
        + ',"bm64_rel":' + str(rel_64)
        + ',"bm64_sqnr_db":' + str(sqnr_64)
        + ',"bm64_vs_bm128_rel":' + str(pair_rel)
        + ',"bm64_vs_bm128_sqnr_db":' + str(pair_sqnr)
        + ',"bm128_exact_fraction":' + str(exact_128)
        + ',"bm64_exact_fraction":' + str(exact_64)
        + ',"bm64_vs_bm128_exact_fraction":' + str(pair_exact)
        + ',"rowscale_exact":{"bm128":true,"bm64":true},'
        + '"finite":{"reference":true,"bm128":true,"bm64":true}},'
        '"resources":{"bm128":{"n_regs":' + str(compiled_128.n_regs)
        + ',"n_spills":' + str(compiled_128.n_spills)
        + ',"shared":' + str(compiled_128.metadata.shared)
        + '},"bm64":{"n_regs":' + str(compiled_64.n_regs)
        + ',"n_spills":' + str(compiled_64.n_spills)
        + ',"shared":' + str(compiled_64.metadata.shared) + '}},'
        '"timing":{"method":"do_bench_warmup5_rep10","scope":"MD_kernel_only",'
        '"groups":[' + ",".join(groups_text) + ']},'
        '"excluded":{"metadata_build":true,"A_GU_quant_BNORM_allocations":true,'
        '"DN":"not_present"},"decision_gate":"BM64_requires_at_least_5pct_stable_MD_reduction",'
        '"limitations":"custom_single_GPU_c10_only_no_P1_integration"}'
    )
    print("P1MD " + report)
    raise RuntimeError("P1MD done")


def run_kernel(x):
    return x


run_diagnostic()
