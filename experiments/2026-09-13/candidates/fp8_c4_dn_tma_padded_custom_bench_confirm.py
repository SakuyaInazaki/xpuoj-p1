"""GeneratedWorkload-only H800 c4 padded DN output-store diagnostic.

Baseline and candidate use identical active c4 DN FP8 math and final gather.
The candidate stores every BM128xBN256 tile into padded C8 with a descriptor;
a timed GPU row map lets the otherwise unchanged gather address valid rows.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


T = 16384
H = 2048
E = 32
I = 1024
KTOP = 4
M = 65536
GRID = 132
BM = 128
BN = 256
BK = 128
GM = 32
PAD_ROWS = 69632


@triton.jit
def setup_metadata_kernel(
    ORDER, INV_ORDER, ROUTE_WEIGHT, COUNTS, EXPERT_IDS, SPLIT_CUM,
    TILE_NUM, TILE_CUM, NUM_TILES, M_CONST: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    expert = tl.program_id(0)
    parity = expert & 1
    n_rows = 2016 + 64 * parity
    row_begin = (expert // 2) * 4096 + parity * 2016
    offs = tl.arange(0, BLOCK_R)
    mask = offs < n_rows
    row = row_begin + offs
    src = (row * 8191) & (M_CONST - 1)
    tl.store(ORDER + row, src, mask=mask)
    tl.store(INV_ORDER + src, row, mask=mask)
    tl.store(ROUTE_WEIGHT + row, 0.25 + (row & 3).to(tl.float32) * 0.125, mask=mask)
    tl.store(COUNTS + expert, n_rows)

    n_tiles = 16 + parity
    tile_begin = (expert // 2) * 33 + parity * 16
    tile_cum = tile_begin + n_tiles
    toffs = tl.arange(0, 32)
    tmask = toffs < n_tiles
    tid = tile_begin + toffs
    tl.store(EXPERT_IDS + tid, expert, mask=tmask)
    tl.store(SPLIT_CUM + tid, row_begin, mask=tmask)
    tl.store(TILE_NUM + tid, n_tiles, mask=tmask)
    tl.store(TILE_CUM + tid, tile_cum, mask=tmask)
    if expert == 0:
        tl.store(NUM_TILES, 528)


@triton.jit
def quantize_rows_kernel(X, Q, SCALE, ROWS: tl.constexpr, K: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    base = row.to(tl.int64) * K
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) * (1.0 / 448.0), 1.0e-12)
    q = (x * (1.0 / scale)).to(tl.float8e4nv)
    tl.store(Q + base + offs, q, mask=mask)
    tl.store(SCALE + row, scale)


@triton.jit
def apply_route_scale_kernel(SCALE, ROUTE_WEIGHT, M_CONST: tl.constexpr):
    row = tl.program_id(0)
    scale = tl.load(SCALE + row) * tl.load(ROUTE_WEIGHT + row)
    # Exercise the active floor path with deterministic zero-scale rows.
    scale = tl.where((row & 4095) == 0, 0.0, scale)
    tl.store(SCALE + row, scale)


@triton.jit
def baseline_dn_kernel(
    A_DESC, A_SCALE, B_DESC, B_SCALE, C, CSCL,
    EXPERT_IDS, SPLIT_SIZE, SPLIT_CUM, TILE_NUM, TILE_CUM, NUM_TILES,
    M_CONST: tl.constexpr, N_CONST: tl.constexpr, K_CONST: tl.constexpr,
    stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(N_CONST, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)
    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, flatten=True):
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
        row_mask = offs_m < row_begin + n_rows
        a_row = row_begin + local_m * BLOCK_M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_row = expert * N_CONST + pid_n * BLOCK_N
        for kk in range(0, tl.cdiv(K_CONST, BLOCK_K)):
            a = A_DESC.load([a_row, kk * BLOCK_K])
            b = B_DESC.load([b_row, kk * BLOCK_K])
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
def candidate_dn_kernel(
    A_DESC, A_SCALE, B_DESC, B_SCALE, C_DESC, C, CSCL,
    EXPERT_IDS, SPLIT_SIZE, SPLIT_CUM, TILE_NUM, TILE_CUM, NUM_TILES,
    M_CONST: tl.constexpr, N_CONST: tl.constexpr, K_CONST: tl.constexpr,
    stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(N_CONST, BLOCK_N)
    total_tiles = tl.load(NUM_TILES)
    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, flatten=True):
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
        row_mask = offs_m < row_begin + n_rows
        a_row = row_begin + local_m * BLOCK_M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_row = expert * N_CONST + pid_n * BLOCK_N
        for kk in range(0, tl.cdiv(K_CONST, BLOCK_K)):
            a = A_DESC.load([a_row, kk * BLOCK_K])
            b = B_DESC.load([b_row, kk * BLOCK_K])
            acc = tl.dot(a, b.T, acc)
        a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        b_scale = tl.load(B_SCALE + expert * N_CONST + offs_n)
        acc = acc * b_scale[None, :]
        row_max = tl.max(tl.abs(acc), axis=1)
        s = tl.maximum(a_scale * row_max * (1.0 / 448.0), 1.0e-12)
        q = (acc * (a_scale / s)[:, None]).to(tl.float8e4nv)
        tl.store(CSCL + offs_m * num_block_n + pid_n, s, mask=row_mask)
        padded_row = (t_cum - t_num + local_m) * BLOCK_M
        C_DESC.store([padded_row, pid_n * BLOCK_N], q)


@triton.jit
def build_pad_row_kernel(PAD_ROW, EXPERT_IDS, SPLIT_SIZE, SPLIT_CUM,
                         TILE_NUM, TILE_CUM, NUM_TILES,
                         BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    total_tiles = tl.load(NUM_TILES)
    if pid_m < total_tiles:
        expert = tl.load(EXPERT_IDS + pid_m)
        n_rows = tl.load(SPLIT_SIZE + expert)
        row_begin = tl.load(SPLIT_CUM + pid_m)
        t_num = tl.load(TILE_NUM + pid_m)
        t_cum = tl.load(TILE_CUM + pid_m)
        local_m = pid_m - (t_cum - t_num)
        lane = tl.arange(0, BLOCK_M)
        row = row_begin + local_m * BLOCK_M + lane
        mask = row < row_begin + n_rows
        tl.store(PAD_ROW + row, pid_m * BLOCK_M + lane, mask=mask)


@triton.jit
def validate_pad_row_kernel(PAD_ROW, STATUS, M_CONST: tl.constexpr,
                            CAPACITY: tl.constexpr, BLOCK_R: tl.constexpr):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = row < M_CONST
    pair = row // 4096
    local = row % 4096
    expected = pair * 4224 + local + tl.where(local >= 2016, 32, 0)
    actual = tl.load(PAD_ROW + row, mask=mask, other=0)
    bad = (actual != expected) | (actual < 0) | (actual >= CAPACITY)
    tl.atomic_or(STATUS, 1, mask=tl.sum(bad.to(tl.int32), axis=0) != 0)


@triton.jit
def gather_f8_kernel(
    DOWN, DSCL, ORDER, OUT, T_CONST: tl.constexpr, H_CONST: tl.constexpr,
    stride_dm, stride_dh, stride_om, stride_oh, K_BRANCH: tl.constexpr,
    NCHUNK: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    t_mask = offs_t < T_CONST
    h_mask = offs_h < H_CONST
    mask = t_mask[:, None] & h_mask[None, :]
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    for j in tl.static_range(K_BRANCH):
        src = tl.load(ORDER + offs_t * K_BRANCH + j, mask=t_mask, other=0)
        d = tl.load(DOWN + src[:, None] * stride_dm + offs_h[None, :] * stride_dh,
                    mask=mask, other=0.0)
        sc = tl.load(DSCL + src * NCHUNK + (pid_h * BLOCK_H) // 256,
                     mask=t_mask, other=0.0)
        acc += d.to(tl.float32) * sc[:, None]
    tl.store(OUT + offs_t[:, None] * stride_om + offs_h[None, :] * stride_oh,
             acc.to(tl.bfloat16), mask=mask)


@triton.jit
def gather_padded_f8_kernel(
    DOWN, DSCL, ORDER, PAD_ROW, OUT,
    T_CONST: tl.constexpr, H_CONST: tl.constexpr,
    stride_dm, stride_dh, stride_om, stride_oh, K_BRANCH: tl.constexpr,
    NCHUNK: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    t_mask = offs_t < T_CONST
    h_mask = offs_h < H_CONST
    mask = t_mask[:, None] & h_mask[None, :]
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    for j in tl.static_range(K_BRANCH):
        src = tl.load(ORDER + offs_t * K_BRANCH + j, mask=t_mask, other=0)
        padded_src = tl.load(PAD_ROW + src, mask=t_mask, other=0)
        d = tl.load(DOWN + padded_src[:, None] * stride_dm + offs_h[None, :] * stride_dh,
                    mask=mask, other=0.0)
        sc = tl.load(DSCL + src * NCHUNK + (pid_h * BLOCK_H) // 256,
                     mask=t_mask, other=0.0)
        acc += d.to(tl.float32) * sc[:, None]
    tl.store(OUT + offs_t[:, None] * stride_om + offs_h[None, :] * stride_oh,
             acc.to(tl.bfloat16), mask=mask)


@triton.jit
def compare_kernel(A, B, MISMATCH, NUMEL: tl.constexpr, KIND: tl.constexpr,
                   SLOT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    if KIND == 0:
        av = tl.load(A + offs, mask=mask, other=0.0).to(tl.uint8, bitcast=True)
        bv = tl.load(B + offs, mask=mask, other=0.0).to(tl.uint8, bitcast=True)
    else:
        av = tl.load(A + offs, mask=mask, other=0.0)
        bv = tl.load(B + offs, mask=mask, other=0.0)
    count = tl.sum((av != bv).to(tl.int32), axis=0)
    tl.atomic_add(MISMATCH + SLOT, count)


@triton.jit
def compare_mapped_fp8_kernel(A, B, PAD_ROW, MISMATCH,
                              M_CONST: tl.constexpr, H_CONST: tl.constexpr,
                              BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    numel = M_CONST * H_CONST
    mask = offs < numel
    row = offs // H_CONST
    col = offs % H_CONST
    padded_row = tl.load(PAD_ROW + row, mask=mask, other=0)
    av = tl.load(A + offs, mask=mask, other=0.0).to(tl.uint8, bitcast=True)
    bv = tl.load(B + padded_row * H_CONST + col, mask=mask,
                 other=0.0).to(tl.uint8, bitcast=True)
    tl.atomic_add(MISMATCH, tl.sum((av != bv).to(tl.int32), axis=0))


@triton.jit
def zero_rows_check_kernel(DOWN, SCALE, PAD_ROW, STATUS, H_CONST: tl.constexpr,
                           NCHUNK: tl.constexpr, BLOCK_H: tl.constexpr):
    zero_index = tl.program_id(0)
    row = zero_index * 4096
    padded_row = tl.load(PAD_ROW + row)
    offs = tl.arange(0, BLOCK_H)
    for start in range(0, H_CONST, BLOCK_H):
        q = tl.load(DOWN + padded_row * H_CONST + start + offs).to(tl.float32)
        tl.atomic_or(STATUS, 1, mask=tl.sum((q != 0).to(tl.int32), axis=0) != 0)
    chunks = tl.arange(0, NCHUNK)
    sc = tl.load(SCALE + row * NCHUNK + chunks)
    tl.atomic_or(STATUS, 2, mask=tl.sum((sc != 1.0e-12).to(tl.int32), axis=0) != 0)


def launch_baseline(a_desc, a_scale, b_desc, b_scale, out, out_scale,
                    expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles):
    return baseline_dn_kernel[(GRID,)](
        a_desc, a_scale, b_desc, b_scale, out, out_scale, expert_ids, counts,
        split_cum, tile_num, tile_cum, num_tiles, M_CONST=M, N_CONST=H,
        K_CONST=I, stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
        num_warps=8, num_stages=4,
    )


def launch_candidate(a_desc, a_scale, b_desc, b_scale, out_desc, out, out_scale,
                     expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles):
    return candidate_dn_kernel[(GRID,)](
        a_desc, a_scale, b_desc, b_scale, out_desc, out, out_scale,
        expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M_CONST=M, N_CONST=H, K_CONST=I, stride_cm=out.stride(0),
        stride_cn=out.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_M=GM, num_warps=8, num_stages=4,
    )


def launch_gather(down, scale, order, out):
    return gather_f8_kernel[(triton.cdiv(T, 32), triton.cdiv(H, 256))](
        down, scale, order, out, T_CONST=T, H_CONST=H,
        stride_dm=down.stride(0), stride_dh=down.stride(1),
        stride_om=out.stride(0), stride_oh=out.stride(1), K_BRANCH=KTOP,
        NCHUNK=H // 256, BLOCK_T=32, BLOCK_H=256,
        num_warps=8, num_stages=1,
    )


def launch_padded_gather(down, scale, order, pad_row, out):
    return gather_padded_f8_kernel[(triton.cdiv(T, 32), triton.cdiv(H, 256))](
        down, scale, order, pad_row, out, T_CONST=T, H_CONST=H,
        stride_dm=down.stride(0), stride_dh=down.stride(1),
        stride_om=out.stride(0), stride_oh=out.stride(1), K_BRANCH=KTOP,
        NCHUNK=H // 256, BLOCK_T=32, BLOCK_H=256,
        num_warps=8, num_stages=1,
    )


def run_diagnostic():
    device = "cuda"
    order = torch.empty((M,), dtype=torch.int32, device=device)
    inv_order = torch.empty((M,), dtype=torch.int32, device=device)
    route_weight = torch.empty((M,), dtype=torch.float32, device=device)
    counts = torch.empty((E,), dtype=torch.int32, device=device)
    metadata_size = 560
    expert_ids = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    split_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_num = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    tile_cum = torch.empty((metadata_size,), dtype=torch.int32, device=device)
    num_tiles = torch.empty((1,), dtype=torch.int32, device=device)
    setup_metadata_kernel[(E,)](
        order, inv_order, route_weight, counts, expert_ids, split_cum,
        tile_num, tile_cum, num_tiles, M_CONST=M, BLOCK_R=4096,
        num_warps=8, num_stages=1,
    )

    act_src = torch.randn((M, I), dtype=torch.bfloat16, device=device)
    act = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=device)
    act_scale = torch.empty((M,), dtype=torch.float32, device=device)
    quantize_rows_kernel[(M,)](act_src, act, act_scale, ROWS=M, K=I,
                               BLOCK_K=I, num_warps=8, num_stages=1)
    del act_src
    apply_route_scale_kernel[(M,)](act_scale, route_weight, M_CONST=M,
                                    num_warps=1, num_stages=1)

    weight_src = torch.randn((E * H, I), dtype=torch.bfloat16, device=device)
    weight = torch.empty((E * H, I), dtype=torch.float8_e4m3fn, device=device)
    weight_scale = torch.empty((E * H,), dtype=torch.float32, device=device)
    quantize_rows_kernel[(E * H,)](weight_src, weight, weight_scale,
                                   ROWS=E * H, K=I, BLOCK_K=I,
                                   num_warps=8, num_stages=1)
    del weight_src

    a_desc = TensorDescriptor(act, act.shape, act.stride(), [BM, BK])
    b_desc = TensorDescriptor(weight, weight.shape, weight.stride(), [BN, BK])
    base = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=device)
    cand = torch.empty((PAD_ROWS, H), dtype=torch.float8_e4m3fn, device=device)
    base_scale = torch.empty((M, H // BN), dtype=torch.float32, device=device)
    cand_scale = torch.empty((M, H // BN), dtype=torch.float32, device=device)
    cand_desc = TensorDescriptor(cand, cand.shape, cand.stride(), [BM, BN])
    pad_row = torch.empty((M,), dtype=torch.int32, device=device)
    base_final = torch.empty((T, H), dtype=torch.bfloat16, device=device)
    cand_final = torch.empty((T, H), dtype=torch.bfloat16, device=device)

    def call_base_dn():
        return launch_baseline(a_desc, act_scale, b_desc, weight_scale, base,
                               base_scale, expert_ids, counts, split_cum,
                               tile_num, tile_cum, num_tiles)

    def call_cand_dn():
        return launch_candidate(a_desc, act_scale, b_desc, weight_scale, cand_desc,
                                cand, cand_scale, expert_ids, counts, split_cum,
                                tile_num, tile_cum, num_tiles)

    def call_base_combo():
        call_base_dn()
        return launch_gather(base, base_scale, inv_order, base_final)

    def call_cand_combo():
        build_pad_row_kernel[(PAD_ROWS // BM,)](
            pad_row, expert_ids, counts, split_cum, tile_num, tile_cum,
            num_tiles, BLOCK_M=BM, num_warps=4, num_stages=1)
        call_cand_dn()
        return launch_padded_gather(cand, cand_scale, inv_order, pad_row, cand_final)

    compiled_base = call_base_dn()
    compiled_map = build_pad_row_kernel[(PAD_ROWS // BM,)](
        pad_row, expert_ids, counts, split_cum, tile_num, tile_cum,
        num_tiles, BLOCK_M=BM, num_warps=4, num_stages=1)
    compiled_cand = call_cand_dn()
    compiled_gather = launch_gather(base, base_scale, inv_order, base_final)
    compiled_padded_gather = launch_padded_gather(
        cand, cand_scale, inv_order, pad_row, cand_final)

    mismatch = torch.zeros((3,), dtype=torch.int32, device=device)
    compare_mapped_fp8_kernel[(triton.cdiv(M * H, 1024),)](
        base, cand, pad_row, mismatch, M_CONST=M, H_CONST=H, BLOCK=1024,
        num_warps=4, num_stages=1)
    compare_kernel[(triton.cdiv(M * (H // BN), 1024),)](
        base_scale, cand_scale, mismatch, NUMEL=M * (H // BN), KIND=1,
        SLOT=1, BLOCK=1024, num_warps=4, num_stages=1)
    compare_kernel[(triton.cdiv(T * H, 1024),)](
        base_final, cand_final, mismatch, NUMEL=T * H, KIND=1, SLOT=2,
        BLOCK=1024, num_warps=4, num_stages=1)
    zero_status = torch.zeros((1,), dtype=torch.int32, device=device)
    validate_pad_row_kernel[(triton.cdiv(M, 1024),)](
        pad_row, zero_status, M_CONST=M, CAPACITY=PAD_ROWS, BLOCK_R=1024,
        num_warps=4, num_stages=1)
    zero_rows_check_kernel[(M // 4096,)](
        cand, cand_scale, pad_row, zero_status, H_CONST=H, NCHUNK=H // BN,
        BLOCK_H=256, num_warps=8, num_stages=1)
    mismatch_q = int(mismatch[0].item())
    mismatch_scale = int(mismatch[1].item())
    mismatch_final = int(mismatch[2].item())
    zero_code = int(zero_status[0].item())
    if mismatch_q != 0 or mismatch_scale != 0 or mismatch_final != 0 or zero_code != 0:
        print('P1MD {"kind":"fp8_c4_dn_tma_padded_confirm_custom","math_check_passed":false,'
              '"fp8_bit_mismatch":' + str(mismatch_q)
              + ',"scale_mismatch":' + str(mismatch_scale)
              + ',"bf16_final_mismatch":' + str(mismatch_final)
              + ',"zero_status":' + str(zero_code) + '}')
        raise RuntimeError("P1MD c4 padded DN TMA store exact check failed")

    groups = []
    for group in range(6):
        if group == 1 or group == 2 or group == 5:
            cand_dn_ms = float(triton.testing.do_bench(call_cand_dn, warmup=5, rep=25))
            base_dn_ms = float(triton.testing.do_bench(call_base_dn, warmup=5, rep=25))
            cand_combo_ms = float(triton.testing.do_bench(call_cand_combo, warmup=5, rep=25))
            base_combo_ms = float(triton.testing.do_bench(call_base_combo, warmup=5, rep=25))
            order_name = "BA"
        else:
            base_dn_ms = float(triton.testing.do_bench(call_base_dn, warmup=5, rep=25))
            cand_dn_ms = float(triton.testing.do_bench(call_cand_dn, warmup=5, rep=25))
            base_combo_ms = float(triton.testing.do_bench(call_base_combo, warmup=5, rep=25))
            cand_combo_ms = float(triton.testing.do_bench(call_cand_combo, warmup=5, rep=25))
            order_name = "AB"
        groups.append('{"group":' + str(group) + ',"order":"' + order_name
                      + '","baseline_dn_ms":' + str(base_dn_ms)
                      + ',"candidate_dn_ms":' + str(cand_dn_ms)
                      + ',"candidate_over_baseline_dn":' + str(cand_dn_ms / base_dn_ms)
                      + ',"baseline_dn_gather_ms":' + str(base_combo_ms)
                      + ',"candidate_rowmap_dn_padded_gather_ms":' + str(cand_combo_ms)
                      + ',"candidate_combo_over_baseline_dn_gather":'
                      + str(cand_combo_ms / base_combo_ms) + '}')

    report = ('{"kind":"fp8_c4_dn_tma_padded_confirm_custom","shape":{"T":16384,'
              '"H":2048,"E":32,"I":1024,"topk":4,"M":65536},'
              '"route":{"counts":"2016_2080_alternating","total_tiles":528,'
              '"all_experts_have_partial_BM128_tail":true,"route_weights":'
              '"positive_0.25_to_0.625_embedded_in_activation_scale",'
              '"zero_scale_rows":16},"geometry":{"grid":132,"BM":128,'
              '"BN":256,"BK":128,"GM":32,"warps":8,"stages":4,'
              '"flatten":true,"actual_tiles":528,"padded_capacity_rows":69632,'
              '"actual_padded_rows":67584},"change":'
              '"all_tiles_C_descriptor_store_to_padded_C8_plus_timed_GPU_PAD_ROW_and_padded_gather",'
              '"math_check":{"full_fp8_bitwise":true,"full_scale_exact":true,'
              '"full_final_bf16_exact":true,"zero_rows_exact":true,'
              '"pad_row_full_analytic_exact_unique_in_capacity":true,'
              '"fp8_bit_mismatch":0,"scale_mismatch":0,"bf16_final_mismatch":0},'
              '"resources":{"baseline":{"n_regs":' + str(compiled_base.n_regs)
              + ',"n_spills":' + str(compiled_base.n_spills)
              + ',"shared":' + str(compiled_base.metadata.shared)
              + '},"candidate":{"n_regs":' + str(compiled_cand.n_regs)
              + ',"n_spills":' + str(compiled_cand.n_spills)
              + ',"shared":' + str(compiled_cand.metadata.shared)
              + '},"rowmap":{"n_regs":' + str(compiled_map.n_regs)
              + ',"n_spills":' + str(compiled_map.n_spills)
              + ',"shared":' + str(compiled_map.metadata.shared)
              + '},"baseline_gather":{"n_regs":' + str(compiled_gather.n_regs)
              + ',"n_spills":' + str(compiled_gather.n_spills)
              + ',"shared":' + str(compiled_gather.metadata.shared)
              + '},"padded_gather":{"n_regs":' + str(compiled_padded_gather.n_regs)
              + ',"n_spills":' + str(compiled_padded_gather.n_spills)
              + ',"shared":' + str(compiled_padded_gather.metadata.shared)
              + '}},"timing":{"method":"do_bench_warmup5_rep25",'
              '"scope":"DN_only_diagnostic_and_main_baseline_DN_gather_vs_candidate_rowmap_DN_padded_gather",'
              '"groups":[' + ",".join(groups) + ']},"excluded":'
              '{"allocation":true,"random_generation":true,"quantization":true,'
              '"common_metadata":true,"descriptors":true,"math_checks":true,'
              '"candidate_rowmap":false},"limitations":'
              '"single_GPU_c4_padded_store_plus_rowmap_not_P1_end_to_end_DN_only_excludes_rowmap"}')
    print("P1MD " + report)
    raise RuntimeError("P1MD done")


def run_kernel(x):
    return x


run_diagnostic()
