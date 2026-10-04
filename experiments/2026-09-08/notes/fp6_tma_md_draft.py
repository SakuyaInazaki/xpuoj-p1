"""Standalone c9/c10 FP6-TMA MD draft; not wired into a submission."""

import torch
import triton
import triton.language as tl
import triton_dist

from triton.tools.tensor_descriptor import TensorDescriptor


@triton_dist.jit
def fp6_decode_tile(q4, q2, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Decode K-contiguous lo4/hi2 planes to an FP8 [N, K] tile."""
    lane2 = tl.arange(0, 2)
    lane4 = tl.arange(0, 4)
    lo4 = (tl.expand_dims(q4, 2) >> (lane2[None, None, :] * 4)) & 15
    hi2 = (tl.expand_dims(q2, 2) >> (lane4[None, None, :] * 2)) & 3
    lo4 = tl.reshape(lo4, (BLOCK_N, BLOCK_K))
    hi2 = tl.reshape(hi2, (BLOCK_N, BLOCK_K))
    code = lo4 | (hi2 << 4)

    ab = code & 31
    small = tl.where(
        ab == 0, 0,
        tl.where(
            ab == 1, 80,
            tl.where(
                ab == 2, 88,
                tl.where(
                    ab == 3, 92,
                    tl.where(
                        ab == 4, 96,
                        tl.where(ab == 5, 98, tl.where(ab == 6, 100, 102)),
                    ),
                ),
            ),
        ),
    )
    magnitude = tl.where(ab >= 8, ab + 96, small)
    sign = (code & 32) << 2
    raw = (magnitude | sign).to(tl.uint8)
    return raw.to(tl.float8e4nv, bitcast=True)


@triton_dist.jit
def fp6_tma_md_kernel(
    A, A_SCALE, BNORM, B4_DESC, B2_DESC, B_SCALE, W, ORDER, ACT, AMAX,
    expert_ids, split_size, split_size_cum, tile_num, tile_cum, num_tiles,
    M, I, K: tl.constexpr,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, KTOP: tl.constexpr,
):
    """Current c9/c10 call-3 MD math with packed FP6 descriptor loads."""
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I, BLOCK_N)
    total_tiles = tl.load(num_tiles)

    # Keep the current outer persistent-loop pipeline unchanged.
    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, num_stages=2):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n

        expert = tl.load(expert_ids + pid_m)
        n_rows = tl.load(split_size + expert)
        row_begin = tl.load(split_size_cum + pid_m)
        t_num = tl.load(tile_num + pid_m)
        t_cum = tl.load(tile_cum + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(
            local_m, pid_n, t_num, num_block_n, GROUP_M
        )

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
                a_ptrs, mask=row_mask[:, None], other=0.0,
                eviction_policy="evict_last",
            )
            q4g = B4_DESC.load([b_row_g, k * (BLOCK_K // 2)])
            q2g = B2_DESC.load([b_row_g, k * (BLOCK_K // 4)])
            q4u = B4_DESC.load([b_row_u, k * (BLOCK_K // 2)])
            q2u = B2_DESC.load([b_row_u, k * (BLOCK_K // 4)])
            bg = fp6_decode_tile(q4g, q2g, BLOCK_N, BLOCK_K)
            bu = fp6_decode_tile(q4u, q2u, BLOCK_N, BLOCK_K)
            acc_g = tl.dot(a, bg.T, acc_g)
            acc_u = tl.dot(a, bu.T, acc_u)
            a_ptrs += BLOCK_K * stride_ak

        a_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)
        g_scale = a_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I) + offs_n
        )[None, :]
        u_scale = a_scale[:, None] * tl.load(
            B_SCALE + expert * (2 * I) + I + offs_n
        )[None, :]
        g = acc_g * g_scale
        u = acc_u * u_scale
        src = tl.load(ORDER + offs_m, mask=row_mask, other=0)
        w = tl.load(W + src, mask=row_mask, other=0.0).to(tl.float32)
        silu = tl.fdiv(
            g, 1.0 + tl.exp2(-g * 1.4426950408889634), ieee_rounding=False
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
        tl.store(AMAX + offs_m, s, mask=row_mask)


def fp6_tma_md_host(
    a_q, a_s, b4, b2, b_s, bnorm, weights, order,
    expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
):
    """Launch the standalone fixed c9/c10 FP6 MD draft."""
    M = int(order.shape[0])
    K = int(a_q.shape[1])
    E, N2, Khalf = b4.shape
    I = N2 // 2
    assert b4.dtype == torch.uint8 and b2.dtype == torch.uint8
    assert b4.is_contiguous() and b2.is_contiguous()
    assert Khalf * 2 == K and tuple(b2.shape) == (E, N2, K // 4)
    assert E == 256 and K == 4096 and I in (2048, 1536)
    assert tuple(b_s.shape) == (E, N2) and int(bnorm.numel()) == E
    assert int(weights.numel()) == M
    assert M % int(a_q.shape[0]) == 0
    ktop = M // int(a_q.shape[0])
    assert ktop == 8

    b4_flat = b4.view(E * N2, K // 2)
    b2_flat = b2.view(E * N2, K // 4)
    b4_desc = TensorDescriptor(
        b4_flat, b4_flat.shape, b4_flat.stride(), [128, 64]
    )
    b2_desc = TensorDescriptor(
        b2_flat, b2_flat.shape, b2_flat.stride(), [128, 32]
    )
    act = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=a_q.device)
    rowscale = torch.empty(M, dtype=torch.float32, device=a_q.device)
    fp6_tma_md_kernel[(132,)](
        a_q, a_s, bnorm, b4_desc, b2_desc, b_s, weights, order, act, rowscale,
        expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M, I, K,
        a_q.stride(0), a_q.stride(1), act.stride(0), act.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=32, KTOP=ktop,
        num_warps=8, num_stages=3,
    )
    return act, rowscale
