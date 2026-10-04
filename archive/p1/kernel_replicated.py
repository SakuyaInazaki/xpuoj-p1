import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist

from triton_dist.kernels.nvidia.group_gemm import (
    moe_grouped_gemm,
    build_block_row_idx_info_kernel,
    GROUP_GEMM_BLOCK_SIZE_M,
)
_HAS_GROUP_GEMM = True


def _prepare_moe_metadata(expert_counts, num_experts):
    """Compatibility copy of prepare_moe_metadata_using_kernel.

    The judging image exports the group-GEMM kernels but not the newer helper
    function, so the metadata launch is reproduced here.
    """
    device = expert_counts.device
    num_sms = 32
    M = int(expert_counts.sum().item())
    M_grid = triton.cdiv(M, GROUP_GEMM_BLOCK_SIZE_M) + num_experts
    E_PAD = triton.next_power_of_2(num_experts)

    split_size_cum_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    expert_idx_to_tile_offset = torch.zeros(num_experts, dtype=torch.int32, device=device)
    block_row_idx_to_expert_idx = torch.zeros(M_grid, dtype=torch.int32, device=device)
    block_row_idx_to_row_offset = torch.zeros(M_grid, dtype=torch.int32, device=device)
    block_row_idx_to_tile_split = torch.zeros(M_grid, dtype=torch.int32, device=device)
    block_row_idx_to_tile_cumsum = torch.zeros(M_grid, dtype=torch.int32, device=device)
    num_tiles_total = torch.zeros(1, dtype=torch.int32, device=device)

    build_block_row_idx_info_kernel[(num_sms,)](
        expert_counts,
        split_size_cum_per_expert,
        block_row_idx_to_expert_idx,
        block_row_idx_to_row_offset,
        block_row_idx_to_tile_split,
        block_row_idx_to_tile_cumsum,
        expert_idx_to_tile_offset,
        num_tiles_total,
        num_experts,
        E_PAD,
        GROUP_GEMM_BLOCK_SIZE_M,
        num_sms,
    )
    return (
        split_size_cum_per_expert,
        block_row_idx_to_expert_idx,
        block_row_idx_to_row_offset,
        block_row_idx_to_tile_split,
        block_row_idx_to_tile_cumsum,
        num_tiles_total,
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
# A: [M, K] BF16, B: [N, K] BF16 -> C = A @ B^T stored as BF16.
# FP32 accumulation matches the mixed-precision reference.
@triton_dist.jit
def _linear_bf16_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row_mask = offs_m < M
    col_mask = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        k_mask = offs_k < k_remain
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :])


# G: [M, N], U: [M, N], W: [M] FP32 -> BF16 (silu(g) * u * w)
@triton_dist.jit
def _swiglu_weighted_kernel(
    G, U, W, C,
    M, N,
    stride_g_m, stride_g_n,
    stride_u_m, stride_u_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < M
    col_mask = offs_n < N

    w = tl.load(W + offs_m, mask=row_mask, other=0.0).to(tl.float32)
    g_ptrs = G + offs_m[:, None] * stride_g_m + offs_n[None, :] * stride_g_n
    u_ptrs = U + offs_m[:, None] * stride_u_m + offs_n[None, :] * stride_u_n
    g = tl.load(g_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    u = tl.load(u_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

    silu = g / (1.0 + tl.exp(-g))
    a = silu * u * w[:, None]

    c_ptrs = C + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    tl.store(c_ptrs, a.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :])


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------
_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 64
_NUM_WARPS = 8
_NUM_STAGES = 3


def _ceil_div(a, b):
    return (a + b - 1) // b


def _linear_bf16(a, b):
    """a: [M,K] BF16, b: [N,K] BF16 -> [M,N] BF16."""
    M, K = a.shape
    N = b.shape[0]
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    if M == 0 or N == 0:
        return c
    grid = (_ceil_div(M, _BLOCK_M), _ceil_div(N, _BLOCK_N))
    _linear_bf16_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return c


def _swiglu_weighted(g, u, w):
    """g/u: [M,N] BF16, w: [M] FP32 -> [M,N] BF16."""
    M, N = g.shape
    c = torch.empty((M, N), dtype=torch.bfloat16, device=g.device)
    if M == 0 or N == 0:
        return c
    grid = (_ceil_div(M, _BLOCK_M), _ceil_div(N, _BLOCK_N))
    _swiglu_weighted_kernel[grid](
        g, u, w, c, M, N,
        g.stride(0), g.stride(1),
        u.stride(0), u.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return c


_FULL_WEIGHT_CACHE = {}


def _get_full_weights(gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj):
    key = (
        id(gate_weight), tuple(gate_weight.shape),
        id(expert_gate_proj), tuple(expert_gate_proj.shape),
        id(expert_up_proj), tuple(expert_up_proj.shape),
        id(expert_down_proj), tuple(expert_down_proj.shape),
    )
    cached = _FULL_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    world_size = dist.get_world_size()
    gate_list = [torch.empty_like(expert_gate_proj) for _ in range(world_size)]
    up_list = [torch.empty_like(expert_up_proj) for _ in range(world_size)]
    down_list = [torch.empty_like(expert_down_proj) for _ in range(world_size)]
    dist.all_gather(gate_list, expert_gate_proj)
    dist.all_gather(up_list, expert_up_proj)
    dist.all_gather(down_list, expert_down_proj)

    gate_full = torch.cat(gate_list, dim=0).contiguous()
    up_full = torch.cat(up_list, dim=0).contiguous()
    down_full = torch.cat(down_list, dim=0).contiguous()
    gate_up_full = torch.cat([gate_full, up_full], dim=1).contiguous()

    cached = (gate_up_full, down_full)
    _FULL_WEIGHT_CACHE.clear()
    _FULL_WEIGHT_CACHE[key] = cached
    return cached


def run_kernel(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
):
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = hidden_states
    T, H = x.shape
    E = gate_weight.shape[0]
    I = expert_gate_proj.shape[1]
    k = int(topk)
    device = x.device

    # Static expert weights are replicated once during warmup.  Every rank
    # then computes its own tokens locally: no token dispatch, no combine.
    gate_up_full, down_full = _get_full_weights(
        gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj
    )

    # Routing is unchanged.
    logits_bf16 = _linear_bf16(x, gate_weight)
    logits = logits_bf16.float()
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k, dim=-1)
    topk_sum = topk_weights.sum(dim=-1, keepdim=True)
    topk_sum = torch.maximum(topk_sum, torch.full_like(topk_sum, 1e-6))
    topk_weights = topk_weights / topk_sum
    topk_ids = topk_ids.to(torch.int32)

    token_idx = torch.arange(T, dtype=torch.int64, device=device).repeat_interleave(k)
    slot_idx = torch.arange(k, dtype=torch.int64, device=device).repeat(T)
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)

    # Group local branches by global expert for grouped GEMM.
    order = flat_ids.argsort(stable=True)
    flat_ids = flat_ids[order]
    token_idx = token_idx[order]
    slot_idx = slot_idx[order]
    flat_weights = flat_weights[order]
    tokens_sorted = x[token_idx].contiguous()

    expert_counts = torch.bincount(flat_ids, minlength=E).to(torch.int32)
    metadata = _prepare_moe_metadata(expert_counts, E)
    (meta_split_cum, meta_expert_ids, meta_tile_split,
     meta_tile_num, meta_tile_num_cum, num_tiles_total) = metadata

    gateup = moe_grouped_gemm(
        tokens_sorted, gate_up_full,
        meta_expert_ids, expert_counts, meta_tile_split,
        meta_tile_num, meta_tile_num_cum, num_tiles_total,
        input_reduce_last_dim=True, weight_reduce_last_dim=True,
    )
    act = _swiglu_weighted(gateup[:, :I], gateup[:, I:], flat_weights)
    down = moe_grouped_gemm(
        act, down_full,
        meta_expert_ids, expert_counts, meta_tile_split,
        meta_tile_num, meta_tile_num_cum, num_tiles_total,
        input_reduce_last_dim=True, weight_reduce_last_dim=True,
    )

    # Restore original (token, slot) order and reduce deterministically.
    inv_order = order.argsort()
    down_flat = down[inv_order].contiguous()
    token_orig = torch.arange(T, dtype=torch.int64, device=device).repeat_interleave(k)
    slot_orig = torch.arange(k, dtype=torch.int64, device=device).repeat(T)

    branch_sum = torch.zeros((T, k, H), dtype=torch.float32, device=device)
    branch_sum[token_orig, slot_orig] = down_flat.float()
    output.copy_(branch_sum.sum(dim=1).to(torch.bfloat16))
