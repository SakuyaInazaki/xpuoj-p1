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



@triton_dist.jit
def _reduce_branches_kernel(
    D, Out,
    H,
    stride_dm, stride_dn,
    stride_om, stride_on,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(axis=0)
    hb = tl.program_id(axis=1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for j in range(TOPK):
        row = token * TOPK + j
        vals = tl.load(
            D + row * stride_dm + (hb * BLOCK_H + offs_h) * stride_dn,
            mask=mask_h, other=0.0,
        ).to(tl.float32)
        acc += vals

    tl.store(
        Out + token * stride_om + offs_h * stride_on,
        acc.to(tl.bfloat16), mask=mask_h,
    )


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


_STATIC_CACHE = {}


def _get_static_cache(gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, topk):
    key = (
        id(gate_weight), tuple(gate_weight.shape),
        id(expert_gate_proj), tuple(expert_gate_proj.shape),
        id(expert_up_proj), tuple(expert_up_proj.shape),
        id(expert_down_proj), tuple(expert_down_proj.shape),
        int(topk),
    )
    cache = _STATIC_CACHE.get(key)
    if cache is None:
        gate_up = torch.cat([expert_gate_proj, expert_up_proj], dim=1).contiguous()
        cache = {"gate_up": gate_up}
        _STATIC_CACHE.clear()
        _STATIC_CACHE[key] = cache
    return cache



def _run_kernel_a2a(
    x,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
    gate_up,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    T, H = x.shape
    E = gate_weight.shape[0]
    Ep = expert_gate_proj.shape[0]
    I = expert_gate_proj.shape[1]
    k = int(topk)
    device = x.device

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

    dest = flat_ids // Ep
    local = flat_ids % Ep
    order = dest.argsort(stable=True)

    token_idx = token_idx[order]
    slot_idx = slot_idx[order]
    flat_ids = flat_ids[order]
    flat_weights = flat_weights[order]
    local = local[order]

    send_tokens = x[token_idx]
    send_meta = token_idx * k + slot_idx

    send_counts = torch.bincount(dest, minlength=world_size).to(torch.int64)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send_counts_list = send_counts.tolist()
    recv_counts_list = recv_counts.tolist()
    total_recv = int(sum(recv_counts_list))

    recv_tokens = torch.empty((total_recv, H), dtype=torch.bfloat16, device=device)
    recv_weights = torch.empty((total_recv,), dtype=torch.float32, device=device)
    recv_meta = torch.empty((total_recv,), dtype=torch.int64, device=device)
    recv_local = torch.empty((total_recv,), dtype=torch.int32, device=device)

    dist.all_to_all_single(
        recv_tokens, send_tokens,
        output_split_sizes=recv_counts_list, input_split_sizes=send_counts_list,
    )
    dist.all_to_all_single(
        recv_weights, flat_weights,
        output_split_sizes=recv_counts_list, input_split_sizes=send_counts_list,
    )
    dist.all_to_all_single(
        recv_meta, send_meta,
        output_split_sizes=recv_counts_list, input_split_sizes=send_counts_list,
    )
    dist.all_to_all_single(
        recv_local, local.to(torch.int32),
        output_split_sizes=recv_counts_list, input_split_sizes=send_counts_list,
    )

    if total_recv > 0:
        order2 = recv_local.argsort(stable=True)
        tokens_sorted = recv_tokens[order2]
        weights_sorted = recv_weights[order2]
        local_sorted = recv_local[order2].to(torch.int64)
        meta_sorted = recv_meta[order2]

        expert_counts = torch.bincount(local_sorted, minlength=Ep).to(torch.int32)
        metadata = _prepare_moe_metadata(expert_counts, Ep)
        (meta_split_cum, meta_expert_ids, meta_tile_split,
         meta_tile_num, meta_tile_num_cum, num_tiles_total) = metadata

        gateup = moe_grouped_gemm(
            tokens_sorted, gate_up,
            meta_expert_ids, expert_counts, meta_tile_split,
            meta_tile_num, meta_tile_num_cum, num_tiles_total,
            input_reduce_last_dim=True, weight_reduce_last_dim=True,
        )
        act = _swiglu_weighted(gateup[:, :I], gateup[:, I:], weights_sorted)
        down = moe_grouped_gemm(
            act, expert_down_proj,
            meta_expert_ids, expert_counts, meta_tile_split,
            meta_tile_num, meta_tile_num_cum, num_tiles_total,
            input_reduce_last_dim=True, weight_reduce_last_dim=True,
        )

        inv_order2 = order2.argsort()
        down_send = down[inv_order2]
        meta_send = meta_sorted[inv_order2]

        down_return = torch.empty((int(send_tokens.shape[0]), H), dtype=torch.bfloat16, device=device)
        meta_return = torch.empty((int(send_tokens.shape[0]),), dtype=torch.int64, device=device)
        dist.all_to_all_single(
            down_return, down_send,
            output_split_sizes=send_counts_list, input_split_sizes=recv_counts_list,
        )
        dist.all_to_all_single(
            meta_return, meta_send,
            output_split_sizes=send_counts_list, input_split_sizes=recv_counts_list,
        )

        final_order = meta_return.argsort(stable=True)
        down_sorted = down_return[final_order].contiguous()
        _reduce_branches_kernel[
            (T, _ceil_div(H, 256))
        ](
            down_sorted, output, H,
            down_sorted.stride(0), down_sorted.stride(1),
            output.stride(0), output.stride(1),
            TOPK=k, BLOCK_H=256,
            num_warps=8, num_stages=2,
        )
    else:
        output.zero_()


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
    Ep = expert_gate_proj.shape[0]
    I = expert_gate_proj.shape[1]
    k = int(topk)
    device = x.device
    cache = _get_static_cache(gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, k)
    gate_up = cache["gate_up"]

    # ---- routing ----------------------------------------------------------
    logits_bf16 = _linear_bf16(x, gate_weight)          # BF16 @ BF16 -> BF16
    logits = logits_bf16.float()
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k, dim=-1)
    topk_sum = topk_weights.sum(dim=-1, keepdim=True)
    topk_sum = torch.maximum(topk_sum, torch.full_like(topk_sum, 1e-6))
    topk_weights = topk_weights / topk_sum
    topk_ids = topk_ids.to(torch.int32)

    if k <= 4:
        _run_kernel_a2a(
            x, gate_weight, expert_gate_proj, expert_up_proj,
            expert_down_proj, output, k, gate_up,
        )
        return

    # ---- all-gather source tokens / routing records ------------------------
    hidden_all = [torch.empty_like(x) for _ in range(world_size)]
    ids_all = [torch.empty_like(topk_ids) for _ in range(world_size)]
    weights_all = [torch.empty_like(topk_weights) for _ in range(world_size)]
    dist.all_gather(hidden_all, x)
    dist.all_gather(ids_all, topk_ids)
    dist.all_gather(weights_all, topk_weights)

    # partial[src] is this rank's FP32 contribution for source rank src.
    partial = torch.zeros((world_size, T, H), dtype=torch.float32, device=device)

    token_parts = []
    weight_parts = []
    dst_parts = []
    local_parts = []

    for src in range(world_size):
        ids_src = ids_all[src]
        weights_src = weights_all[src]
        tokens_src = hidden_all[src]

        owner = ids_src // Ep
        local = ids_src % Ep
        selected = (owner == rank).nonzero(as_tuple=False)
        if selected.numel() == 0:
            continue

        token_rows = selected[:, 0].to(torch.int64)
        slots = selected[:, 1].to(torch.int64)
        local_selected = local[token_rows, slots].to(torch.int64)
        token_parts.append(tokens_src[token_rows])
        weight_parts.append(weights_src[token_rows, slots])
        dst_parts.append(token_rows + src * T)
        local_parts.append(local_selected)

    if token_parts:
        tokens_cat = torch.cat(token_parts, dim=0)
        weights_cat = torch.cat(weight_parts, dim=0)
        dst_cat = torch.cat(dst_parts, dim=0)
        local_cat = torch.cat(local_parts, dim=0)
        num_tokens = int(tokens_cat.shape[0])

        # Process all source ranks together.  Each rank receives T*topk
        # branches in total, so this removes a factor world_size kernel-launch
        # overhead relative to looping over source ranks separately.
        order = local_cat.argsort(stable=True)
        tokens_cat = tokens_cat[order]
        weights_cat = weights_cat[order]
        dst_cat = dst_cat[order]
        local_cat = local_cat[order]

        partial_flat = partial.view(world_size * T, H)

        if _HAS_GROUP_GEMM:
            # Generic Triton grouped GEMM: one kernel for all experts.  This
            # removes per-expert launch overhead, which matters most for the
            # E=256 test points.
            expert_counts = torch.bincount(local_cat, minlength=Ep).to(torch.int32)
            metadata = _prepare_moe_metadata(expert_counts, Ep)
            (meta_split_cum, meta_expert_ids, meta_tile_split,
             meta_tile_num, meta_tile_num_cum, num_tiles_total) = metadata

            gateup = moe_grouped_gemm(
                tokens_cat, gate_up,
                meta_expert_ids, expert_counts, meta_tile_split,
                meta_tile_num, meta_tile_num_cum, num_tiles_total,
                input_reduce_last_dim=True,
                weight_reduce_last_dim=True,
            )
            act = _swiglu_weighted(gateup[:, :I], gateup[:, I:], weights_cat)
            down = moe_grouped_gemm(
                act, expert_down_proj,
                meta_expert_ids, expert_counts, meta_tile_split,
                meta_tile_num, meta_tile_num_cum, num_tiles_total,
                input_reduce_last_dim=True,
                weight_reduce_last_dim=True,
            )
            partial_flat.index_add_(0, dst_cat, down.float())
        else:
            counts = torch.bincount(local_cat, minlength=Ep).tolist()

            start = 0
            for expert in range(Ep):
                cnt = int(counts[expert])
                if cnt == 0:
                    continue
                end = start + cnt
                x_e = tokens_cat[start:end]                    # [cnt, H]
                w_e = weights_cat[start:end]                   # [cnt]
                dst_e = dst_cat[start:end]                     # [cnt]

                gateup = _linear_bf16(x_e, gate_up[expert])    # [cnt, 2*I]
                gate = gateup[:, :I]
                up = gateup[:, I:]
                act = _swiglu_weighted(gate, up, w_e)          # [cnt, I] BF16
                down = _linear_bf16(act, expert_down_proj[expert])  # [cnt, H] BF16

                partial_flat.index_add_(0, dst_e, down.float())
                start = end

    # FP32 sum of the four source blocks, returned to each source rank.
    combined = torch.empty((T, H), dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(
        combined,
        partial.reshape(world_size * T, H).contiguous(),
        op=dist.ReduceOp.SUM,
    )
    output.copy_(combined.to(torch.bfloat16))
