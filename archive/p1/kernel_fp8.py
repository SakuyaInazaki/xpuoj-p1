import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist

from triton_dist.kernels.nvidia.group_gemm import (
    build_block_row_idx_info_kernel,
    GROUP_GEMM_BLOCK_SIZE_M,
)



@triton_dist.jit
def _fp8_group_gemm_kernel(
    A, A_SCALE, B, B_SCALE, C,
    expert_ids, split_size, split_size_cum, tile_num, tile_num_cum, num_tiles_total,
    M, N, K,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    total_tiles = tl.load(num_tiles_total)
    if pid >= total_tiles * num_block_n:
        return
    pid_m = pid // num_block_n
    pid_n = pid % num_block_n

    expert = tl.load(expert_ids + pid_m)
    n_rows = tl.load(split_size + expert)
    row_begin = tl.load(split_size_cum + pid_m)
    t_num = tl.load(tile_num + pid_m)
    t_cum = tl.load(tile_num_cum + pid_m)
    local_m = pid_m - (t_cum - t_num)

    offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_mask = offs_m < row_begin + n_rows
    col_mask = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        km = offs_k < rem
        a = tl.load(a_ptrs, mask=row_mask[:, None] & km[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=km[:, None] & col_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
    b_scale = tl.load(B_SCALE + expert * N + offs_n, mask=col_mask, other=1.0)
    acc = acc * a_scale[:, None] * b_scale[None, :]
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :])


def _prepare_moe_metadata(expert_counts, num_experts):
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
        expert_counts, split_size_cum_per_expert, block_row_idx_to_expert_idx,
        block_row_idx_to_row_offset, block_row_idx_to_tile_split,
        block_row_idx_to_tile_cumsum, expert_idx_to_tile_offset, num_tiles_total,
        num_experts, E_PAD, GROUP_GEMM_BLOCK_SIZE_M, num_sms)
    return (split_size_cum_per_expert, block_row_idx_to_expert_idx,
            block_row_idx_to_row_offset, block_row_idx_to_tile_split,
            block_row_idx_to_tile_cumsum, num_tiles_total)


def _quant_weight(w):
    amax = w.float().abs().amax(dim=2, keepdim=True)
    scale = torch.maximum(amax / 448.0, torch.full_like(amax, 1e-12))
    q = (w.float() / scale).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.squeeze(2).contiguous()


def _quant_act(a):
    amax = a.float().abs().amax(dim=1, keepdim=True)
    scale = torch.maximum(amax / 448.0, torch.full_like(amax, 1e-12))
    q = (a.float() / scale).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.squeeze(1).contiguous()


_STATIC = {}


def _get_static(gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, topk):
    key = (id(gate_weight), tuple(gate_weight.shape), id(expert_gate_proj), tuple(expert_gate_proj.shape),
           id(expert_up_proj), tuple(expert_up_proj.shape), id(expert_down_proj), tuple(expert_down_proj.shape), int(topk))
    c = _STATIC.get(key)
    if c is None:
        gate_up = torch.cat([expert_gate_proj, expert_up_proj], dim=1).contiguous()
        gu_q, gu_s = _quant_weight(gate_up)
        dn_q, dn_s = _quant_weight(expert_down_proj)
        del gate_up
        c = (gu_q, gu_s, dn_q, dn_s)
        _STATIC.clear()
        _STATIC[key] = c
    return c


@triton_dist.jit
def _linear_bf16_kernel(A, B, C, M, N, K, sam, sak, sbn, sbk, scm, scn,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = A + om[:, None] * sam + ok[None, :] * sak
    bp = B + ok[:, None] * sbk + on[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    rm = om < M
    cn = on < N
    for k in range(0, tl.cdiv(K, BK)):
        rem = K - k * BK
        km = ok < rem
        a = tl.load(ap, mask=rm[:, None] & km[None, :], other=0.0)
        b = tl.load(bp, mask=km[:, None] & cn[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        ap += BK * sak
        bp += BK * sbk
    cp = C + om[:, None] * scm + on[None, :] * scn
    tl.store(cp, acc.to(tl.bfloat16), mask=rm[:, None] & cn[None, :])


def _linear(a, b):
    M, K = a.shape
    N = b.shape[0]
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    _linear_bf16_kernel[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=128, BN=128, BK=64, num_warps=8, num_stages=3)
    return c


@triton_dist.jit
def _swiglu_kernel(G, U, W, C, M, N, sgm, sgn, sum_, sun, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    rm = om < M
    cn = on < N
    w = tl.load(W + om, mask=rm, other=0.0).to(tl.float32)
    g = tl.load(G + om[:, None] * sgm + on[None, :] * sgn, mask=rm[:, None] & cn[None, :], other=0.0).to(tl.float32)
    u = tl.load(U + om[:, None] * sum_ + on[None, :] * sun, mask=rm[:, None] & cn[None, :], other=0.0).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))) * u * w[:, None]
    tl.store(C + om[:, None] * scm + on[None, :] * scn, a.to(tl.bfloat16), mask=rm[:, None] & cn[None, :])


def _fp8_gemm(a, b_q, b_s, expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles):
    a_q, a_s = _quant_act(a)
    M, K = a.shape
    G, N, K2 = b_q.shape
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    nblocks = triton.cdiv(N, 128)
    grid = (triton.cdiv(M, 128) + G) * nblocks
    _fp8_group_gemm_kernel[(grid,)](
        a_q, a_s, b_q, b_s, c, expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M, N, K, a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1), b_q.stride(2),
        c.stride(0), c.stride(1), BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        num_warps=8, num_stages=3)
    return c


def run_kernel(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, output, topk):
    rank = dist.get_rank()
    world = dist.get_world_size()
    T, H = hidden_states.shape
    E = gate_weight.shape[0]
    Ep = expert_gate_proj.shape[0]
    I = expert_gate_proj.shape[1]
    k = int(topk)
    dev = hidden_states.device
    gu_q, gu_s, dn_q, dn_s = _get_static(gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, k)

    logits = _linear(hidden_states, gate_weight).float()
    probs = torch.softmax(logits, dim=-1)
    tw, ids = torch.topk(probs, k, dim=-1)
    s = tw.sum(dim=-1, keepdim=True)
    tw = tw / torch.maximum(s, torch.full_like(s, 1e-6))
    ids = ids.to(torch.int32)

    # Use the all_gather + local expert architecture for k > 4, and A2A for k <= 4.
    if k <= 4:
        # A2A path
        token_idx = torch.arange(T, dtype=torch.int64, device=dev).repeat_interleave(k)
        slot_idx = torch.arange(k, dtype=torch.int64, device=dev).repeat(T)
        fids = ids.reshape(-1).to(torch.int64)
        fw = tw.reshape(-1)
        dest = fids // Ep
        local = fids % Ep
        order = dest.argsort(stable=True)
        token_idx = token_idx[order]
        slot_idx = slot_idx[order]
        fids = fids[order]
        fw = fw[order]
        local = local[order]
        send_tokens = hidden_states[token_idx]
        send_meta = token_idx * k + slot_idx
        SHIFT = 1 << 20
        send_packed = local.to(torch.int64) * SHIFT + send_meta
        send_counts = torch.bincount(dest, minlength=world).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        scl = send_counts.tolist()
        rcl = recv_counts.tolist()
        tr = int(sum(rcl))
        recv_tokens = torch.empty((tr, H), dtype=torch.bfloat16, device=dev)
        recv_w = torch.empty((tr,), dtype=torch.float32, device=dev)
        recv_p = torch.empty((tr,), dtype=torch.int64, device=dev)
        dist.all_to_all_single(recv_tokens, send_tokens, output_split_sizes=rcl, input_split_sizes=scl)
        dist.all_to_all_single(recv_w, fw, output_split_sizes=rcl, input_split_sizes=scl)
        dist.all_to_all_single(recv_p, send_packed, output_split_sizes=rcl, input_split_sizes=scl)
        recv_meta = recv_p % SHIFT
        recv_local = (recv_p // SHIFT).to(torch.int32)
        order2 = recv_local.argsort(stable=True)
        tokens_sorted = recv_tokens[order2].contiguous()
        w_sorted = recv_w[order2]
        local_sorted = recv_local[order2].to(torch.int64)
        meta_sorted = recv_meta[order2]
        counts = torch.bincount(local_sorted, minlength=Ep).to(torch.int32)
        meta = _prepare_moe_metadata(counts, Ep)
        gu = _fp8_gemm(tokens_sorted, gu_q, gu_s, meta[1], counts, meta[2], meta[3], meta[4], meta[5])
        act = torch.empty((tr, I), dtype=torch.bfloat16, device=dev)
        _swiglu_kernel[(triton.cdiv(tr,128), triton.cdiv(I,128))](
            gu[:, :I], gu[:, I:], w_sorted, act, tr, I,
            gu.stride(0), gu.stride(1), gu.stride(0), gu.stride(1), act.stride(0), act.stride(1),
            BM=128, BN=128, num_warps=4, num_stages=2)
        down = _fp8_gemm(act, dn_q, dn_s, meta[1], counts, meta[2], meta[3], meta[4], meta[5])
        inv = order2.argsort()
        down_send = down[inv]
        meta_send = meta_sorted[inv]
        down_ret = torch.empty((T*k, H), dtype=torch.bfloat16, device=dev)
        meta_ret = torch.empty((T*k,), dtype=torch.int64, device=dev)
        dist.all_to_all_single(down_ret, down_send, output_split_sizes=scl, input_split_sizes=rcl)
        dist.all_to_all_single(meta_ret, meta_send, output_split_sizes=scl, input_split_sizes=rcl)
        src_t = meta_ret // k
        src_s = meta_ret % k
        branch = torch.zeros((T, k, H), dtype=torch.float32, device=dev)
        branch[src_t, src_s] = down_ret.float()
        output.copy_(branch.sum(dim=1).to(torch.bfloat16))
    else:
        # all_gather path
        hidden_all = torch.empty((world, T, H), dtype=torch.bfloat16, device=dev)
        ids_all = torch.empty((world, T, k), dtype=torch.int32, device=dev)
        w_all = torch.empty((world, T, k), dtype=torch.float32, device=dev)
        dist.all_gather_into_tensor(hidden_all, hidden_states)
        dist.all_gather_into_tensor(ids_all, ids)
        dist.all_gather_into_tensor(w_all, tw)
        parts_t = []
        parts_w = []
        parts_d = []
        parts_l = []
        for src in range(world):
            owner = ids_all[src] // Ep
            local = ids_all[src] % Ep
            sel = (owner == rank).nonzero(as_tuple=False)
            if sel.numel() == 0:
                continue
            rows = sel[:, 0].to(torch.int64)
            slots = sel[:, 1].to(torch.int64)
            loc = local[rows, slots].to(torch.int64)
            parts_t.append(hidden_all[src][rows])
            parts_w.append(w_all[src][rows, slots])
            parts_d.append(rows + src * T)
            parts_l.append(loc)
        partial = torch.zeros((world * T, H), dtype=torch.float32, device=dev)
        if parts_t:
            tokens_cat = torch.cat(parts_t, 0)
            w_cat = torch.cat(parts_w, 0)
            dst_cat = torch.cat(parts_d, 0)
            local_cat = torch.cat(parts_l, 0)
            order = local_cat.argsort(stable=True)
            tokens_cat = tokens_cat[order].contiguous()
            w_cat = w_cat[order]
            dst_cat = dst_cat[order]
            local_cat = local_cat[order]
            counts = torch.bincount(local_cat, minlength=Ep).to(torch.int32)
            meta = _prepare_moe_metadata(counts, Ep)
            M = int(tokens_cat.shape[0])
            gu = _fp8_gemm(tokens_cat, gu_q, gu_s, meta[1], counts, meta[2], meta[3], meta[4], meta[5])
            act = torch.empty((M, I), dtype=torch.bfloat16, device=dev)
            _swiglu_kernel[(triton.cdiv(M,128), triton.cdiv(I,128))](
                gu[:, :I], gu[:, I:], w_cat, act, M, I,
                gu.stride(0), gu.stride(1), gu.stride(0), gu.stride(1), act.stride(0), act.stride(1),
                BM=128, BN=128, num_warps=4, num_stages=2)
            down = _fp8_gemm(act, dn_q, dn_s, meta[1], counts, meta[2], meta[3], meta[4], meta[5])
            partial.index_add_(0, dst_cat, down.float())
        if T * H >= 20 * 1024 * 1024:
            combined = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
            dist.reduce_scatter_tensor(combined, partial.view(world*T,H).to(torch.bfloat16).contiguous(), op=dist.ReduceOp.SUM)
            output.copy_(combined)
        else:
            combined = torch.empty((T, H), dtype=torch.float32, device=dev)
            dist.reduce_scatter_tensor(combined, partial.view(world*T,H).contiguous(), op=dist.ReduceOp.SUM)
            output.copy_(combined.to(torch.bfloat16))
