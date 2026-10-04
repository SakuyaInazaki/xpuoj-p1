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


def _pm(expert_counts, num_experts, total_rows):
    device = expert_counts.device
    num_sms = 32
    M = int(total_rows)
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


def run_kernel(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
):
    device = hidden_states.device
    a = torch.empty((256, 8), dtype=torch.bfloat16, device=device)
    w = torch.empty((2, 4, 8), dtype=torch.bfloat16, device=device)
    counts = torch.zeros(2, dtype=torch.int32, device=device)
    counts += 128
    meta = _pm(counts, 2, 256)
    c = moe_grouped_gemm(
        a, w, meta[1], counts, meta[2], meta[3], meta[4], meta[5],
        input_reduce_last_dim=True, weight_reduce_last_dim=True,
        num_warps=8, num_stages=3, GROUP_SIZE_M=8, PERSISTENT=False,
    )
    raise RuntimeError('KW_OK shape=' + str(c.shape) + ' dtype=' + str(c.dtype))
