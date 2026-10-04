import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist

from triton_dist.kernels.nvidia.group_gemm import (
    moe_grouped_gemm,
    moe_grouped_gemm_2weights,
    build_block_row_idx_info_kernel,
    GROUP_GEMM_BLOCK_SIZE_M,
)


def run_kernel(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, output, topk):
    raise RuntimeError(
        "PROBE_DIR gg=" + str(dir(moe_grouped_gemm)) + " || 2w=" + str(dir(moe_grouped_gemm_2weights))
    )
