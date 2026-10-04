import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist
from triton_dist.kernels.nvidia.group_gemm import moe_grouped_gemm_kernel_nk_const


def run_kernel(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
):
    moe_grouped_gemm_kernel_nk_const[(1,)]()
