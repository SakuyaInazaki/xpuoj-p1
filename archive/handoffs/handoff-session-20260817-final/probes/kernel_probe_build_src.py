import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist
from triton_dist.kernels.nvidia.group_gemm import build_block_row_idx_info_kernel


def run_kernel(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
):
    k = build_block_row_idx_info_kernel
    raise RuntimeError('BUILD_ARGS ' + repr(k.arg_names) + ' SIG ' + repr(k.signature) + ' SRC ' + repr(k.src))
