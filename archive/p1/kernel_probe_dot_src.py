import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist
from triton_dist.kernels.nvidia.group_gemm import dot_k_const, dot_2parts_k_const


def run_kernel(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
):
    raise RuntimeError('DOT_ARGS ' + repr(dot_k_const.arg_names) + ' SIG ' + repr(dot_k_const.signature) + ' SRC ' + repr(dot_k_const.src))
