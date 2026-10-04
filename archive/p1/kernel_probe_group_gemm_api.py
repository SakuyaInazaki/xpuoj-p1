import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist
from triton_dist.kernels.nvidia import group_gemm as gg


def run_kernel(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
):
    parts = []
    for name in dir(gg):
        if 'gemm' in name.lower() or 'block' in name.lower() or 'moe' in name.lower():
            parts.append(name + '=' + repr(getattr(gg, name)))
    raise RuntimeError('PROBE_API ' + ' || '.join(parts))
