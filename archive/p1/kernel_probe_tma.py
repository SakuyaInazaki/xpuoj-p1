import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist


def run_kernel(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    output,
    topk,
):
    names = [
        'make_tensor_descriptor',
        'load_tensor_descriptor',
        'descriptor_load',
        'descriptor_store',
        'make_tensor_descriptor_host',
        'tma_load',
        'tma_store',
    ]
    parts = []
    for name in names:
        parts.append(name + '=' + repr(getattr(tl, name, 'MISSING')))
    parts.append('triton_dist_tma=' + repr(getattr(triton_dist.language, 'tma', 'MISSING')))
    raise RuntimeError('TMA_PROBE ' + ' || '.join(parts))
