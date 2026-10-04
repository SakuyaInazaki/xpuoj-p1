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
    tools = getattr(triton, 'tools', 'MISSING')
    tdmod = getattr(tools, 'tensor_descriptor', 'MISSING') if tools != 'MISSING' else 'MISSING'
    cls = getattr(tdmod, 'TensorDescriptor', 'MISSING') if tdmod != 'MISSING' else 'MISSING'
    parts = [
        'tools=' + repr(tools),
        'tdmod=' + repr(tdmod),
        'TensorDescriptor=' + repr(cls),
    ]
    raise RuntimeError('TMA_HOST_PROBE ' + ' || '.join(parts))
