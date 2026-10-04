import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist

from triton_dist.kernels.nvidia.group_gemm import moe_grouped_gemm


def run_kernel(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, output, topk):
    _a = hidden_states[:4, :4].contiguous()
    _b = expert_gate_proj[:2, :4, :4].contiguous()
    _ids = torch.zeros(2, dtype=torch.int32, device=hidden_states.device)
    _counts = torch.ones(2, dtype=torch.int32, device=hidden_states.device)
    _split = torch.zeros(2, dtype=torch.int32, device=hidden_states.device)
    _tile = torch.ones(2, dtype=torch.int32, device=hidden_states.device)
    _cum = torch.zeros(2, dtype=torch.int32, device=hidden_states.device)
    _total = torch.ones(1, dtype=torch.int32, device=hidden_states.device)
    _out = torch.empty((4, 4), dtype=torch.bfloat16, device=hidden_states.device)
    moe_grouped_gemm(
        _a, _b, _ids, _counts, _split, _tile, _cum, _total,
        out=_out,
        input_reduce_last_dim=True,
        weight_reduce_last_dim=True,
    )
    raise RuntimeError("NO_TYPE_ERROR")
