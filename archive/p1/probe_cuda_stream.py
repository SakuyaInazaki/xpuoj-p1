import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist


def run_kernel(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, output, topk):
    _s = torch.cuda.Stream(device=hidden_states.device)
    _e = torch.cuda.Event(enable_timing=True)
    _e.record(_s)
    raise RuntimeError("STREAM_OK " + str(_s) + " " + str(_e))
