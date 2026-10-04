import torch
import torch.distributed as dist

def run_kernel(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, output, topk):
    raise RuntimeError(
        "SHAPES "
        + "T=" + str(hidden_states.shape[0])
        + " H=" + str(hidden_states.shape[1])
        + " E=" + str(gate_weight.shape[0])
        + " I=" + str(expert_gate_proj.shape[1])
        + " topk=" + str(topk)
        + " world=" + str(dist.get_world_size())
        + " rank=" + str(dist.get_rank())
        + " dtype=" + str(hidden_states.dtype)
    )
