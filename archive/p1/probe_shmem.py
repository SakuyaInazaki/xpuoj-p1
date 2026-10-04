import torch
import torch.distributed as dist
from triton_dist.utils import is_shmem_initialized


def run_kernel(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, output, topk):
    raise RuntimeError(
        "PROBE shmem_init=" + str(is_shmem_initialized())
        + " rank=" + str(dist.get_rank())
        + " world=" + str(dist.get_world_size())
    )
