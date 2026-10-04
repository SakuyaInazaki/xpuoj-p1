import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist

from triton_dist.kernels.nvidia.group_gemm import (
    moe_grouped_gemm,
    moe_grouped_gemm_2weights,
    build_block_row_idx_info_kernel,
    GROUP_GEMM_BLOCK_SIZE_M,
)


def run_kernel(hidden_states, gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, output, topk):
    _c = moe_grouped_gemm.__code__
    _d = moe_grouped_gemm.__defaults__
    _doc = str(moe_grouped_gemm.__doc__)
    _c2 = moe_grouped_gemm_2weights.__code__
    _d2 = moe_grouped_gemm_2weights.__defaults__
    _doc2 = str(moe_grouped_gemm_2weights.__doc__)
    raise RuntimeError(
        "PROBE_MOE_GG vars=" + str(_c.co_varnames) + " argc=" + str(_c.co_argcount) +
        " defaults=" + str(_d) + " doc=" + _doc[:1500] +
        " || PROBE_2W vars=" + str(_c2.co_varnames) + " argc=" + str(_c2.co_argcount) +
        " defaults=" + str(_d2) + " doc=" + _doc2[:1500]
    )
