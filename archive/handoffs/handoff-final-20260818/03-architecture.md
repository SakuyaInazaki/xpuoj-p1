# 03 当前架构与代码位置

## 推荐 base：`p1/kernel_115907_backup.py`（115907）

约 2559 行，存在历史遗留重复定义（`_prepare_moe_metadata`、`_linear_bf16`、
`_get_full_weights` 等）；Python 以后定义为准，不要惊慌。

当前 `p1/kernel.py` = 115950，比 115907 只多一个 case2 gather 变体：
- 115950：case2 用 `_gather_tokens_amax_bm256`（BM256/H128, num_warps=4）；
- 115907：case2 用 `x[token_idx].contiguous()` + `_quant_act_fp8`。

### 12 个 case 路径

115907 / 115950 中所有 12 个评测 shape 全部走 `_run_replicated`：

| case | gateup | down |
|---|---|---|
| 1 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 2 | **plain FP8 GEMM** -> SwiGLU quant | FP8 persistent down |
| 3 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 4 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 5 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 6 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 7 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 8 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 9 | FP8 fused gateup+SwiGLU+amax（lowmem full weights） | FP8 persistent down |
| 10 | FP8 fused gateup+SwiGLU+amax（lowmem full weights） | FP8 persistent down |
| 11 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |
| 12 | FP8 fused gateup+SwiGLU+amax | FP8 persistent down |

没有 BF16 专家 GEMM 路径；case4/6/11 也已经是 FP8。

### 关键函数/行号（以当前 `p1/kernel.py` 为准）

| 行号 | 内容 |
|---:|---|
| 246 / 1799 | `_prepare_moe_metadata` 两份；后者生效 |
| 374 | `_gather_branch_sum_kernel` |
| 466 | `_gather_branch_sum` |
| 906 | `_fused_gateup_swiglu_kernel` |
| 969 | `_fp8_group_gemm_kernel_persistent` |
| 1022 | `_quant_weight_fp8` |
| 1059 | `_gather_tokens_amax_kernel`（gather+amax） |
| 1108 | `_quant_act_fp8` |
| 1131 | `_gather_tokens_amax_bm256`（仅 115950 case2 用） |
| 1147 | `_gather_tokens_amax` |
| 1163 | `_quant_act_fp8_with_amax` |
| 1238 | `_swiglu_quant_fp8`（case2） |
| 1733 | `_fp8_group_gemm_pre` |
| 1760 | `_fused_gateup_swiglu_bf16` host |
| 1783 | `_quant_act_fp8_from_amax` |
| 2186 | `_run_replicated` |
| 2328 | `run_kernel` |

### `_run_replicated` 主流程

1. 获取权重：
   - case9（E=256,I=2048）：`_get_full_fp8_weights_lowmem`，逐 rank all_gather，
     分块量化进 FP8 buffer，不保留 BF16 full。
   - 其他 FP8 case：先 `_get_full_weights` 缓存 full BF16，再按 shape 缓存 FP8 q/scale。
   - case2 也走 FP8 缓存（gateup INT8 已废弃）。
2. 路由：`_linear_bf16(x, gate_weight)` -> BF16 logits -> FP32 softmax -> topk -> 归一化。
3. 排序：
   - `flat_ids.argsort(stable=True)`；
   - `token_idx = token_idx[order]`；
   - 115907：case2 用 `x[token_idx].contiguous()`，其他 case 用 `_gather_tokens_amax`；
   - 115950：case2 用 `_gather_tokens_amax_bm256`，其他同 115907。
   - `torch.bincount` -> `_prepare_moe_metadata`。
4. activation 量化：
   - gather+amax 路径：gather 时同时 atomic_max 得到全局 amax，
     再 `_quant_act_fp8_with_amax` 得到 FP8 A；
   - case2 115907 路径：`_quant_act_fp8` 在 sorted tokens 上单算 amax + quant。
5. gateup：
   - case2：`_fp8_group_gemm_pre`（plain）-> `_swiglu_quant_fp8` -> FP8 down；
   - 其他：`_fused_gateup_swiglu_bf16` 直接得到 act BF16 和全局 amax，
     再 `_quant_act_fp8_from_amax` -> FP8 down。
6. final：`inv_order = order.argsort()` -> `_gather_branch_sum` 单遍 gather+FP32 累加写 output。

### custom GEMM 细节

- FP8 down persistent kernel：`BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=3`，grid=`(132,)`。
- fused gateup kernel：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=3`，非 persistent。
  - 每个 program 同时算 gate tile 和 up tile，两个 `tl.dot`，
    应用 `A_SCALE * B_SCALE` 后算 `silu(gate)*up*weight`，
    `atomic_max` 写全局 AMAX，并写 `act_bf16`。
- 所有 custom kernel 的 `expert * stride_be` 基址必须 `expert.to(tl.int64)`。
- 当前所有 K/N 维度都恰好是 BLOCK_K/BLOCK_N 整数倍，因此 custom GEMM 已删掉
  K-remainder mask 和 N-column mask；只保留最后一块 M 的 row mask。
- swizzle 只在同一 expert 的 tile 内进行，`row_begin`/`n_rows` 语义不变。

### 缓存设计

| 缓存 | key | 说明 |
|---|---|---|
| `_FULL_WEIGHT_CACHE` | shape tuple | full BF16 权重；不能放 id |
| `_FULL_FP8_CACHE` / `_FULL_INT8_CACHE` | shape | 低精度 full weights |
| `_FULL_FP8_LOWMEM_CACHE` | shape | case9 lowmem full FP8 |
| `_TOKEN_IDX_CACHE` | T,k,device | 静态 token index |
| NVSHMEM buffer 缓存 | shape/dtype | 不可乱 clear 整个 dict |

### 数值正确性要点

- 所有 `argsort` 必须 `stable=True`。
- FP8 activation 用全局 scale；fused gateup 用 FP32 acc 计算 SwiGLU 后先存 BF16 act，
  再统一 FP8 量化。
- gather+amax 的全局 max 与原 `_quant_act_fp8` 语义一致：
  sorted tokens 只是原始 token 的重复，因此 gather 时对重复行求 max 等于对 x 求 max。
- final 必须 FP32 累加后写 BF16。
- 确定性依赖固定编译参数、stable sort、固定归并顺序；不要引入 atomic 影响最终求和的路径。
