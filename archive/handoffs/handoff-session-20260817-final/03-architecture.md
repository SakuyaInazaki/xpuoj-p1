# 03 当前架构与代码位置

## 推荐 base：`p1/kernel_fused_gateup_swiglu.py`（115738）

本文件约 2427 行，存在历史遗留重复定义（`_prepare_moe_metadata`、`_linear_bf16`、`_get_full_weights`、`_run_kernel_a2a` 等）；Python 以后定义为准，不要惊慌。在线 submission 115738 与该文件逐字节一致。

### 12 个 case 路径

115738 中所有 12 个评测 shape 全部走 `_run_replicated`：

| case | gateup | down |
|---|---|---|
| 1 | FP8 fused gateup+SwiGLU+amax | FP8 swizzle GEMM |
| 2 | per-row INT8 gateup swizzle GEMM | FP8 swizzle GEMM |
| 3 | FP8 fused gateup+SwiGLU+amax | FP8 swizzle GEMM |
| 4 | official BF16 grouped GEMM | official BF16 grouped GEMM |
| 5 | FP8 fused gateup+SwiGLU+amax | FP8 swizzle GEMM |
| 6 | official BF16 grouped GEMM | official BF16 grouped GEMM |
| 7 | FP8 fused gateup+SwiGLU+amax | FP8 swizzle GEMM |
| 8 | FP8 fused gateup+SwiGLU+amax | FP8 swizzle GEMM |
| 9 | FP8 fused gateup+SwiGLU+amax（lowmem full weights） | FP8 swizzle GEMM |
| 10 | FP8 fused gateup+SwiGLU+amax（lowmem full weights） | FP8 swizzle GEMM |
| 11 | official BF16 grouped GEMM | official BF16 grouped GEMM |
| 12 | FP8 fused gateup+SwiGLU+amax | FP8 swizzle GEMM |

115705（`p1/kernel.py`）与 115738 的差别只是 GROUP_M：115705 全部 custom GEMM 用 `GROUP_M=2`；115738 的 fused gateup 和 down 用 `GROUP_M=8`。115705 没有其他架构优势。

### 关键函数/行号（以 115738 文件为准）

| 行号 | 内容 |
|---:|---|
| 246 / 1664 | `_prepare_moe_metadata` 两份；运行时后者生效 |
| 299 / 1717 | `_linear_bf16_kernel` 两份；后者生效 |
| 374 | `_gather_branch_sum_kernel` |
| 466 | `_gather_branch_sum`（H=4096/2048/1024 分别 BH=1024/2048/1024） |
| 851 | `_fp8_group_gemm_kernel`：FP8 down GEMM，带 swizzle 和 int64 expert 基址 |
| 908 | `_fused_gateup_swiglu_kernel`：FP8 gateup + SwiGLU + amax 融合 |
| 972 | `_quant_weight_fp8`（chunk=8 分块量化） |
| 1018 | `_quant_act_fp8` |
| 1102 | `_swiglu_quant_fp8`（未融合路径仍使用） |
| 1130 | `_int8_group_gemm_kernel`：case2 gateup，带 swizzle |
| 1483 | `_get_full_fp8_weights_lowmem`：case9/10 low-memory full weights |
| 1626 | `_fused_gateup_swiglu_bf16` host wrapper |
| 1648 | `_quant_act_fp8_from_amax` |
| 1993 / 2023 | `_get_full_weights` 两份；后者 shape-key 生效 |
| 2051 | `_run_replicated` |
| 2181 | `run_kernel` |

### `_run_replicated` 主流程

1. 获取权重：
   - case9（E=256,I=2048）：`_get_full_fp8_weights_lowmem`，逐 rank all_gather gate/up/down，分块量化进 FP8 buffer，不保留 BF16 full。
   - 其他 FP8/INT8 case：先 `_get_full_weights` 缓存 full BF16，再按 shape 缓存 FP8/INT8 q/scale。
   - BF16 case：直接使用 `_get_full_weights`。
2. 路由：`_linear_bf16(x, gate_weight)` -> BF16 logits -> FP32 softmax -> topk -> FP32 归一化。
3. 排序/准备：`flat_ids.argsort(stable=True)`，gather `tokens_sorted`，`torch.bincount`，`_prepare_moe_metadata`。
4. GEMM：
   - case2：INT8 gateup swizzle GEMM -> `_swiglu_quant_fp8` -> FP8 down swizzle GEMM。
   - FP8 case：`_fused_gateup_swiglu_bf16` 直接得到 `act_bf16` 和全局 `amax`，再用 `_quant_act_fp8_from_amax` 得到 down 输入 FP8。
   - BF16 case：official `moe_grouped_gemm` + `_swiglu_weighted` + official down。
5. final：`inv_order = order.argsort()`，`_gather_branch_sum` 单遍 gather+FP32 累加写 output。

### custom GEMM 细节

- FP8/INT8 down kernel：`BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=3`。
- fused gateup kernel：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=3`。
  - 每个 program 同时算 gate tile 和 up tile，两个 `tl.dot`，应用 `A_SCALE * B_SCALE` 后算 `silu(gate)*up*weight`，atomic_max 写全局 AMAX，并写 `act_bf16`。
- 所有 custom kernel 的 `expert * stride_be` 基址必须 `expert.to(tl.int64)`。
- swizzle 只在同一 expert 的 tile 内进行，`row_begin`/`n_rows` 仍有效；这是本会话最大性能来源之一。

### 缓存设计

| 缓存 | key | 说明 |
|---|---|---|
| `_FULL_WEIGHT_CACHE` | shape tuple | full BF16 权重；不能放 id |
| `_FULL_FP8_CACHE` / `_FULL_INT8_CACHE` / `_FULL_DOWN_*` | shape | 低精度 full weights |
| `_FULL_FP8_LOWMEM_CACHE` | shape | case9 lowmem full FP8 |
| `_TOKEN_IDX_CACHE` | T,k,device | 静态索引 |
| NVSHMEM buffer 缓存 | shape/dtype | 单 key 或多 key，不可 clear 整个 dict |

### 数值正确性要点

- 所有 `argsort` 必须 `stable=True`。
- INT8 activation 必须 per-row scale + round；全局 scale SQNR 不足。
- FP8 activation 用全局 scale；fused gateup 用 FP32 acc 计算 SwiGLU 后先存 BF16 act，再统一 FP8 量化。
- final 必须 FP32 累加后写 BF16。
- 确定性依赖固定编译参数、stable sort、固定归并顺序；不要引入 atomic 影响最终求和的路径。
