# 03 当前架构（`p1/kernel.py` = 116882 / v159，约 3217 行）

文件仍有历史遗留重复定义（`_prepare_moe_metadata`、`_linear_bf16`、`_run_kernel_a2a`、
`_get_full_weights` 等）。Python 以后定义为准；实际 12 个评测 shape 全部走最后一份
`_run_replicated`（行 2821）和最后一份 `_linear_bf16`（行 2564）。

## 12 个 case 路径总表

| case | token gather | token FP8 | gateup | activation FP8 | down |
|---|---|---|---|---|---|
| 1 | order-derived row gather + per-row amax | per-row scale | fused orderW, persistent, row-A | row scale | TMA-B row-A |
| 2 | custom order-derived gather，无 amax | global `_quant_act_fp8` | plain FP8 grouped GEMM，BN256 | SwiGLU orderW，global scale | TMA-B global-A |
| 3,4,5,6,7,8,9,10,11,12 | order-derived row gather + per-row amax | per-row scale | fused orderW, persistent, row-A | row scale | TMA-B row-A |
| 9 权重准备 | `_get_full_fp8_weights_lowmem`（E=256/I=2048 防 OOM） | 同上 | 同上 | 同上 | 同上 |

没有 BF16 专家 GEMM 路径；case4/6/11 也是 FP8。

## 关键函数/行号（以后定义为准）

| 行号 | 内容 |
|---:|---|
| 375 | `_gather_branch_sum_kernel`（final gather+sum） |
| 467 | `_gather_branch_sum` host |
| 969 | `_fused_gateup_swiglu_kernel_rowA_persistent_tiles`（旧 sorted-W 版本，当前基本不用） |
| 1032 | `_fused_gateup_swiglu_kernel_rowA_persistent_tiles_orderW`（当前 fused gateup） |
| 1159 | `_fp8_group_gemm_kernel_persistent_tma_host`（case2 down，global A scale） |
| 1207 | `_fp8_group_gemm_kernel_persistent_tma_row`（其他 case down，row A scale） |
| 1255 | `_fp8_group_gemm_kernel_persistent`（case2 plain gateup） |
| 1308 | `_quant_weight_fp8`（per-expert/per-N weight scale） |
| 1422 | `_quant_act_fp8`（case2 token global quant） |
| 1487 | `_gather_tokens_row_amax_order_kernel`（非 case2 gather+row amax） |
| 1513 | `_gather_tokens_row_amax_order` host |
| 1531 | `_gather_tokens_from_order_kernel`（case2 token gather，无 amax） |
| 1555 | `_gather_tokens_from_order` host |
| 1587 | `_quant_act_fp8_with_amax` |
| 1631 | `_swiglu_amax_orderW_kernel`（case2 当前生效） |
| 1658 | `_swiglu_to_fp8_orderW_kernel`（case2 当前生效） |
| 1748 | `_swiglu_quant_fp8_orderW` host |
| 2104 | `_get_full_fp8_weights` |
| 2127 | `_get_full_fp8_weights_lowmem`（case9） |
| 2261 | `_fp8_group_gemm_pre_tma_host`（case2 down host） |
| 2280 | `_fp8_group_gemm_pre_tma_row`（非 case2 down host） |
| 2308 | `_fused_gateup_swiglu_rowA_persistent_tiles_orderW` host |
| 2428 | `_prepare_moe_metadata`（生效副本） |
| 2564 | `_linear_bf16`（生效副本） |
| 2821 | `_run_replicated` |
| 2971 | `run_kernel` |

## `_run_replicated` 主流程（当前 v159）

1. 权重：
   - case9（E=256,I=2048）：`_get_full_fp8_weights_lowmem`，逐 rank all_gather，分块量化 FP8，不保留 BF16 full。
   - 其他 FP8 case：`_get_full_weights` 缓存 full BF16，再 `_get_full_fp8_weights` 缓存 FP8。
2. 路由：`_linear_bf16(x, gate_weight)` -> BF16 logits -> FP32 softmax -> topk -> 归一化。
   - `topk_ids` 保持 int64，只 `reshape(-1)`，不再 int64->int32->int64。
   - E<=16 路由：BM64/BN16/BK128/w4/s3。
   - E=32：BM128/BN32/BK64/w4；E=64：BM128/BN64/BK64/w8；其他 BM128/BN128/BK64/w8。
3. 排序：
   - `order = flat_ids.argsort(stable=True)`。
   - case2：`_gather_tokens_from_order(x, order, k)`，直接 `src // k` 取源行，无 token_idx 中间张量。
   - 其他：`_gather_tokens_row_amax_order(x, order, k)`，同样 `src // k`，同时 per-row atomic amax。
4. token FP8：
   - case2：`_quant_act_fp8`（global amax + global quant）。
   - 其他：`_quant_act_fp8_row_from_amax`（per-row scale）。
5. metadata：`torch.bincount` -> `_prepare_moe_metadata`（num_sms=32）。
6. gateup：
   - case2：`_fp8_group_gemm_pre` plain persistent，再 `_swiglu_quant_fp8_orderW`。
   - 其他：`_fused_gateup_swiglu_rowA_persistent_tiles_orderW`。
7. down：
   - case2：`_fp8_group_gemm_pre_tma_host`（global A scale）。
   - 其他：`_fp8_group_gemm_pre_tma_row`（row A scale）。
8. final：`inv_order = order.argsort()` -> `_gather_branch_sum`。

## fused gateup orderW 细节（当前核心）

- host 行 2308，kernel 行 1032。
- 参数：grid=132，BM128/BN128/BK128/**GROUP_M=16**/w8/**num_stages=4**。
- 与旧 sorted-W 版本唯一核心区别：
  ```python
  src = tl.load(ORDER + offs_m, mask=row_mask, other=0)
  w = tl.load(W + src, mask=row_mask, other=0.0).to(tl.float32)
  ```
  因此 `_run_replicated` 对非 case2 不再执行 `flat_weights = flat_weights[order]`。
- 每 tile 仍 immediate `tl.atomic_max(AMAX + offs_m, row_max, mask=row_mask, sem="relaxed")`。

## case2 专属路径

- token gather：`_gather_tokens_from_order_kernel`（行 1531），BM128/BH128/w8/s1，无 amax。
- token quant：`_quant_act_fp8`（global）。
- gateup：`_fp8_group_gemm_kernel_persistent`，BM128/BN256/BK128/GM8/w8/s3。
- SwiGLU：`_swiglu_amax_orderW_kernel` + `_swiglu_to_fp8_orderW_kernel`，
  BM128/BN256/w8；两个 kernel 都按 `ORDER[offs_m]` 读原始 route weight。
- down：`_fp8_group_gemm_pre_tma_host`，global A scale。

## 所有 activation amax 的 relaxed atomic

当前活跃 amax 路径全部使用 `sem="relaxed"`：
- `_gather_tokens_row_amax_order_kernel`
- `_fused_gateup_swiglu_kernel_rowA_persistent_tiles_orderW`
- `_act_amax_fp8_kernel`（case2 token quant）
- `_swiglu_amax_orderW_kernel`（case2 SwiGLU）
- `_gather_tokens_amax_kernel`（旧路径）
这是 v129 的晋升改动，不要退回默认 `acq_rel`。

## TMA down GEMM 细节

- import：`from triton.tools.tensor_descriptor import TensorDescriptor`。
- host 每次创建 descriptor；本会话验证 descriptor 缓存无稳定收益，保持每次创建。
- B flatten `[G*N,K]`，descriptor block `[256,128]`；kernel `b = B_DESC.load([b_row, k*BLOCK_K])`，
  `acc = tl.dot(a, b.T, acc)`。
- 参数：persistent grid=132，BM128/BN256/BK128/GROUP_M=8/w8/s3。
- A 普通 `tl.load`；不要用 A TMA、A+B TMA、C TMA store（均更慢）。

## custom GEMM 细节

- 所有 custom kernel 的 `expert * stride_be` 基址必须 `expert.to(tl.int64)`。
- 当前所有 K/N 都恰好是 BLOCK 整数倍；custom GEMM 已删 K/N remainder mask，只保留最后一块 M 的 row mask。
- swizzle 只在同一 expert 内进行；`row_begin`/`n_rows` 语义不变。
- `_prepare_moe_metadata` 的 `split_size_cum` 实参是 `block_row_idx_to_row_offset`。
- FP8 down persistent launch 语法 `kernel[(132,)]`，不是 `kernel[((132,),)]`。

## 缓存设计

| 缓存 | key | 说明 |
|---|---|---|
| `_FULL_WEIGHT_CACHE` | shape tuple | full BF16 权重；不能放 id |
| `_FULL_FP8_CACHE` / `_FULL_INT8_CACHE` | shape | 低精度 full weights |
| `_FULL_FP8_LOWMEM_CACHE` | shape | case9 lowmem full FP8 |
| `_TOKEN_IDX_CACHE` | T,k,device | 现在 case2 不再使用；历史保留 |
| NVSHMEM buffer 缓存 | shape/dtype | 不可乱 clear 整个 dict |

## 数值正确性要点

- 所有 `argsort` 必须 `stable=True`（final inverse 的 `order.argsort()` 是排列求逆，键唯一，可不用 stable）。
- 所有 orderW kernel 读取的 `ORDER` 都是同一个 `order`（sorted position -> original flat index）。
- final 必须 FP32 累加后写 BF16。
- 确定性依赖固定编译参数、stable sort、固定归并顺序；不要引入 atomic sum。
- 不要用 side stream 做 `inv_order = order.argsort()`：116841 出现 DETERMINISM FAIL。
