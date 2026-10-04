# 03 当前架构（以 `p1/kernel.py` = 116310 / kernel_v98_token_row.py 为准）

文件约 2700 行，仍有历史遗留重复定义（`_prepare_moe_metadata`、`_linear_bf16`、
`_run_kernel_a2a`、`_get_full_weights` 等）。Python 以后定义为准，不要惊慌；
实际 12 个评测 shape 全部走最后一份 `_run_replicated`。

## 12 个 case 路径总表

| case | token 量化 | gateup | activation 量化 | down |
|---|---|---|---|---|
| 1,3,4,5,6,7,8,9,10,11,12 | gather + per-row amax；FP8 per-row scale | fused gateup+SwiGLU+per-row AMAX，row-A scale | FP8 per-row scale | TMA-B persistent，row-A scale |
| 2 | `x[token_idx].contiguous()`；global FP8 quant | plain FP8 grouped GEMM（regular persistent，K constexpr） | `_swiglu_quant_fp8` global scale | TMA-B persistent，global A scale |
| 9 权重准备 | `_get_full_fp8_weights_lowmem`（E=256/I=2048 防 OOM） | 同上 | 同上 | 同上 |

没有 BF16 专家 GEMM 路径；case4/6/11 也是 FP8。

## 关键函数/行号

| 行号 | 内容 |
|---:|---|
| 851 | `_fp8_group_gemm_kernel`（旧 non-persistent，当前评测路径基本不用） |
| 906 | `_fused_gateup_swiglu_kernel_rowA`（fused gateup 当前生效版本） |
| 968 | `_fused_gateup_swiglu_kernel`（旧 global-scale 版本，保留） |
| 1031 | `_fp8_group_gemm_kernel_persistent_tma_host`（case2 down 用，global A scale） |
| 1079 | `_fp8_group_gemm_kernel_persistent_tma_row`（其他 case down 用，row A scale） |
| 1127 | `_fp8_group_gemm_kernel_persistent`（case2 gateup regular） |
| 1181 | `_quant_weight_fp8`（per-expert/per-N weight scale） |
| 1217 | `_gather_tokens_amax_kernel`（旧 global amax gather） |
| 1242 | `_quant_bf16_to_fp8_kernel`（global quant） |
| 1267 | `_quant_bf16_to_fp8_row_kernel`（per-row quant） |
| 1318 | `_gather_tokens_row_amax_kernel`（gather + per-row amax，当前非 case2 使用） |
| 1343 | `_gather_tokens_row_amax` host |
| 1359 | `_gather_tokens_amax` host（旧 global，当前基本不用） |
| 1391 | `_swiglu_amax_kernel` / `_swiglu_to_fp8_kernel` / `_swiglu_quant_fp8`（case2 用） |
| 1587 | `_row_amax_kernel`（per-row amax，供 INT8/per-row 实验使用） |
| 1945 | `_fp8_group_gemm_pre`（case2 gateup regular，BM128/BN256/BK128/GM8/w8/s3） |
| 1963 | `_fp8_group_gemm_pre_tma_host`（case2 down，global A scale） |
| 1982 | `_fp8_group_gemm_pre_tma_row`（其他 down，row A scale） |
| 2033 | `_fused_gateup_swiglu_rowA` host（当前 fused gateup） |
| 2056 | `_quant_act_fp8_from_amax`（global） |
| 2072 | `_quant_act_fp8_row_from_amax`（per-row） |
| 2088 | `_prepare_moe_metadata`（生效副本） |
| 2447 | `_get_full_weights`（shape-key，生效副本） |
| 2475 | `_run_replicated` |
| 2623 | `run_kernel` |

## `_run_replicated` 主流程（当前）

1. 权重：
   - case9（E=256,I=2048）：`_get_full_fp8_weights_lowmem`，逐 rank all_gather，分块量化到 FP8，不保留 BF16 full。
   - 其他 FP8 case：`_get_full_weights` 缓存 full BF16，再 `_get_full_fp8_weights` 缓存 FP8 q/scale。
2. 路由：`_linear_bf16(x, gate_weight)` -> BF16 logits -> FP32 softmax -> topk -> 归一化。
3. 排序：
   - `flat_ids.argsort(stable=True)`；
   - `token_idx = token_idx[order]`、`flat_weights = flat_weights[order]`；
   - case2：`x[token_idx].contiguous()`，后续 global quant；
   - 其他 case：`_gather_tokens_row_amax`：gather 成 sorted BF16，同时
     `atomic_max(ROW_AMAX + offs_m, row_max)` 得到每个 sorted-row 的 amax。
4. token FP8：
   - case2：`_quant_act_fp8`（global amax + quant）；
   - 其他：`_quant_act_fp8_row_from_amax`（per-row scale）。
5. metadata：`torch.bincount` -> `_prepare_moe_metadata`。
6. gateup：
   - case2：`_fp8_group_gemm_pre` regular persistent -> `_swiglu_quant_fp8`；
   - 其他：`_fused_gateup_swiglu_rowA`：每个 program 两个 `tl.dot`，
     写 `act_bf16`，并对每行 `atomic_max(ACT_ROW_AMAX)`；
     之后 `_quant_act_fp8_row_from_amax`。
7. down：
   - case2：`_fp8_group_gemm_pre_tma_host`（global A scale）；
   - 其他：`_fp8_group_gemm_pre_tma_row`（row A scale）。
8. final：`inv_order = order.argsort()` -> `_gather_branch_sum` 单遍 gather+FP32 累加写 output。

## TMA down GEMM 细节（当前最重要改动）

- import：`from triton.tools.tensor_descriptor import TensorDescriptor`。
- 每个 down 调用在 host 创建：
  ```python
  b_flat = b_q.view(G * N, K)
  b_desc = TensorDescriptor(b_flat, b_flat.shape, b_flat.stride(), [256, 128])
  ```
  `b_q` 原本是 `[E,N,K]`，flatten 后第 `expert*N + n` 行就是该 expert 的第 n 行。
- kernel 中：
  ```python
  b_row = expert * N + pid_n * BLOCK_N
  b = B_DESC.load([b_row, k * BLOCK_K])   # shape [BN, BK] = [N-block, K-block]
  acc = tl.dot(a, b.T, acc)               # a 仍用普通 tl.load，[BM, BK]
  ```
- 参数：persistent grid=132，BM128/BN256/BK128/GROUP_M=8/w8/s3。
- A 保持普通 `tl.load`（试过 A TMA、A+B TMA、C TMA store 均变慢）。
- B 仍按原布局；不要转置成 `[G,K,N]`。

## per-row FP8 scale 细节

- 目标：避免单个 global `AMAX` 的 atomic 争用，同时数值上更接近 row-wise 量化。
- token gather：
  - `_gather_tokens_row_amax_kernel` 每 tile 计算 `row_max = max(abs(vals), axis=1)`，
    `tl.atomic_max(AMAX + offs_m, row_max)`。
- token quant：
  - `_quant_bf16_to_fp8_row_kernel` 加载 `INV_SCALE[offs_m]`，`q = (a * inv_scale[:, None]).to(float8e4nv)`。
- fused gateup：
  - `_fused_gateup_swiglu_kernel_rowA` 从 `A_SCALE[offs_m]` 取行 scale；
  - `g_scale = a_scale[:, None] * B_SCALE...`，`u_scale` 同理；
  - 输出 `act_bf16` 并对每行 `atomic_max(AMAX + offs_m, row_max, mask=row_mask)`。
- down：
  - `_fp8_group_gemm_kernel_persistent_tma_row` 最后
    `acc = acc * a_scale[:, None] * b_scale[None, :]`。

## custom GEMM 细节

- FP8 down persistent TMA/regular：BM128/BN256/BK128/GM8/w8/s3/grid132。
- fused gateup：BM128/BN128/BK128/GM8/w8/s3，非 persistent；
  每个 program 两个 dot（gate/up），epilogue 写 BF16 act + per-row AMAX。
- case2 plain gateup：regular persistent，BM128/BN256/BK128/GM8/w8/s3。
- 所有 custom kernel 的 `expert * stride_be` 基址必须 `expert.to(tl.int64)`。
- 当前所有 K/N 都恰好是 BLOCK 整数倍，custom GEMM 已删 K/N remainder mask，
  只保留最后一块 M 的 row mask。
- swizzle 只在同一 expert 内进行；`row_begin`/`n_rows` 语义不变。
- `_prepare_moe_metadata` 的 `split_size_cum` 实参是 `block_row_idx_to_row_offset`，
  不是 `rows_splits_cum_per_expert`。

## 缓存设计

| 缓存 | key | 说明 |
|---|---|---|
| `_FULL_WEIGHT_CACHE` | shape tuple | full BF16 权重；不能放 id |
| `_FULL_FP8_CACHE` / `_FULL_INT8_CACHE` | shape | 低精度 full weights |
| `_FULL_FP8_LOWMEM_CACHE` | shape | case9 lowmem full FP8 |
| `_TOKEN_IDX_CACHE` | T,k,device | 静态 token index |
| NVSHMEM buffer 缓存 | shape/dtype | 不可乱 clear 整个 dict |

## 数值正确性要点

- 所有 `argsort` 必须 `stable=True`。
- gather 行 amax 语义与 global 一致：sorted tokens 是原始 token 的重复，每行 max 相等。
- FP8 激活现在有 global（case2）和 per-row（其他 case）两套；修改时注意 down kernel
  必须使用配套的 `_tma_host` / `_tma_row`。
- final 必须 FP32 累加后写 BF16。
- 确定性依赖固定编译参数、stable sort、固定归并顺序；不要引入影响最终求和的 atomic。
