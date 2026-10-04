# 03 当前架构（`p1/kernel.py` = 117300 / v233，约 3262 行）

文件仍有历史遗留重复定义（`_linear_bf16`、`_prepare_moe_metadata`、`_run_kernel_a2a` 等）。
Python 以后定义为准；12 个评测 shape 全部走最后一份 `_run_replicated`（行 2866）
和最后一份 `_linear_bf16`（行 2601）。

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
| 247 | `_prepare_moe_metadata`（历史副本，基本不用） |
| 404 | `_gather_branch_sum_kernel_tiled`（final 2D tiling） |
| 449 | `_linear_bf16`（历史副本） |
| 500 | `_gather_branch_sum`（final host，当前生效） |
| 1069 | `_fused_gateup_swiglu_kernel_rowA_persistent_tiles_orderW`（当前 fused gateup kernel） |
| 1292 | `_fp8_group_gemm_kernel_persistent`（case2 plain gateup） |
| 1364 | `_act_amax_fp8_kernel`（case2 token global amax） |
| 1459 | `_quant_act_fp8`（case2 token quant，当前 BM128/BK64） |
| 1524 | `_gather_tokens_row_amax_order_kernel`（非 case2 gather+row amax） |
| 1550 | `_gather_tokens_row_amax_order` host（BM128/BH128） |
| 1568 | `_gather_tokens_from_order_kernel`（case2 token gather） |
| 1592 | `_gather_tokens_from_order` host（**BM64/BH256**） |
| 1668 | `_swiglu_amax_orderW_kernel`（case2 生效） |
| 1695 | `_swiglu_to_fp8_orderW_kernel`（case2 生效） |
| 1785 | `_swiglu_quant_fp8_orderW` host（**BN64**） |
| 2280 | `_fp8_group_gemm_pre`（case2 plain gateup host，BM128/BN256/BK128/GM8/w8/s3/grid132） |
| 2298 | `_fp8_group_gemm_pre_tma_host`（case2 down，global A scale） |
| 2317 | `_fp8_group_gemm_pre_tma_row`（非 case2 down，row A scale） |
| 2345 | `_fused_gateup_swiglu_rowA_persistent_tiles_orderW` host（**GM32/w8/s4/grid132**） |
| 2465 | `_prepare_moe_metadata`（生效副本，num_sms=32） |
| 2601 | `_linear_bf16`（生效副本） |
| 2866 | `_run_replicated` |
| 3016 | `run_kernel` |

## `_run_replicated` 主流程（当前 v233）

1. 权重：
   - case9（E=256,I=2048）：`_get_full_fp8_weights_lowmem`，逐 rank all_gather，分块量化 FP8，不保留 BF16 full。
   - 其他 FP8 case：`_get_full_weights` 缓存 full BF16，再 `_get_full_fp8_weights` 缓存 FP8。
2. 路由：`_linear_bf16(x, gate_weight)` -> BF16 logits -> FP32 softmax -> topk -> 归一化。
   - 路由 tile 表（当前生效）：
     - N<=16（E8）：BM64/BN16/BK128/w4/s3。
     - N<=32（E32）：BM64/BN32/BK128/w4/s3。
     - N<=64（E64）：BM128/BN64/BK128/w8/s3。
     - N==96（E96）：BM64/BN128/BK128/w4/s3。
     - N>64 其他（E256）：BM128/BN128/BK64/w8/s3。
3. 排序：
   - `order = flat_ids.argsort(stable=True)`。
   - case2：`_gather_tokens_from_order(x, order, k)`，BM64/BH256/w8，直接 `src // k` 取源行，无 token_idx 中间张量。
   - 其他：`_gather_tokens_row_amax_order(x, order, k)`，BM128/BH128/w8，同时 per-row atomic amax。
4. token FP8：
   - case2：`_quant_act_fp8`，amax kernel BM128/BK64/s1 + quant kernel BM128/BK64/s2，global scale。
   - 其他：`_quant_act_fp8_row_from_amax`，per-row scale。
5. metadata：`torch.bincount` -> `_prepare_moe_metadata`（num_sms=32）。
6. gateup：
   - case2：`_fp8_group_gemm_pre` plain persistent，BM128/BN256/BK128/GM8/w8/s3/grid132，输出 BF16 `[M,2I]`。
   - 其他：`_fused_gateup_swiglu_rowA_persistent_tiles_orderW`，BM128/BN128/BK128/GM32/w8/s4/grid132。
7. down：
   - case2：`_swiglu_quant_fp8_orderW`（BN64，amax s1/quant s2） -> `_fp8_group_gemm_pre_tma_host`（global A scale）。
   - 其他：`_quant_act_fp8_row_from_amax` -> `_fp8_group_gemm_pre_tma_row`（row A scale）。
8. final：`inv_order = order.argsort()` -> `_gather_branch_sum`。

## final gather 细节（本会话最大改动）

- kernel：`_gather_branch_sum_kernel_tiled`（行 404）。
- host：`_gather_branch_sum`（行 500）。
- 当前配置：
  - case1/2（H=4096,T=16384,k=2）：BT8/BH1024/w8。
  - 其他 H>=1024：BT32/BH1024/w32。
  - H>=512：BT8/BH128/w8；更小：BT8/BH128/w4。
- 每个 CTA 累加 `BLOCK_T` 个 token 的 `K_BRANCH` 个 down 行，FP32 累加后写 BF16。
- 数值/确定性：与旧一 token 一 CTA 版本相同求和顺序，逐字节一致。
- 已排除：BT64/w32（H1024）、case1/2 BT32、H3584 BH512、case9/10 单独 BT16——均无稳定收益或更慢。

## case2 专属路径细节（当前）

- token gather：`_gather_tokens_from_order_kernel`，**BM64/BH256/w8/s1**。
  - 已排除：BM128/BH128 旧值、BM32、BH512、BM256 历史慢。
- token quant：`_quant_act_fp8`，**BM128/BK64**。
  - 已排除：BK32、BM64、BM64/BK64。
- gateup：`_fp8_group_gemm_pre`，BM128/BN256/BK128/GM8/w8/s3/grid132。
  - 已排除：BN128、BK64/256、s2/s4、GM4/16、grid264、nonpersistent、BM64+metadata、
    dual gateup（normal/interleave/amax）、直接 FP8 gate/up 输出（v177 长期 Pending，未采纳）。
- SwiGLU：`_swiglu_quant_fp8_orderW`，BM128/**BN64**，amax s1/quant s2。
  - 已排除：BN128（旧）、BN256（final4 旧）、BN32、BN512 历史 TLE。
- down：`_fp8_group_gemm_pre_tma_host`，BM128/BN256/BK128/GM8/w8/s3/grid132。
  - 已排除：w4/w16、BN128、BK256/s2、grid264、num_ctas。

## fused gateup orderW 细节（当前核心）

- host 行 2345，kernel 行 1069。
- 参数：grid=132，BM128/BN128/BK128/**GROUP_M=32**/w8/**num_stages=4**。
- 每 tile 立即 `tl.atomic_max(AMAX + offs_m, row_max, mask=row_mask, sem="relaxed")`。
- orderW：kernel 内 `src = ORDER[offs_m]`，读原始 route weight，不预做 `flat_weights[order]`。
- 已排除：GM64、GM32+s3、grid264/128/160、s5（OOR）、w16、M-major persistent。

## 所有 activation amax 的 relaxed atomic

当前活跃 amax 路径全部使用 `sem="relaxed"`：
- `_gather_tokens_row_amax_order_kernel`
- `_fused_gateup_swiglu_kernel_rowA_persistent_tiles_orderW`
- `_act_amax_fp8_kernel`（case2 token quant）
- `_swiglu_amax_orderW_kernel`（case2 SwiGLU）

这是 final4 v129 的晋升改动，不要退回默认 `acq_rel`。

## TMA down GEMM 细节

- import：`from triton.tools.tensor_descriptor import TensorDescriptor`。
- host 每次创建 descriptor；descriptor 缓存无稳定收益，保持每次创建。
- B flatten `[G*N,K]`，descriptor block `[256,128]`；kernel `b = B_DESC.load([b_row, k*BLOCK_K])`，
  `acc = tl.dot(a, b.T, acc)`。
- 参数：persistent grid=132，BM128/BN256/BK128/GM8/w8/s3。
- A 普通 `tl.load`；不要用 A TMA、A+B TMA、C TMA store（均更慢）。
- FP8 down persistent launch 语法 `kernel[(132,)]`，不是 `kernel[((132,),)]`。

## custom GEMM 细节

- 所有 custom kernel 的 `expert * stride_be` 基址必须 `expert.to(tl.int64)`。
- 当前所有 K/N 都恰好是 BLOCK 整数倍；custom GEMM 已删 K/N remainder mask，只保留最后一块 M 的 row mask。
- swizzle 只在同一 expert 内进行；`row_begin`/`n_rows` 语义不变。
- `_prepare_moe_metadata` 的 `split_size_cum` 实参是 `block_row_idx_to_row_offset`。

## 缓存设计

| 缓存 | key | 说明 |
|---|---|---|
| `_FULL_WEIGHT_CACHE` | shape tuple | full BF16 权重；不能放 id |
| `_FULL_FP8_CACHE` / `_FULL_INT8_CACHE` | shape | 低精度 full weights |
| `_FULL_FP8_LOWMEM_CACHE` | shape | case9 lowmem full FP8 |
| `_TOKEN_IDX_CACHE` | T,k,device | 当前 case2 不再使用；历史保留 |
| NVSHMEM buffer 缓存 | shape/dtype | 不可乱 clear 整个 dict |

## 数值正确性要点

- 所有 `argsort` 必须 `stable=True`。
- 所有 orderW kernel 读取的 `ORDER` 都是同一个 `order`（sorted position -> original flat index）。
- final 必须 FP32 累加后写 BF16。
- case2 SwiGLU BN64 只是切分 tile 尺寸，每个元素计算顺序相同，输出应与 BN128/256 逐字节一致。
- case2 gather BM64/BH256 只改 tile，gather 结果逐元素一致。
- case2 token quant BK64 只改 tile，global amax 与 quant 结果逐元素一致。
