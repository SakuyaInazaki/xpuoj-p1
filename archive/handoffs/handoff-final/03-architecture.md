# 03 当前 canonical（114970）架构与代码位置

文件：`p1/kernel.py` = `p1/kernel_case9_int8_both_clean.py`。
注意：该文件第 1488 行附近开始存在一段历史遗留的重复定义（`_prepare_moe_metadata`、
`_linear_bf16`、`_run_kernel_a2a` 等），后面第 1847 行起又有正式定义；Python 以后定义为准，
在线 submission 也是这段代码并通过。**不要因为重复定义惊慌，但清理它必须重新提交。**

## 路径选择（`run_kernel`，约 1989 行起）

当前 12 个 case 实际走的路径：

| case | 路径 |
|---|---|
| 1/2 | replicated，E8；case2 gateup 用 per-row INT8，down 用 FP8 |
| 3/4 | replicated，E32 |
| 5/6 | replicated，E64；case5 gateup+down 用 FP8，case6 用 BF16 |
| 7/8 | replicated，E96；FP8 |
| 9 | **唯一非 replicated**：hidden direct AG（E256 专用 chunks=32）+ local expert INT8 gateup/down + BF16 partial reduce |
| 10 | replicated，E256/I1536；gateup BF16，down per-row INT8 |
| 11/12 | replicated，E32；case12 FP8 |

旧的 A2A/sorted A2A 代码仍在文件中，但对评测 shape 已不可达。

## 关键函数/行号

| 行号 | 内容 |
|---|---|
| 100-124 | `_get_direct_ag_buf` / `_direct_allgather`（支持 chunks 参数） |
| 110-124 | E256 hidden AG 当前用 `chunks=32` |
| 126-171 | route flat AG 工具（case6 历史路径使用） |
| 175-230 | sorted dispatch 工具（评测 shape 已不可达） |
| 246-296 | `_prepare_moe_metadata`（第一份，实际运行时被后一份覆盖） |
| 374-410 | `_gather_branch_sum`：H=4096 用 BLOCK_H=1024，H=2048 用 2048，H=1024 用 1024 |
| 505-516 | `_get_token_idx` 静态索引缓存 |
| 517-536 | `_get_static_cache`：**shape-key** 缓存 local gate_up |
| 851-898 | `_fp8_group_gemm_kernel`（BLOCK_M=128,BLOCK_N=256,BLOCK_K=128,w8,s3） |
| 900-906 | `_quant_weight_fp8`：per-expert/per-N weight scale |
| 908-950 | FP8 activation quant + `_quant_act_fp8` |
| 953-1044 | 融合 SwiGLU→FP8 quant：`_swiglu_amax_kernel` + `_swiglu_to_fp8_kernel` + `_swiglu_quant_fp8` |
| 1047-1145 | INT8 grouped GEMM / row quant kernel |
| 1157-1273 | per-row INT8 quant + fused SwiGLU INT8 quant |
| 1275-1347 | `_quant_weight_int8`、`_quant_act_int8`、`_swiglu_quant_int8`、`_int8_group_gemm_pre` |
| 1354-1373 | full INT8 gateup 缓存（case2 使用） |
| 1375-1396 | full FP8 权重缓存（shape-key） |
| 1398-1417 | full INT8 down 缓存（case10 使用） |
| 1419-1431 | local INT8 gateup 缓存（case9 使用） |
| 1433-1445 | local INT8 down 缓存（case9 使用） |
| 1447-1459 | local FP8 缓存 |
| 1461-1486 | `_fp8_group_gemm_pre` / `_fp8_group_gemm` |
| 1817-1845 | `_get_full_weights`（**shape-key**，all_gather 全量专家权重） |
| 1875-1987 | `_run_replicated` |
| 1989-... | `run_kernel` 与 case9 allgather 路径 |

## replicated 路径（`_run_replicated`）

1. `_get_full_weights` 按 shape 缓存 full `gate_up [E,2I,H]` 和 `down [E,H,I]`。
2. 路由：本地 `_linear_bf16` → softmax → topk → 归一化。
3. 对 topk ids 稳定 argsort，gather 出 `tokens_sorted`。
4. 按 E/H/I 选择 GEMM：
   - case1/5/E96：FP8 both。
   - case2：INT8 gateup + FP8 down。
   - case10：BF16 gateup + INT8 down。
   - 其他：官方 BF16 `moe_grouped_gemm`。
5. FP8/INT8 权重缓存全部按 shape。
6. 最终 `_gather_branch_sum` 单遍 gather+FP32 累加写 output。

## case9 allgather 路径

1. 本地路由。
2. `_direct_allgather` hidden（E256 专用 chunks=32）。
3. ids/weights 走 NCCL `all_gather_into_tensor`（试过 direct flat 和打包，均未稳定胜出）。
4. `selected_flat = (owner==rank).nonzero`。
5. 先 gather token，再 argsort local，再二次 gather；两次 gather 方案比单次 gather 稳定更快。
6. local expert GEMM：
   - 当前：INT8 gateup + INT8 down（114970）。
   - 备选：FP8 both（114971 case9 4.936ms，略好但总分受噪声影响）。
7. `partial [world*T,H] BF16` index_add，BF16 `reduce_scatter_tensor`；output 连续时直接写 output。

## 缓存设计（核心经验）

| 缓存 | key | 说明 |
|---|---|---|
| `_get_static_cache` | shape + topk | local gate_up；**不能放 id** |
| `_get_full_weights` | shape | replicated full weights；**不能放 id** |
| `_get_full_fp8_weights` / `_get_full_int8_weights` / `_get_full_down_int8_weights` | shape | 低精度 full weights |
| `_get_static_fp8` / `_get_static_int8_*` | shape | case9 local 低精度权重 |
| `_get_direct_recv_buf` / `_get_direct_ag_buf` | shape | 大 NVSHMEM buffer，单条目缓存足够 |
| `_get_sorted_buf` / `_get_route_ag_buf` | shape+dtype | 小 buffer，**多 key 缓存，不能 clear 整个 dict** |
| `_get_token_idx` | T,k,device | 静态索引 |

## 正确性保证

- 所有 argsort 必须 `stable=True`。
- 权重/输出缓存按 shape 是安全的：题目权重只读固定，harness 只换张量对象。
- FP8/INT8 权重 scale 在预热期量化并缓存；activation 每次运行重新量化。
- INT8 activation 必须 per-row scale + round；全局 scale SQNR 不足。
- FP8 activation 当前为全局 scale，融合 SwiGLU quant 避免 act BF16 物化。
