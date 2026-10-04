# 03 当前 canonical（114706）架构与代码位置

文件：`p1/kernel.py`（与 `p1/kernel_sorted_v3_hybrid.py` 一致）。

## 12 个测试点 shape

| case | T | H | E | I | topk |
|---:|---:|---:|---:|---:|---:|
| 1 | 16384 | 4096 | 8 | 8192 | 2 |
| 2 | 16384 | 4096 | 8 | 14336 | 2 |
| 3 | 16384 | 2048 | 32 | 2048 | 4 |
| 4 | 16384 | 2048 | 32 | 1024 | 4 |
| 5 | 8192 | 3584 | 64 | 2560 | 8 |
| 6 | 8192 | 3584 | 64 | 1024 | 8 |
| 7 | 16384 | 4096 | 96 | 2048 | 3 |
| 8 | 16384 | 4096 | 96 | 1024 | 3 |
| 9 | 4096 | 4096 | 256 | 2048 | 8 |
| 10 | 4096 | 4096 | 256 | 1536 | 8 |
| 11 | 65536 | 1024 | 32 | 1024 | 2 |
| 12 | 65536 | 1024 | 32 | 2048 | 2 |

全局专家 e 的 owner rank = `e // Ep`，local id = `e % Ep`，`Ep=E/4`。

## 路径选择（`run_kernel`，约 813 行起）

| Shape/path | 实现 |
|---|---|
| case4/11/12 | `_run_replicated`：预热期 all_gather 全量专家权重，本地 grouped GEMM，免 token 通信。 |
| E=96、E=32、E8 case1（I=8192） | `_run_kernel_a2a_sorted`：sorted direct dispatch。 |
| E8 case2（I=14336） | `_run_kernel_a2a`：旧 direct dispatch + weight/local 打包 + no-meta 回传。 |
| topk>4（case5/6/9/10） | 自写 direct all-gather hidden + NCCL 路由 allgather（case5 packed）+ grouped GEMM + BF16 partial + reduce_scatter。 |

## 关键函数 / 行号

| 行号 | 内容 |
|---:|---|
| 19-33 | `_shifted_cumsum`、`_get_direct_recv_buf` |
| 35-71 | `_direct_a2a_kernel` / `_direct_a2a`（旧 A2A case2 使用） |
| 75-119 | `_direct_allgather_kernel` / `_direct_allgather`（k>4 hidden AG） |
| 122-178 | sorted dispatch 标量 buffer cache、`_sorted_dispatch_kernel` |
| 180-193 | `_block_flat_indices`（旧路径） |
| 194-245 | `_prepare_moe_metadata`（自建，`num_sms=32`，热路径无 `.item()`） |
| 247-329 | `_linear_bf16_kernel` / `_linear_bf16`（路由 GEMM；小 N 用 BN16/32/64） |
| 286-361 | `_swiglu_weighted` |
| 382-399 | `_get_static_cache`：预热期缓存 `gate_up = cat(gate,up).contiguous()` |
| 400-574 | `_run_kernel_a2a`：旧路径，case2 专用 |
| 575-719 | `_run_kernel_a2a_sorted`：sorted direct dispatch，E96/E32/E8case1 |
| 721-750 | `_get_full_weights`：replicated 路径全量权重缓存 |
| 752-812 | `_run_replicated` |
| 813-968 | `run_kernel` 入口与 k>4 路径 |

## sorted direct dispatch 细节（`_run_kernel_a2a_sorted`）

1. 路由后按 global expert 稳定排序，生成 `send_tokens`、`send_meta`。
2. `expert_counts` 通过 `dist.all_gather_into_tensor` 收集到 `counts_all [world,E]`。
3. 计算槽位：
   - `expert_total = counts_all.sum(dim=0)`
   - `global_base = expert_total.cumsum(0) - expert_total`
   - `owner_first = (arange(E)//Ep)*Ep`
   - `slot_base = global_base - global_base[owner_first] + counts_all.cumsum(0)-counts_all)[rank]`
   - 意义：本地专家序列内，先按 local expert 排序，再按 source rank 排序，再按 token 原序。
4. `_sorted_dispatch_kernel` 直接 put：
   - token：`recv_tokens + dst_off*H`，每 chunk `num_rows*H*2` 字节。
   - E32：put weight 与 meta 到对应槽位；E8/E96：只 put token+weight。
   - 最后 `nvshmem_barrier_all_on_stream()`。
5. 专家侧：
   - `tokens_sorted = recv_tokens[:total_recv]`，无需 argsort / 二次 token gather。
   - `src_sorted` 由 `counts_all` 本地重建（repeat_interleave）。
   - grouped GEMM 后：
     - E8/E96：`down[src_sorted.argsort(stable=True)]` 恢复 source-block 顺序，NCCL variable A2A 回传；
       源 rank 用预先计算的 `final_order = send_meta.argsort()` 归并，无回传 meta。
     - E32：`partial.index_add_` + `reduce_scatter_tensor`（output 连续时直接写 output）。
6. chunks 配置：
   - E96：`4`
   - E32：`8`
   - E8：`32`

## 旧 A2A 路径（case2，`_run_kernel_a2a`）

- 保持 114668 结构：NVSHMEM source-block direct dispatch；
- E8 case2 把 FP32 weight + local id 打包进一个 int64 A2A；
- 去掉回传 meta A2A；返回顺序依赖 direct dispatch ordering invariant：
  down 返回行序 == `send_meta` 顺序，源 rank 本地 argsort 即可归并。

## k>4 路径细节

- hidden all-gather：`_direct_allgather`，128 chunks / 8 warps，kernel 写全部 rank block
  （包括本地），随后 `nvshmem_barrier_all_on_stream()`。
- 路由 gather：
  - case5：ids+weights 打包 int64，一次 NCCL `all_gather_into_tensor`。
  - 其他：ids int32 与 weights float32 分别 NCCL all_gather。
- 选分支 -> local argsort -> 两次 token gather（本会话试过单次 gather 变体，未晋升）。
- 本地 grouped GEMM + SwiGLU + down。
- `partial [world*T,H] BF16` 做 `index_add_`，BF16 `reduce_scatter_tensor`；
  output 连续时直接 reduce_scatter 到 output。

## 数值语义（必须保持）

- route：BF16 GEMM -> BF16 logits -> FP32 softmax -> topk -> FP32 权重归一化（分母 max 1e-6）。
- gate/up：BF16 GEMM -> BF16 -> FP32。
- SwiGLU：FP32 `silu(g)*u*w`。
- down：activation 转 BF16 -> BF16 GEMM -> FP32 累加 -> BF16 output。
- 确定性：stable argsort、固定编译参数、固定归并顺序；避免任何 atomic 顺序影响最终求和顺序。
