# 05 全部踩坑、沙箱限制、必须注意的细节

## A. 沙箱 / Language validation / Import

- 禁止 `try/except`：历史上直接 Language validation 崩溃。
- 禁止 import `inspect` / `json`；诊断用 `repr()` 和字符串拼接。
- 禁止访问任何 dunder attribute：`triton.__version__` 也会被拦截
  （本会话 116215 踩到）。但 JITFunction 的 `.arg_names` / `.signature` / `.src` 允许。
- 禁止 `torch.tensor(...)`；用 `torch.zeros(...) + ...` 或 `torch.full`。
- 禁止 `torch.matmul`。
- 禁止函数式 `torch.argsort` / `torch.repeat_interleave` / `torch.cumsum`；用 tensor method。
- 禁止 `tensor.index_select`、`tensor.clamp_min`、`scatter_reduce_`、`tensor.data_ptr()`。
- `torch.topk` 的 `k` 必须位置传参。
- 禁止 import `nvshmem` 模块；NVSHMEM 经 `triton_dist.language.extra.libshmem_device` 与
  `nvshmem_create_tensor` / `nvshmem_barrier_all_on_stream` 使用。
- 全局非字面量赋值可能失败；全局 dict 字面量 + 函数内 mutate 可用。
- `torch.cuda.Stream/Event` 可用；诊断 raise 中可用 `torch.cuda.synchronize()`。
- 每个 direct allgather 后必须单独 barrier；不要多个 AG 共用一个 barrier。

## B. 本会话新踩的坑

### 1. device-side TMA descriptor 不可用

- `tl.make_tensor_descriptor` 在探测中存在，但在远程 Triton-dist 版本里，
  无论 `@triton_dist.jit` 还是 `@triton.jit`，调用都会报：
  `ValueError: Did you forget to add @triton.jit ? (_semantic argument must be provided...)`
- 结论：**不要使用 device-side `tl.make_tensor_descriptor`**。
- 可用替代：host-side `from triton.tools.tensor_descriptor import TensorDescriptor`，
  在 host 构造 descriptor 后作为参数传入 kernel。

### 2. host-side TensorDescriptor 的可用用法

- 对 `[E,N,K]` 权重，flatten 成 `[E*N,K]` 后创建 2D descriptor：
  ```python
  b_flat = b_q.view(G * N, K)
  b_desc = TensorDescriptor(b_flat, b_flat.shape, b_flat.stride(), [256, 128])
  ```
- kernel 中 `b = B_DESC.load([expert*N + pid_n*BN, k*BK])`，随后
  `acc = tl.dot(a, b.T, acc)`。
- FP8 BF16 descriptor 均可用；block 内层尺寸 128（FP8 128B / BF16 256B）满足对齐。
- 需要 `@triton.jit` 的 kernel 接收 descriptor 参数。

### 3. TMA 的适用范围（实测）

**有效：**
- FP8 down GEMM 的 B 矩阵 TMA load，保留 A 普通 `tl.load`、C 普通 store，
  persistent grid132、BM128/BN256/BK128/GM8/w8/s3。

**无效/变慢：**
- TMA 用于 case2 plain gateup（116244 变慢）。
- TMA 用于 fused gateup（116249 变慢）。
- C TMA store（116255 变慢）。
- A+B 双 TMA（116262 明显变慢）。
- contiguous-M 调度替代 swizzle（116264 明显变慢）。
- BN128（116256）、BK256/s2（116216/116270）、s4（116252）、s2（116270）、
  outer `flatten=True`（116281 不稳定）、inner `warp_specialize=True`（116287 变慢）。
- routing GEMM 的 A/B host-descriptor TMA（116549 变慢）。

### 4. num_ctas / thread-block cluster

- fused kernel 加 `num_ctas=2` 直接 `RuntimeError: PassManager::run failed`。
- 仅 down TMA 加 `num_ctas=2` 可编译但明显变慢（116545）。
- 当前版本不要使用 `num_ctas`。

### 5. per-tile FP8 activation scale + tiled down GEMM

- v62 曾尝试：每个 `[128,K-tile]` 独立 scale，down kernel 每 K block
  `partial = tl.dot(a,b); acc += partial * scale`。
- 结果 timeUsed 63403，严重变慢。**per-K-block scale 会破坏 dot 累加流水，不要做。**
- 正确路线是 per-row scale：每行一个 scale，dot 正常累加，epilogue 一次性乘行 scale。

### 6. fused gateup persistent / M-only 重排 + batched row amax

- 尝试把多个 N-block 的 row max 在寄存器累积后再 atomic 一次：
  - persistent grid132 版本：SQNR 严重错误。
  - M-only grid 版本：SQNR 严重错误。
  - 即使 row_max_acc 从 `-inf` 改为 0 仍错误。
- 原因未完全定位，但该方向已放弃。当前必须保持每个 tile 立即
  `tl.atomic_max(AMAX + offs_m, row_max, mask=row_mask)`。

### 7. 逆置换用 index assignment 替代 argsort

- `inv[order] = arange` 在 GPU 上比 `order.argsort()` 慢（116135）。
- 继续使用 `inv_order = order.argsort()`。

### 8. routing 优化

- 改 `topk(logits)` 再归一化：116193 未胜。
- 直接 `topk(BF16 logits)`：116195 未胜。
- route N/K constexpr、route TMA：均未胜。保留现有 BF16 route kernel。

## C. 历史仍然有效的坑（继承自前序会话）

- **E=256 expert 指针 int32 溢出**：`expert * stride_be` 必须 `expert.to(tl.int64)`。
- **case9 full replicated FP8 普通量化 OOM**：必须 `_get_full_fp8_weights_lowmem`。
- **静态缓存按 shape，不能按 id(tensor)**。
- **metadata 语义**：
  - `build_block_row_idx_info_kernel` 输出的 `block_row_idx_to_row_offset`
    传给 custom kernel 的 `split_size_cum`。
  - BLOCK_M 必须与 metadata 的 `GROUP_GEMM_BLOCK_SIZE_M` 一致；BM64 需配套 metadata64。
- **FP8 down persistent launch 语法**：`kernel[(132,)]`，不是 `kernel[((132,),)]`。
- **swizzle 只在同一 expert 内安全**。
- **official FP8 dot**：`dot_k_const` 的 `tl.dot` 对 fp8e4nv 报 Unsupported。
- **case2 full-fused FP8 gateup 连续 TLE**，不要重试。
- **case2 gate/up 拆成两个独立 GEMM TLE**。
- **fused gather + FP8 quant TLE**，不要重试。
- **B 转置布局 [G,K,N]** 在当前 Triton 版本严重变慢。
- **down 直接写 [T,k,H] slot** 比当前 final gather 慢。
- **per-tile INT8 会产生非有限值**；INT8 activation 必须 per-row scale + round。
- **E=96 非 2 的幂**，部分官方 kernel 的 `tl.arange` 会失败。
- **评测机波动大**；同代码 timeUsed 可差 1ms，不要单次下结论。
- direct allgather v1 的“本地块跳过写”会在计时段失败；kernel 必须写全部 rank block。

## D. 提交前检查

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_116310_backup.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

- 每次实验保存候选源码和日志。
- 诊断提交可 raise RuntimeError 打印信息；WA 不降低历史最佳。
- 错误诊断优先查 submission detail 的 `userError` 中的
  `Execution error`、`Pointer argument`、`PassManager`、`SQNR`、`time limit`。
