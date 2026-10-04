# 05 全部踩坑、沙箱限制、必须注意的细节

## A. 沙箱 / Language validation / Import

- 禁止 `try/except`：历史上直接 Language validation 崩溃。
- 禁止 import `inspect` / `json`；诊断用 `repr()` 和字符串拼接。
- 禁止访问任何 dunder attribute：`triton.__version__` 也会被拦截。
- 禁止 `torch.tensor(...)`；用 `torch.zeros(...) + ...` 或 `torch.full`。
- 禁止 `torch.matmul`。
- 禁止函数式 `torch.argsort` / `torch.repeat_interleave` / `torch.cumsum`；用 tensor method。
- 禁止 `tensor.index_select`、`tensor.clamp_min`、`scatter_reduce_`、`tensor.data_ptr()`。
- **本会话新增：`tensor.sort(stable=True)` 被 TensorGuard 禁止（116683）。**
- **本会话新增：functional `torch.sort(...)` 也被拦截（116694）。**
- `torch.topk` 的 `k` 必须位置传参。
- 禁止 import `nvshmem` 模块；NVSHMEM 经 `triton_dist.language.extra.libshmem_device` 与
  `nvshmem_create_tensor` / `nvshmem_barrier_all_on_stream` 使用。
- 全局非字面量赋值可能失败；全局 dict 字面量 + 函数内 mutate 可用。
- `torch.cuda.Stream/Event` 可用；诊断 raise 中可用 `torch.cuda.synchronize()`。
- 每个 direct allgather 后必须单独 barrier；不要多个 AG 共用一个 barrier。

## B. 本会话新踩的坑

### 1. device-side TMA descriptor 不可用（继承确认）

- `tl.make_tensor_descriptor` 在远程 Triton-dist 版本调用报：
  `ValueError: Did you forget to add @triton.jit ? (_semantic argument must be provided...)`
- 继续只用 host-side `TensorDescriptor`。

### 2. case2 full-fused gateup 反复 TLE，彻底关闭

- 历史：115769/115848/115927 TLE。
- 本会话再试 116646：K constexpr + persistent grid132 + per-row activation amax，
  仍然 `total problem time limit`（timeUsed 473140376）。
- 无论 global-A scalar-amax 还是 row-amax，case2 full-fused 都不要重试。

### 3. fused gateup persistent 的正确实现与错误实现

- 有效版本：persistent tile-loop（grid132），每 tile immediate per-row atomic amax。
- v106/v107/v108 的 M-major persistent 曾因 **K 循环里没有推进 `a_ptrs`** 导致 SQNR 错；
  修正后（116666）正确但 timeUsed 53467，明显慢。不要用 M-major。
- `num_stages=5` 直接 OutOfResources（116711）；`num_warps=16` 明显慢（116727）；
  `grid=128/160` 慢（116716/116676）。当前最优：grid132/GM16/w8/s4。

### 4. side stream 做 inverse argsort 会破坏 determinism

- 116841 把 `inv_order = order.argsort()` 放到 side stream，最终 `inv_event.wait()`。
- 结果 case9 `DETERMINISM FAIL mismatched_bytes=30451906/33554432`。
- 结论：不要用 side stream 重叠 `inv_order.argsort()` 与主 GEMM。

### 5. 中间张量全局 buffer cache 会 WA/TLE

- 116679 试图按 shape 缓存 `tokens_sorted/act/amax/metadata` 等中间 tensor。
- 结果整体 time limit 或失败；可能触及 harness 多次调用/异步内存复用边界。
- 不要做全量 buffer cache；静态权重 cache 和 `_TOKEN_IDX_CACHE` 保持原样即可。

### 6. TMA descriptor cache 无稳定收益

- 116667/116724 缓存 host-side TensorDescriptor：第一次像有收益，复测与对照打平。
- 保持每次创建 descriptor，简单安全。

### 7. order32 无收益

- 116792/116795 尝试把 `order` 转 int32 供 gather/fused/SwiGLU 使用：
  整体无稳定收益；case1/2 有时更慢。不要做。

### 8. case2 token amax 与 gather 融合无收益

- scalar atomic 版（116816）和 partial-store + `partial.amax()` 版（116828）都无稳定收益。
- 保留当前 case2 `_gather_tokens_from_order` + 独立 `_quant_act_fp8`。

### 9. route 变体的结论

- E<=16：BM64/BN16/BK128/w4/s3 是当前最优（v159）。
- E<=16 BK256 更慢（116896）；BM64/w8 更慢（116863）。
- E<=32 BM64（116868）和 E<=64 BM64（116888/116892）两次互有胜负，不采用。
- 不要盲扫 route E256 的 BM64/BK128（历史已确认慢）。

## C. 历史仍然有效的坑（继承）

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
- **case2 full-fused、split gate/up、B 转置、fused gather+quant、down slot**：均不要做。
- **per-tile INT8 会产生非有限值**；INT8 activation 必须 per-row scale + round。
- **E=96 非 2 的幂**，部分官方 kernel 的 `tl.arange` 会失败。
- **direct allgather v1 的“本地块跳过写”会在计时段失败**；kernel 必须写全部 rank block。
- 评测机波动大；同代码 timeUsed 可差 1ms+，不要单次下结论。

## D. 提交前检查

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_116882_backup.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

- 每次实验保存候选源码和日志。
- 诊断提交可 raise RuntimeError 打印信息；WA 不降低历史最佳。
- 错误诊断优先查 submission detail 的 `userError` 中的
  `Execution error`、`Pointer argument`、`PassManager`、`SQNR`、`DETERMINISM FAIL`、
  `time limit`、`TensorGuardError`、`OutOfResources`。
