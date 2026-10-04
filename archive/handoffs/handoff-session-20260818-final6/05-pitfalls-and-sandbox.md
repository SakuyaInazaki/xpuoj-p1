# 05 全部踩坑、沙箱限制、必须注意的细节

## A. 沙箱 / Language validation / Import（继承 final4，仍然有效）

- 禁止 `try/except`：历史上直接 Language validation 崩溃。
- 禁止 import `inspect` / `json`；诊断用 `repr()` 和字符串拼接。
- 禁止访问任何 dunder attribute：`triton.__version__` 也会被拦截。
- 禁止 `torch.tensor(...)`；用 `torch.zeros(...) + ...` 或 `torch.full`。
- 禁止 `torch.matmul`。
- 禁止函数式 `torch.argsort` / `torch.repeat_interleave` / `torch.cumsum`；用 tensor method。
- 禁止 `tensor.index_select`、`tensor.clamp_min`、`scatter_reduce_`、`tensor.data_ptr()`。
- `tensor.sort(stable=True)` 被 TensorGuard 禁止；functional `torch.sort(...)` 也被拦截。
- `torch.topk` 的 `k` 必须位置传参。
- 禁止 import `nvshmem` 模块；NVSHMEM 经 `triton_dist.language.extra.libshmem_device` 与
  `nvshmem_create_tensor` / `nvshmem_barrier_all_on_stream` 使用。
- 全局非字面量赋值可能失败；全局 dict 字面量 + 函数内 mutate 可用。
- `torch.cuda.Stream/Event` 可用；诊断 raise 中可用 `torch.cuda.synchronize()`。
- 每个 direct allgather 后必须单独 barrier；不要多个 AG 共用一个 barrier。

## B. 本会话新踩的坑

### 1. 平台偶发 20-30 分钟 Pending

- 117032/v177、117077-117078、117083-117084 等都曾 Pending 很久后正常返回。
- 工具层 `timeout 300s` 杀掉 submit 轮询不代表评测失败；不要重复提交同一文件。
- 用 `Client.get_detail()` 查；`list_submissions` 通常只返回最近约 10 条，不可依赖 takeCount。
- 多个 Pending 可能排队，简单提交也可能先出结果。

### 2. 第一窗大赚经常是噪声，必须多窗 A/B

- 典型例子：
  - v197 N96 BM64/BN64 首窗 43407 vs 43573，反向窗 43663 vs 43642，无收益。
  - v232b H1024 gather BH256 首窗 case11/12 大赚，反向窗 case11/12 变慢。
  - v209 token quant BK64：4 窗 case2 差值为 -0.247 / +0.200 / -0.257 / -0.158，
    3 窗更快才晋升。
  - v229 row gather BM64/BH256：首窗 -0.423，第二窗 +0.212，第三窗 +0.579，关闭。
- 纪律：每个候选至少 2 对同窗口 A/B（候选+当前 base 对照），方向不一致就继续 tie-break 或放弃。

### 3. raw / tb 异常不能作为晋升依据

- 116961 case5 tb=264.994，raw76.67 成 scoreboard best。
- 117151 case9 tb=41.192，raw75.75。
- 117195/117296 等也有 tb 异常。
- 只看 timeUsed/tk；scoreboard 的 raw 不反映真实性能。

### 4. 从旧文件复制候选会引入旧 base

- 本会话多次出现“候选基于旧 base”的问题，例如 v203/v203b、v198/v198b、v188/v188b。
- 提交前一定 `diff -u p1/kernel.py <candidate>` 确认只有目标改动。

### 5. Python host 修改要检查变量定义和分支顺序

- v176：case2 设 `gateup=None` 后，down 分支先判断 `gateup is None`，引用未定义的
  `act_bf16`，UnboundLocalError WA。修复：先判断 `use_case2_plain_fp8`。
- v232：修改 `_gather_tokens_row_amax_order` 时漏掉 `amax = torch.zeros(...)`，
  远程 NameError WA。修复 v232b。
- 教训：不能只依赖 `py_compile`；要 diff + 肉眼检查所有被改 host 函数。

### 6. 大 tile / 多 warp 的 OOR 与慢

- case2 dual BN256/w16：OutOfResources shared（245760 > 232448）。
- down w4：可编译但 timeUsed 108642，严重慢。
- final BT64/w32、fused GM64、case2 SwiGLU BN32/BH512 方向都更慢。

### 7. case2 结构化的失败线

- dual gateup 三种布局（interleave `[M,I,2]`、amax epilogue、normal `[M,2I]`）全部明显慢：
  case2 分别为 13.286 / 12.446 / 10.543 ms，对比 plain 约 10.15-10.38ms。
- dual BN256/w16 直接 OOR。
- case2 gate/up 直接 FP8 输出（v176/v177）先有 Python bug，修复后长期 Pending，未证明收益。
- 结论：不要在 case2 plain gateup 上继续 dual/multi-dot 盲试，除非有 profiling 依据。

### 8. 同窗口对照记录方式

- 日志名要含候选和 control 语义；不要只写 `submit_vXXX.log`。
- 无 `--poll` 的日志只有 submissionId；状态从 API 详情补。
- 最终状态和逐点 tk/tb 应落到 raw JSON（本文件夹已提供）。

## C. 历史仍然有效的坑（继承 final4，不要重试）

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
- **device-side TMA descriptor 不可用**：`tl.make_tensor_descriptor` 报 `_semantic` 错误；
  只用 host-side `TensorDescriptor`。
- **side stream 做 inverse argsort 破坏 determinism**。
- **中间张量全局 buffer cache WA/TLE**。
- **fused gateup M-major persistent 有 SQNR 错误风险**。
- **order 转 int32 无收益**。
- **TMA descriptor cache 无稳定收益**。

## D. 提交前检查

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
diff -u p1/kernel.py p1/<candidate>.py   # 确认只改了目标
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_117300_backup.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

- 每次实验保存候选源码和日志。
- 诊断提交可 raise RuntimeError 打印信息；WA 不降低历史最佳。
- 错误诊断优先查 submission detail 的 `userError` 中的
  `Execution error`、`Pointer argument`、`PassManager`、`SQNR`、`DETERMINISM FAIL`、
  `time limit`、`TensorGuardError`、`OutOfResources`。
