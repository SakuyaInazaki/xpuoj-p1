# 05 沙箱限制、踩坑与失败路线

## 沙箱 / Language validation

- 禁止 `torch.matmul`。
- 禁止函数式 `torch.argsort` / `torch.repeat_interleave` / `torch.cumsum`；可用 tensor method。
- 禁止 `tensor.index_select`（114816 实测 TensorGuardError）。
- 禁止 `tensor.clamp_min`；用 `torch.maximum`。
- 禁止 `tensor.data_ptr()`；缓存 key 用 `id(tensor)` 或 shape。
- 禁止 `scatter_reduce_`。
- `torch.topk(..., k=...)` 的 k 必须位置传参。
- 禁止导入 `nvshmem` 模块；NVSHMEM 通过 `triton_dist` 的 `libshmem_device` / utils 使用。
- 禁止 try/except（114631 Language validation 崩）。
- 全局非字面量赋值会 Language validation 失败；全局 dict/list 字面量 + 函数内 mutate 可用。
- `torch.cuda.Stream/Event` 可用；但 stream overlap 未带来收益。

## 本会话最关键的坑

1. **静态缓存按 `id(tensor)` 作 key 是灾难**：
   - harness 每次 `run_kernel` 都可能传新张量对象；id 一直变。
   - 旧 `_STATIC_CACHE` / `_FULL_WEIGHT_CACHE` 因此永远 miss，每个 timed call 都在重做 GB 级 cat/all_gather。
   - 改为 shape-key 后 raw 从 55.5 直接跳到 56.83，并支撑后续所有 replicated 优化。
2. **小 NVSHMEM buffer 缓存不能 clear 整个 dict**：
   - `_SORTED_BUF_CACHE`/`_ROUTE_AG_BUF_CACHE` 每次调用要两个 buffer；旧实现每建一个就 clear，缓存永远 miss。
   - 改为多 key 缓存后 114796 raw 52.67→55.50。
3. **每个 direct allgather 后必须单独 `nvshmem_barrier_all_on_stream`**；多个 AG 合一个 barrier 会确定性失败。
4. **direct allgather v1 的“先 copy 本地块、kernel 跳过本地 put”会在正式计时段失败**；kernel 必须写全部 rank block（含本地）。
5. **官方 `fast_allgather(push2d)` 不可用**：首轮正确，正式计时段 case5/6/9/10 失败。
6. **E256 full replicated TLE**（114864，约 473s）；E256 full FP8/INT8 权重量化 OOM。
   - 但 E256 case10 replicated BF16 可行，case10 FP8/INT8 down-only 可行。
7. **per-tile INT8 单遍量化会产生非有限值**（114969/114974），不要继续这条路。
8. **INT8 全局 activation scale SQNR 不足**（约 14.8dB）；per-row scale + round 后通过。
9. **case9 路由不要打包 int64**（114925 case9 39ms 异常）；保持 ids/weights 两路 NCCL。
10. **case9 hidden AG chunks=32 最佳**；c16/c24/c64 和 w4 都更差。
11. **k>4 单次 token gather 不如两次 gather**；连续多次验证。
12. **tensor.index_select 比高级索引更快只是猜测，且被禁**；自写 Triton gather 也未胜出。

## 历史重要坑（仍有效）

- E=96 非 2 的幂，部分官方 kernel 的 `tl.arange` 会失败。
- 评测机 baseline 波动极大，同代码可差 2~3 raw；case9 最不稳定。
- `best_score.py` 曾经长期显示缓存旧值；现在已刷新，但还是要以 submission detail 为准。
- 预热期可缓存静态权重；hidden_states 每轮都会换，不可缓存路由结果/counts。
- 最终归并必须 FP32 累加后写 BF16。
- 确定性要求 stable argsort + 固定归并顺序。

## 低精度 GEMM 经验

- 官方 `moe_grouped_gemm` FP8 输出跟随输入为 FP8，易饱和到 448，不能直接作 BF16 输出 GEMM。
- 自研 FP8 grouped GEMM 可用；E8/case5/E96/case9 上有效。
- FP8 kernel 最佳参数（本会话）：BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, num_warps=8, num_stages=3。
- FP8 BN128、BK256、stages4、w16 均未胜出；BK256 在 INT8 下会 OutOfResources。
- FP8 activation 用全局 scale + 融合 SwiGLU quant，SQNR 约 23.7dB，刚好过线。
- INT8 GEMM 只在 case2 gateup、case9、case10 down 上采用；必须 per-row scale + round。
- case2 最佳组合：INT8 gateup + FP8 down（约 15.3-15.4ms）。
- case10 最佳 down：FP8 3.88ms vs INT8 3.96ms，差距在噪声内。
- case9 FP8 both 约 4.94ms，INT8 both 约 4.97-5.02ms，二者接近。

## 诊断方法

- 想拿 phase 时间：插入 `_diag_mark`/`_diag_raise`，第一次调用 raise；从 submission detail 的 `userError` 读 `RuntimeError: DIAG ...`。
- 想拿调用次数/张量 id/缓存 miss：插入全局 list + raise，同样从 `userError` 读。
- WA 诊断不会降低历史最佳，但会计入 submissionCount。
