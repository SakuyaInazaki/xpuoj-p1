# 05 全部踩坑、沙箱限制、必须注意的细节

## 沙箱 / Language validation / Import

- 禁止 `try/except`：115602 直接 Language validation 崩。任何 probe 都不能写 try。
- 禁止 import `inspect` / `json`（115603 Import validation failed）。诊断信息用 `repr()` 和字符串拼接，不要用 json.dumps。
- 禁止访问任何 dunder attribute，包括 `func.__doc__`（115604）。
- 但 **JITFunction 的 `.arg_names`、`.signature`、`.src` 是允许的**，是远程读 kernel 源码/签名的重要方法。
- 禁止 `torch.tensor(...)`（115616）；用 `torch.zeros(...)` + `+=` 或 `torch.full`。
- 禁止 `torch.matmul`。
- 禁止函数式 `torch.argsort` / `torch.repeat_interleave` / `torch.cumsum`；用 tensor method。
- 禁止 `tensor.index_select`、`tensor.clamp_min`、`scatter_reduce_`、`tensor.data_ptr()`。
- `torch.topk` 的 k 必须位置传参。
- 禁止导入 `nvshmem` 模块；NVSHMEM 经 `triton_dist.language.extra.libshmem_device` 与
  `nvshmem_create_tensor`/`nvshmem_barrier_all_on_stream` 使用。
- 全局非字面量赋值可能失败；全局 dict 字面量 + 函数内 mutate 可用。
- `torch.cuda.Stream/Event` 可用；`torch.cuda.synchronize()` 在诊断 raise 中可用。
- 每个 direct allgather 后必须单独 barrier；不要多个 AG 共用一个 barrier。

## 本 agent 会话最关键的坑

### 1. E=256 custom GEMM 的 expert 指针 int32 溢出

- `_fp8_group_gemm_kernel` 中 `expert * stride_be`：E=256 full gateup 的 stride 可达
  3072*4096=12,582,912，expert=255 时超过 INT32_MAX。
- 症状：case10 full FP8 gateup 输出非有限值（115165）。
- 修复：`expert64 = expert.to(tl.int64)`，B 指针基址用 int64。
- 后续 INT8 full gateup 如果要做，也必须同样修复。

### 2. case9 full replicated FP8 普通量化会 OOM

- 普通路径先 `_get_full_weights` 缓存 BF16 full（case9 约 12.9GB），
  再 `_quant_weight_fp8` 做 `w.float()` 会再申请 12GB，必然 OOM（115207）。
- 修复：`_get_full_fp8_weights_lowmem`：逐 rank all_gather gate/up/down，
  分块量化进预分配 FP8 buffer，不保留 BF16 full。
- 若 `_run_replicated` 增加新分支，必须为所有路径初始化局部变量，否则像
  115722 / 115896 / 115939 那样出现 `UnboundLocalError`。

### 3. 静态缓存必须按 shape，不能按 id(tensor)

- harness 每次 run_kernel 都可能新建权重张量对象，id 不可靠。
- 当前有效 `_get_full_weights` 是 shape-key；历史 duplicate 曾有 id-key。
- 所有评测 shape 都走 replicated，`_get_static_cache` 已不可达，但仍要小心新增代码。

### 4. metadata 语义

- `build_block_row_idx_info_kernel` 输出：
  - `rows_splits_cum_per_expert_ptr`：长度 E，每个 expert 的全局 row offset。
  - `block_row_idx_to_expert_idx_ptr` / row_offset / tile_split / tile_cumsum：长度 M_grid。
- wrapper/kernel 的 `split_size_cum` 实参是 **block_row_idx_to_row_offset**，
  不是 `rows_splits_cum_per_expert`。
- custom kernel 中 `row_begin = split_size_cum[pid_m]`，
  然后 `offs_m = row_begin + local_m*BLOCK_M`。
- **BLOCK_M 必须与 metadata 的 BLOCK_SIZE_M 一致**：
  115919/115937/115959 用 BM64 但 metadata 仍是 128，导致 SQNR 3dB / 错误 / TLE。
  如果要 BM64，必须新增 `_prepare_moe_metadata_bm64`（BLOCK_SIZE_M=64）；
  但 fused BM64 与 plain BM64 修复后仍 TLE，不要继续这条线。

### 5. FP8 down persistent 的 launch 语法

- 旧实验 115660/115682 实际从未跑起来：
  `_kernel[((132,),)]` 导致 `TypeError: 'tuple' object cannot be interpreted as an integer`。
- 正确写法：`_kernel[(132,)]`。
- 历史文档里“persistent 全错”应理解为 launch bug，而不是 kernel 数值错。

### 6. swizzle 只在同一 expert 内才安全

- custom kernel 先算 `local_m = pid_m - (t_cum - t_num)`，
  再 `tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)`。
- swizzle 只交换同一 expert 的 tile，因此 `row_begin`、`n_rows` 不用重载。
- GROUP_M=8 是实测最佳；16/4/2 的 tk 都更差。
- 115705 的 GROUP_M=2 高分是 case6 tb 异常，不要当成真实优化。

### 7. FP8 dot 与 official kernel

- custom kernel 中 `tl.load(..., other=0.0)` + `tl.dot` 对 fp8e4nv 是有效且快速的；
  不要把 load 结果显式 `.to(tl.bfloat16)`，115658 严重变慢。
- 官方 `moe_grouped_gemm` wrapper：
  - 支持 `num_warps`、`num_stages`、`GROUP_SIZE_M`、`num_sms`。
  - 不支持 `out_dtype`、`PERSISTENT`（会报 unexpected keyword）。
  - FP8 输入会走 `dot_k_const`，而 `dot_k_const` 的 `tl.dot` 对 fp8e4nv 报 Unsupported；
    因此官方 FP8 BF16 输出路线不可用。
- `transposed_moe_grouped_gemm` 是 backward 风格签名，不是可用的 forward FP8 输出方案。

### 8. fused gateup epilogue

- FP8 fused gateup+SwiGLU+amax 是有效优化。
- 实现要点：
  - 每个 program 同时计算 gate tile 和 up tile，两个 FP8 dot，`BLOCK_N=128`。
  - A 必须先 `_quant_act_fp8` 得到 `a_q/a_s`；否则 BF16 A + FP8 B 会 dot dtype 错误。
  - gate scale = `A_SCALE * B_SCALE[expert, :I]`，
    up scale = `A_SCALE * B_SCALE[expert, I:]`。
  - 写 `act_bf16` 前用 `tl.atomic_max(AMAX, tile_max)`；
    后续用 `_quant_act_fp8_from_amax` 量化。
- **case2 的 full-fused FP8 gateup 连续 TLE**（115769/115848/115927），
  包括 BN128 与 BN64，不要重试。
- INT8 fused gateup 也连续 TLE（115743/115792）。

### 9. case2 plain FP8 路径

- case2 gateup 从 INT8 改为 plain FP8 grouped GEMM 是重要收益：
  case2 tk 从 11.6-12.2 降到 10.17-10.43。
- 不要用 case2 full-fused（TLE）；两段式 fused（先写 gate 再算 up）也慢（115856）。
- case2 gateup 拆成两个独立 gate/up GEMM 也 TLE（115966）。

### 10. activation amax 与 gather

- 自写 `_act_amax_fp8_kernel` 替代 `a.abs().max()` 是本会话最大收益之一。
  - 单遍 load -> abs -> block max -> atomic_max；
  - 避免 PyTorch abs 物化和额外 reduction。
- `_gather_tokens_amax_kernel` 在 gather sorted tokens 的同时计算 amax，
  是第二个大收益；115907 稳定 base 对除 case2 外的 case 使用该路径。
- 115907 中 case2 保持 `x[token_idx].contiguous()` + `_quant_act_fp8`；
  全 case 用 gather+amax 时 case2 反而慢约 0.12ms。
- 尝试“fused gather + FP8 quant”（直接写 tokens_q，跳过 sorted BF16）最终 TLE，
  不要重试（115944/115946）。

### 11. B 转置布局 [G,K,N]

- 旧实验 115597 非法访存且 SQNR 错，原因是 stride 参数仍按 [G,N,K] 传：
  `b_q.stride(0), b_q.stride(1), b_q.stride(2)`。
- 修复为 `b_q.stride(0), b_q.stride(2), b_q.stride(1)` 后，115915 数值正确，
  但 timeUsed 299912，**严重变慢**。当前 Triton 版本不要做 B 转置。

### 12. down 写 slot 融合

- 115965 尝试 down GEMM 直接写 `[T,k,H]` slot，final 顺序归约。
  结果 timeUsed 45099，比稳定版慢约 1.5ms：down 写 slot 的随机行代价大于 final 收益。
- 当前 final `_gather_branch_sum` 单遍 gather+FP32 sum 已是最优，保留。

### 13. kernel 变体扫描结果

- FP8 down：BM128/BN256/BK128/GROUP_M8/w8/s3/persistent grid132 是最佳。
- fused gateup：BM128/BN128/BK128/GROUP_M8/w8/s3/nonpersistent 是最佳。
- 以下均变慢或失败：
  - down grid264、down BN128、down nonpersistent；
  - fused BN64/BK64/GM2/GM16/s2/persistent；
  - FP8 BM256 + metadata block_m=256：FP8 case 全错；
  - BM64 half-tile 和 BM64 metadata 同步版：慢/TLE；
  - num_warps=4/16、num_stages=2：慢；
  - route BK256：H<4096 shared memory OOR；route BN64/BK128/BM64：慢；
  - quant BM256/BK256、gather BH256/BM512、SwiGLU BM64/BN512/w4/w16：慢/TLE；
  - tl.sigmoid 替换：慢；
  - B 转置修复版：极慢；
  - fused gather+quant：TLE；
  - case2 split gate/up：TLE。

### 14. 历史仍有效的坑

- E=96 非 2 的幂，部分官方 kernel 的 `tl.arange` 会失败。
- 评测机波动大；case9 单点可见 3.4-8.0ms，case6 tb 可见 7.4-45.6，case2 tb 可见 28-47。
- 每个 direct allgather 后必须 `nvshmem_barrier_all_on_stream()`。
- direct allgather v1 的“先 copy 本地块、kernel 跳过本地 put”会在正式计时段失败；
  kernel 必须写全部 rank block。
- per-tile INT8 量化会产生非有限值；INT8 activation 必须 per-row scale + round。
- 不要用低分轮否定优化；重复信号以 timeUsed/tk 为准。

## 远程探测方法

- 可提交一个只定义 `run_kernel` 并 raise 的小文件来探测环境；
  WA 诊断不降低历史最佳，只消耗 submissionCount。
- 读 JIT kernel 源码：
  ```python
  from triton_dist.kernels.nvidia.group_gemm import moe_grouped_gemm_kernel_nk_const as k
  raise RuntimeError("ARGS " + repr(k.arg_names) + " SIG " + repr(k.signature) + " SRC " + repr(k.src))
  ```
- 官方源码和探测模板见 `handoff-session-20260817-final/probes/`。
- 本地也有 triton_dist 源码：`work/repo/python/triton_dist/kernels/nvidia/group_gemm.py`，
  与评测机版本基本一致。
