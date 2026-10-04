# XPUOJ P1 MegaMoE 交接文档

> 本文是 DeepSeek Harness 会话 `session-1925ec69-819a-4bb6-81be-c64eb2761293` 的独立交接文档，生成于 2026-08-16。
> 项目内原有的 `HANDOFF.md` 是累积文档：它既包含本轮开始前的状态，也被本轮会话多次追加，且顶部仍有早期状态残留。不要用旧文档顶部的 `kernel.py=114137` 覆盖本文记录的最终状态。
> 本文依据完整导出会话、当前源码哈希、提交日志和 benchmark 文件交叉整理，不包含 `.secrets` 中的任何凭据。

## 1. 最终状态

- 工作目录：`/home/sakimi26/xpuoj-p1`
- 比赛 ID：13
- Problem order：1
- 语言：`triton-dist`
- 当前 canonical：`p1/kernel.py`
- Canonical 归档副本：`p1/kernel_reduceout_c128.py`
- 两者 SHA-256：`eb4efb0bddcef4a65cde1f9490b589a39428a85fdac240543042ba56840fff42`
- 对应 submission：`114382`
- 结果：Accepted，`displayScore=51.08`
- 证据：`logs/submit_reduceout_c128.log`
- `timeUsed=117647`
- `memoryUsed=3223448`
- 本轮最后有效项目修改来自 Turn 9；Turn 10 至 Turn 13 没有产生新的源码或实验结果。

本轮开始前的真实状态来自会话第一次读取旧交接文档时的内容：

- 约 225 次尝试
- 当时 canonical 对应 submission `113512`
- 提交详情/榜单记录为 46.58
- 主架构仍以 NCCL variable-size A2A 和 all-gather 路径为主

本轮结束后：

- 平台/API 在会话中报告尝试计数 332
- 本地日志能证明 331 个唯一 submission ID；另有一个提交前 `FileNotFoundError` 日志
- Canonical 从 46.58 推进到 submission `114382` 的 51.08
- 排行榜接口曾长期返回缓存的 `113512/46.58`，但该观察需要在线重新运行 `best_score.py` 才能确认当前是否仍然成立

## 2. Canonical 架构

`p1/kernel.py` 的当前策略如下：

| Shape/path | 实现 |
|---|---|
| case4/11/12 | 预热 all-gather 全量专家权重，replicated 本地 grouped GEMM |
| 其他 `topk<=4` | NVSHMEM direct token dispatch |
| E8/E96 combine | NCCL variable-size 回传 |
| E32 combine | BF16 dense partial + `reduce_scatter_tensor` |
| `topk>4` | all-gather hidden/route + local grouped GEMM + BF16 partial + reduce-scatter |
| case5 route | ids 与 weights 打包为 int64，只进行一次 route all-gather |
| 最终归并 | `sum(dim=1, dtype=torch.float32)` |
| reduce-scatter | output 连续时直接写 output，避免中间 `[T,H]` buffer/copy |
| direct dispatch | 大 payload 使用 128 chunks、8 warps；其他情况使用 8 chunks |
| route GEMM | 自写 Triton BF16 GEMM，小专家数使用 `BLOCK_N=16/32/64` 特化 |
| metadata | 固定 `num_sms=32`，自建 row metadata，热路径无 `.item()` |

关键代码位置以 `p1/kernel.py` 为准：

- Direct dispatch 参数：约第 60-71 行
- Metadata 构造：约第 88-132 行
- Route GEMM：约第 224-252 行
- E8/E96 NCCL variable combine：约第 400-415 行
- E32 dense reduce-scatter：约第 416-434 行
- case4 replicated：约第 546-553 行
- case11/12 replicated：约第 555-561 行
- case5 packed route：约第 592-608 行
- 大 shape BF16 reduce-scatter：约第 691-705 行

## 3. 本轮演进时间线

### 3.1 Turn 1：修复基础热路径，46.58 到 48.58

首先确认 `topk<=4` 路径重复计算 routing：`run_kernel` 已计算一次，旧 `_run_kernel_a2a` 内又计算一次。修复后继续加入：

- case11 replicated
- case12 replicated
- stable-sort 后 view-sum，替代 `[T,k,H]` FP32 `branch_sum`
- metadata 热路径去除 `.item()`
- route GEMM 小 N 特化
- all-gather BF16 reduce 阈值从 20M 降到 10M

关键提交：

| Submission | 状态 | 分数 | 说明 |
|---:|---|---:|---|
| 114007 | Accepted | 43.83 | 修复重复 routing |
| 114015 | Accepted | 48.25 | view-sum + metadata 去 `.item()` |
| 114017 | Accepted | 48.33 | route 小 N 特化 |
| 114019 | Accepted | 48.58 | BF16 阈值降到 10M |

### 3.2 Turn 2：NVSHMEM direct 与 replicated 路径

本阶段将 `topk<=4` token dispatch 迁移到 NVSHMEM direct put，并探索 E8/E32/E96 的不同 combine 方式，同时加入 case4 replicated、k>4 prep 向量化与 BF16 partial。

关键提交：

| Submission | 状态 | 分数 | 说明 |
|---:|---|---:|---|
| 114061 | Accepted | 48.00 | Direct 架构可运行 |
| 114067 | Accepted | 48.92 | NVSHMEM direct 全路径 |
| 114068 | Accepted | 49.25 | 加 case4 replicated |
| 114082 | Accepted | 50.92 | 阶段峰值，但 case1/2 波动较大 |
| 114134 | Accepted | 49.92 | Hybrid combine 稳健候选 |
| 114137 | Accepted | 48.67 | E32 dense 固定策略；旧文档顶部所指的历史版本 |

### 3.3 Turn 3-4：FP8 第二条路线

已确认：

- 官方 `moe_grouped_gemm` 没有 `out=` 参数
- FP8 输入时输出 dtype 跟随 FP8，值容易饱和到 448，后乘 scale 无法恢复已经丢失的信息
- 自研 FP8 grouped GEMM 可以输出 BF16，局部 GEMM 有收益
- Triton BF16→FP8 cast 优化到约 0.15ms
- BF16 `.abs().max()` 路径将 amax 降到约 0.3ms
- BN256 是已测配置中最好的一组

局部 phase 数据：

| Phase | FP8 | BF16 official |
|---|---:|---:|
| case1 gateup | 约 5.14ms | 约 7.45ms |
| case1 down | 约 1.96ms | 约 3.06ms |

但端到端仍未超过 BF16 canonical：

- case1：BF16 约 16.5ms，FP8 both 约 17.5-18.1ms
- case2：BF16 约 26.0ms，FP8 both 约 27.9-30.6ms
- FP8 gateup-only 与 down-only 均未胜
- BK64、BK256、stages=4、warps=16 更慢或 OOM
- FP8 token dispatch SQNR 约 28.2-28.4dB，但总分约 48.5
- E96 FP8 token 更慢，且 FP8 token buffer 与 BF16 down buffer 并存可能触发 NVSHMEM OOM

FP8 原型应保留给未来 4×H800 + NCU/SASS 环境，不宜继续无 profiler 盲扫。

### 3.4 Turn 5-6：稳定组合与低风险优化

`kernel_combo_v5.py` 固化了以下组合：

- Direct dispatch 64 chunks / 8 warps
- E8/E96 使用 NCCL variable combine
- E32 使用 dense reduce
- case4/11/12 replicated
- k>4 使用 BF16 partial + vectorized prep

随后加入：

- `sum(dim=1, dtype=torch.float32)`，避免先 `.float()` 物化大张量
- case5 route ids/weights 打包实验
- A2A weight/meta/local/src 打包实验
- 多路径动态 autotune 实验

关键结果：

| Submission | 状态 | 分数 | 说明 |
|---:|---|---:|---|
| 114246 | Accepted | 49.42 | combo_v5 |
| 114285 | TimeLimitExceeded | - | 6 路 autotune 超过 500 秒总限制 |
| 114301 | Accepted | 49.50 | `sum(dtype=float32)` |
| 114302 | Accepted | 48.92 | route idx cache，无收益 |
| 114309 | Accepted | 49.00 | all-gather route 打包 |
| 114319 | Accepted | 49.50 | A2A weight/meta 打包，打平 |

### 3.5 Turn 7-8：得到最终 51.08

Turn 7 仅对 case5 `(E=64,I=2560,k=8)` 打包 ids+weights，并让 BF16 partial/E32 dense combine 的 reduce-scatter 在 output 连续时直接写 output。

| Submission | 状态 | 分数 | 说明 |
|---:|---|---:|---|
| 114348 | Accepted | 50.00 | case5 packed route |
| 114354 | Accepted | 50.58 | reduce-scatter 直接写 output |

Turn 8 扫描 direct dispatch chunk/warp 配置：

| Submission | 状态 | 分数 | 说明 |
|---:|---|---:|---|
| 114382 | Accepted | 51.08 | 128 chunks / 8 warps，最终 canonical |
| 114386 | Accepted | 48.50 | 256 chunks |
| 114391 | Accepted | 49.42 | 128 chunks / 4 warps |
| 114396 | Accepted | 48.50 | 96 chunks |
| 114401 | Accepted | 49.75 | route idx cache |
| 114406 | Accepted | 50.75 | 全 E weight/meta 打包 |
| 114409 | Accepted | 50.75 | 仅 E8 打包；case2 有改善但总分未胜 |

会话中解析到的 114382 各点如下；原始逐点 detail 未单独保存在 `logs/`，总分与状态可由 `logs/submit_reduceout_c128.log` 独立确认：

| Case | tk(ms) | 单点分 |
|---:|---:|---:|
| 1 | 15.48 | 56 |
| 2 | 25.73 | 52 |
| 3 | 8.29 | 45 |
| 4 | 5.27 | 48 |
| 5 | 10.64 | 54 |
| 6 | 7.00 | 53 |
| 7 | 11.22 | 47 |
| 8 | 9.06 | 44 |
| 9 | 7.09 | 54 |
| 10 | 5.90 | 53 |
| 11 | 4.55 | 55 |
| 12 | 7.41 | 52 |

### 3.6 Turn 9：Phase profile 与最终证伪

`p1/kernel_c128_diag.py` 是故意抛异常以获取 phase 数据的诊断文件，不能作为正式提交。Submission 114465 的 WrongAnswer 是预期行为。

会话中记录的 phase 摘要：

| 路径 | 主要阶段 |
|---|---|
| A2A case1 | route 0.55 / dispatch 2.5 / gateup 7.1 / activation 0.54 / down 2.9 / final 4.2ms |
| A2A case2 | gateup 12.9 / down 5.1ms |
| A2A case7 | dispatch 3.2 / gateup 2.8 / down 1.3 / final 4.2ms |
| All-gather case5 | gather 1.6 / select+sort 3.7 / gateup 5.6 / down+reduce 3.9ms |
| All-gather case6 | gather 1.7 / select+sort 2.3 / gateup 1.4 / down+reduce 4.5ms |
| Replicated case11/12 | 整路径约 4.6 / 6.2ms |

这些 phase 数值存在于会话和旧累积 `HANDOFF.md` 的摘要中，`logs/submit_c128_diag.log` 只保存 WrongAnswer 终态，未保存原始异常 detail。

最后一批实验：

| Submission | 状态 | 分数 | 结论 |
|---:|---|---:|---|
| 114465 | WrongAnswer | 0 | 故意 diag raise |
| 114468 | WrongAnswer | 32.08 | `scatter_reduce_` 被 TensorGuard 禁止 |
| 114470 | Accepted | 49.08 | dst 排序后 index_add，退化 |
| 114472 | Accepted | 48.33 | metadata zeros→empty，退化 |
| 114487 | Accepted | 45.33 | E64 `num_sms=64`，明显退化 |
| 114490 | WrongAnswer | 40.00 | uint32 weight bit-cast 失败 |

上述文件均未覆盖 canonical `p1/kernel.py`。

## 4. 重要文件

| 文件 | 用途 |
|---|---|
| `p1/kernel.py` | 当前 canonical，精确匹配 114382 |
| `p1/kernel_reduceout_c128.py` | 114382 的命名归档副本 |
| `p1/kernel_c128_packweight_e8.py` | 114409，50.75；当前最接近但未晋升的候选 |
| `p1/kernel_c128_packweight.py` | 114406，50.75；全 E 打包，内存更高 |
| `p1/kernel_sumdtype_packroute_case5_reduceout.py` | 114354，50.58 |
| `p1/kernel_sumdtype_packroute_case5.py` | 114348，50.00 |
| `p1/kernel_combo_v5_sumdtype.py` | 114301，49.50 |
| `p1/kernel_combo_v5.py` | 114246，49.42 |
| `p1/kernel_directfull_v2_case4_hybridcombine.py` | 114134，49.92 的早期 hybrid 版本 |
| `p1/kernel_c128_diag.py` | 诊断专用，会故意失败，不可正式提交 |
| `p1/kernel_scatter_reduce.py` | 114468，TensorGuard 禁止 |
| `p1/kernel_sortdst_indexadd.py` | 114470，49.08 |
| `p1/kernel_meta_empty.py` | 114472，48.33 |
| `p1/kernel_ag_sms64_e64.py` | 114487，45.33 |
| `p1/kernel_c128_packweight_e8_uint.py` | 114490，WrongAnswer |
| `p1/kernel_fp8_repl_proto*.py` | BF16-output FP8 grouped GEMM 原型 |
| `p1/kernel_fp8_repl_tritonquant*.py` | Triton activation quant 原型 |
| `p1/kernel_fp8_tokens_e8_v2.py` | FP8 token dispatch 原型 |
| `scripts/xpuoj_api.py` | 登录、提交和查询 API 封装 |
| `scripts/submit.py` | 提交并轮询结果 |
| `scripts/best_score.py` | 查询 scoreboard cache，不等于 submission detail |
| `logs/` | 提交终态与诊断日志 |
| `benchmarks/submission_history.md` | 只更新到 114137，已过时 |
| `benchmarks/submission_details_cache.json` | 仅 75 条、截至 114011，已过时 |
| `.secrets/xpuoj.json` | 登录凭据；保持 600，不得打印、复制到交接或提交 |

## 5. 已证伪或不应盲目重复

- 多路径动态 autotune：超过 500 秒总时限
- Python/CUDA stream overlap：case5 约从 10.9ms 退化到 15.8ms
- `scatter_reduce_`：TensorGuard 禁止
- metadata `zeros -> empty`：114472 仅 48.33
- E64 grouped GEMM `num_sms=64`：114487 仅 45.33
- dst 先排序再 `index_add_`：114470 仅 49.08
- uint32 weight bit-cast：114490 WrongAnswer
- route idx cache：114401 仅 49.75，历史多个 idxcache 版本也无稳定收益
- chunks=256/96：均 48.50
- chunks=128/warps=4：49.42
- 全 E packweight：50.75，未胜 canonical 且内存更高
- E8-only packweight：50.75；case2 有信号，但不是新 canonical
- case1/2/E96 replicated：不划算
- case3 all-gather：修复重复 routing 后不再占优
- E96 FP8 token：更慢且可能 OOM
- FP8 BK64/BK256/stages4/warps16：更慢或 OOM
- 官方 FP8 grouped GEMM 直接作为 BF16 替代：输出饱和问题不可接受
- 随机 grouped-GEMM 参数扫描：无 profiler 时信息收益很低
- 重复提交相同代码碰运气：51.08 是一次线上观测，不是已证明的稳定分布

## 6. 环境与沙箱约束

评测环境已知：

- 4×H800
- NVSHMEM 已初始化
- `triton_dist`
- 有 `moe_grouped_gemm`
- 有 `build_block_row_idx_info_kernel`
- 没有 `prepare_moe_metadata_using_kernel`
- E96 时 `Ep=24`，不是 2 的幂，部分 kernel 不支持

已遇到的约束：

- 禁止 `torch.matmul`
- 函数式 `torch.argsort` 禁止；tensor `.argsort(stable=True)` 可用
- 函数式 `torch.repeat_interleave` 禁止；tensor method 可用
- 函数式 `torch.cumsum` 禁止；tensor method 可用
- 禁止 `tensor.clamp_min`
- `torch.topk` 的 k 必须位置传参
- 禁止 `tensor.data_ptr()`
- 禁止导入 `nvshmem`
- 全局非字面量赋值会 validation 失败
- `scatter_reduce_` 被 TensorGuard 禁止
- `torch.cuda.Event` 与 `torch.cuda.Stream` 可用，但本轮 overlap 实验没有收益

## 7. 常用操作

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/kernel.py
python scripts/submit.py p1/kernel.py --poll --interval 10 --timeout 1800
python scripts/best_score.py
```

注意：

- `submit.py` 每次只查询最近 5 个提交；并发提交可能让目标 ID 跌出窗口并持续轮询到 timeout
- `best_score.py` 查询 scoreboard cache，不能替代 submission detail
- Submission detail API 是 `POST /api/submission/getSubmissionDetail`
- `userOutput` 中的 `OJRESULT v1 <hash> <base64json>` 可解码得到 `tb_time_ms` 等原始数据
- 当前没有自动维护完整 `submission_details_cache.json` 与 `submission_history.md` 的可靠流程
- 提交会消耗平台尝试次数，先保存候选源码和日志再提交

## 8. 下一步优先级

1. 保持 `p1/kernel.py` 与 `p1/kernel_reduceout_c128.py` 不变，先校验 SHA-256。
2. 优先获取 4×H800 调试环境；无本地 profiler 时继续远程盲扫边际收益很低。
3. 若只能远程提交，唯一相对合理的现成候选是 `p1/kernel_c128_packweight_e8.py`，但 114409 总分仍低于 canonical。
4. 主攻 case1/2 的 gateup/down，开发真正高吞吐且输出 BF16 的 FP8 grouped GEMM。
5. 使用 NCU/SASS 分析 custom FP8 kernel 的 B tile 复用、K-loop pipeline 与 WGMMA。
6. 尝试在 kernel/持久化层做 direct dispatch 与专家计算流水，而不是 Python stream overlap。
7. 优化 case5 的 select+sort（约 3.7ms）和 gateup（约 5.6ms）。
8. 优化 all-gather path 的 down + index_add + reduce（约 3-4.5ms）。
9. 不再重复第 5 节已证伪的方向。
10. 若关注排行榜，在线重跑 `best_score.py` 并直接查询 114382 detail，区分榜单缓存与提交详情。

## 9. 会话结束原因

本会话共有 13 个 turn、397 条 assistant message、391 次 bash tool call。有效实验集中在 Turn 1-9。

- Turn 9 内发生过 300 秒 stream idle timeout 和多次 `TRANSPORT`，自动重试后最终完成
- Turn 10 因 DeepSeek API `TRANSPORT`/`TIMEOUT` 结束，没有项目修改
- Turn 11 连续出现 300 秒 stream idle timeout，随后被后续操作结束
- Turn 12 没有 assistant 输出或工具调用
- Turn 13 将 reasoning effort 改为 high 后请求仍在生成前被结束，没有生成交接文件
- 最后一次成功 usage 已包含约 525K cache-read tokens；minimal preset 没有 compaction，超长上下文和当时不稳定的 2.4 GHz 网络共同导致后续请求脆弱

因此，新的接手会话应直接阅读本文和当前落盘文件，不应导入旧会话的全部 525K token 历史。

## 10. 接手检查清单

```bash
cd /home/sakimi26/xpuoj-p1
sha256sum p1/kernel.py p1/kernel_reduceout_c128.py
python -m py_compile p1/kernel.py
```

预期两个 SHA-256 均为：

```text
eb4efb0bddcef4a65cde1f9490b589a39428a85fdac240543042ba56840fff42
```

随后阅读：

1. 本文
2. `README.md`
3. `p1/kernel.py`
4. `logs/submit_reduceout_c128.log`
5. 若继续 FP8，再阅读 `p1/kernel_c128_packweight_e8.py` 与相关 FP8 原型

不要先阅读或重新提交全部历史候选；先确认 canonical、目标瓶颈和剩余尝试预算。
