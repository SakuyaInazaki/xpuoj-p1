# P1 MegaMoE 优化指引：v890 → raw 90 / net 80

> **历史版本提示（2026-09-30）：** EP2 metadata 修复和 single-peer 验证已完成，后续 BM256 也有正式负结果。新的优先级、71 个 SID 恢复台账和 slot DN 融合方案见 [9 月 30 日指引](OPTIMIZATION_GUIDE_2026-09-30.md)。下文保留原日期证据，不再作为当前待办。

日期：2026-09-29（Asia/Shanghai）。截止：**2026-10-01 23:59**，用户已确认。工作区：`/Users/sakimi/Desktop/xpuoj-p1`。

**当前正式排行榜成绩是 75.17，最佳提交为 [149493](https://xpuoj.com/contest/13/submissions/149493)，该提交 raw 85.17；距离目标 raw 90 / net 80 均差 4.83 分。** 本文的固定基准性能模型用于选择实验，不能替代、下调或重新定义这个榜面成绩。

**第一优先级：修正 EP2 原型的 metadata 接线错误，再验证真正的单 peer 通信能否保留计算收益。** 这次在 21 份历史候选中定位了同一类错误，已经提供只改两处参数的候选、补丁和 CPU 反例。静态 DN 尺度和 PAD_DELTA 融合已进入生产，不能重复立项。纯排序/metadata 微调停止。I 维两卡切分也有旧负结果，只允许在显著减少通信和辅助开销的前提下做有界复查。

本次完成：题面与补充讨论审阅、v890 源码及历史记录审计、34 个已有 SID 的只读恢复、349 条有效逐案评分复算、EP2 参数修复与 CPU 验证、两卡 I 维切分的布局验证。**本次新正式提交 0 次、custom 0 次，没有声称获得新的 GPU 提速；生产 `p1/kernel.py` 未改。** 用户已禁止使用 `deepseek-brainstorm` 技能，要求已写入 [AGENTS.md](AGENTS.md)。本指引由直接源码分析形成，没有外部脑暴结果。

## 1. 执行者先看这张表

| 优先级 | 工作 | 首个交付物 | 继续条件 | 停止条件 |
|---|---|---|---|---|
| P0 | 冻结 v890、修正路径/评分证据、验证缓存和任意调用语义 | manifest、逐案表、目标家族调用路径表 | 所有候选始终完整计算本次输入 | 不能用只在某个调用窗口成立的结果晋升 |
| A1，立即 | 修复 EP2 MD/DN 的 per-tile 起点参数 | 本文附带的两处修复 + 最小单卡 MD/DN 对照 | oracle、finite、覆盖、重复性通过 | 两次明确针对错误的诊断仍不能消歧则暂停 GPU 队列 |
| A2，主线 | EP2 固定容量索引合并 + 单 peer FP8 dispatch / partial return | c9 端到端候选，其他案仍用 v890 | 至少 5% 全案降时信号，目标 8%–15% 以上 | 计算节省付不起通信/归并/等待；不以修好挂死等同成功 |
| B，条件备用 | 重新审计 I 维两卡切分 TP2，使用本地路由、FP8 peer 通信、tiled 权重 | 与 A 使用同一通信成本的计算代理 | 计算侧明显优于 EP2 或避开其已证实瓶颈 | 原 145084 的 BF16 world all-gather + world reduce-scatter 原样方案不重跑 |
| C，高风险小预算 | c11 按专家对分组的 DN + 确定归并 | 包含 pair metadata 的 ACT→output 对照 | 子链至少 10%，全案有望至少 3% | padding、ACT gather、B 重读使总子链无收益 |
| D，收口 | 合并独立胜者，更新固定基准收益，再做完整复测 | 最终单文件、SID、SHA、回退路径 | 正确性和收益同时成立 | 截止前停止加入新通信协议 |

推荐把 A、C 分给两个 coding agent；B 先由 A 的执行者做成本审计，有计算证据再实现。单独指定一个平台操作者串行提交。本文不要求创建新的 Codex 聊天，也未代替用户派发其他 agent。

## 2. 当前真实起点与证据等级

### 2.1 当前状态

只读核验时间：**2026-09-29 02:55:58 UTC+8**。该时间之后的队列和榜单需要执行者重新查询。

| 项目 | 核实结果 |
|---|---|
| 当前榜分 / 最佳 SID | **75.17 / 149493**，该 SID raw 85.17 |
| 账号综合名次 | 4，三题总分 246.09；不是 P1 单题名次 |
| P1 提交数 | 3370；当前新提交按 10 分罚分上限计算 |
| 最新 P1 SID | 150945，Accepted；它是 v904 候选，**不是生产 v890** |
| 最近返回的提交 | 接口实际返回 10 项，未见 Pending/Running；这不证明更早记录或其他会话永远无在途 |
| P1 分布式 custom | `available=false`；不要计划依赖它 |
| 生产 | `p1/kernel.py` = `p1/kernel_v890_fused_delta.py` |
| 生产 SHA-256 | `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc` |
| 上一生产 | v882，SHA `f54ff731cb5ae8193e992c9cddf42f0390b14bc6add1d18ced300db9d956af69` |
| 更早回退 | v844，SHA `d1f1e0697c3e99a86f3d34bea6509755be6f70470a2373ec0bd355819c671f9e` |

证据：[平台快照](reports/2026-09-29-platform-readonly.json)、[逐 SID 原始日志与源码哈希](reports/2026-09-29-submission-audit.json)、[源码和积分审计](reports/2026-09-29-v890-audit.json)。现有目录没有 Git 仓库；本轮用文件 SHA、差异补丁和独立候选保护生产，不假设有可回滚的 Git commit。

### 2.2 三种数字必须分开

1. **榜面分**：官方当前 75.17，目标 80，差 4.83。这是竞赛成绩。
2. **某次提交的 raw**：来自该次平台 `displayScore`，同时受该次 `tk`、`tb` 等影响。
3. **固定 tb 模型**：冻结一组 tb，用可比较的 tk 估算结构收益。这是实验决策工具，不是平台重新判定的成绩。

例如，完全相同的 v890：SID 150917 的 c4 为 `tk=0.803, tb=14.970, q=94`；SID 150943 为 `tk=0.779, tb=5.005, q=86`。后者算子更快，却因 baseline 时间不同得到更低单案分数。只看 raw 排名会选错候选。

最佳 SID 149493 的已存详情中，c11/c12 的 tk 是 `0.325/0.000 ms`，与当前正常样本相差很大。这解释了为什么不能把其每案 tk 当作后续性能基线；**不改变平台已经显示的 75.17，也不据此宣布该成绩无效**。异常原因尚无结论。[已有详情](reports/2026-09-27-v844-cases.json)

### 2.3 标注约定

- **已验证**：源码可直接确定、CPU 检查通过，或有对应 OJ 记录；须说明是哪一层。
- **推导**：由尺寸、数学关系或成本公式得出，尚非测得性能。
- **待测**：需要 custom/P1 证明的数值、编译、通信或收益假设。

Accepted 不能证明比基线快；CPU 索引检查不能证明 GPU 不挂；单卡 custom 不能认证四卡协议；有限几次正式 AC 不能替代跨调用和缓存合同分析。

## 3. 题目、硬件与可用实验能力

每 rank 有本地 `X[T,H]` 和 `E/4` 个专家。共享 router 执行 BF16 matmul，结果先按 BF16 保存，再用 FP32 softmax/top-k/归一化。每个所选专家计算 `g=xGᵀ`、`u=xUᵀ`、`a=w·SiLU(g)·u`、`c=aDᵀ`，最终各分支用 FP32 合并并写 BF16 output。参考链在投影、激活输入和 down 输出处有 BF16 舍入，改变结合顺序或舍入位置需要重新验收。[题面](1-full.md)

硬要求是 SQNR≥22 dB、全部输出有限且完整写入、输入不变、相同输入独立调用逐字节确定。同一测试点权重与 topk 可预处理缓存；hidden_states 会在预热后更换。切换测试点时必须处理静态参数变化。主 GEMM 不允许用高层 torch 矩阵乘或官方融合 MoE 替代。

用户提供的 [addinfo.md](/Users/sakimi/Desktop/addinfo/addinfo.md) 已明确：

- P1 为 Triton-dist 3.4 / Triton 3.4；fork commit 未提供。不要把 P2/P3 的 3.6 行为当作 P1 的能力。
- 4×H800 80G SXM，默认 700 W，卡间拓扑均为 NV8。NV8 不等于实测应用带宽。
- 通信和 NVSHMEM 已初始化；允许从 `triton_dist.utils` 使用状态检查、对称 tensor 分配/释放和 stream barrier。不得重复 init/finalize；所有 rank 的集体分配顺序一致。
- 分布式设备操作要使用 `triton_dist.jit`，不是普通 `triton.jit`。
- 500 秒是一次 torchrun 批量评测的总预算，包含预热/JIT。答复写“十个点”，当前已有详情包含 12 个正式案；不能理解成每个新 kernel 都有 500 秒。
- 尚无确定可用的逐 kernel profile 入口；讨论中已有 Event 被沙箱拒绝的报告。不能把历史 Event 工具直接当作现行权限。

本轮打开了公开题目和讨论 URL，但浏览工具未取得可读正文；题目细节以项目保存的完整题面和用户补充材料为依据，线上状态以授权的只读 API 为依据。

官方 [Triton v3.4 Hopper Gluon 入口](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py) 仅暴露对应 TMA/mbarrier/fence 入口，没有可直接照搬最新版教程的显式 `warpgroup_mma`。Hopper 的 TMA/L2 结构可参考 [NVIDIA 架构指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)，不能据 H100 理论带宽宣布这台 H800 已达到或尚余多少带宽。

## 4. v890 已经做了什么

当前 12 个正式形状均进入 replicated：预热收集全专家 FP8 权重，每卡独立处理本卡 token，稳态没有 token dispatch/combine。它没有重复计算其他三个 rank 的 token。

```text
静态：全专家权重收集 → FP8/scale/布局预处理 → 缓存
动态：router → expert 排序/metadata → token FP8
      → MD：gate+up+SwiGLU+route weight+ACT FP8
      → DN：down+输出量化
      → 按原 token/slot 归并 → BF16 output
```

相对 v844，v890 已增加：

| 机制 | 当前覆盖 | 注意事项 |
|---|---|---|
| 静态 L1 界 DN 输出尺度 | c3/c4/c6/c8/c11/c12 | 当前实际系数是 **1.02×L**，不是早期笔记的 1.1 或 4 |
| offsets 融合 PAD_DELTA | 八个 padded 案 | 已消掉独立 pad_delta launch，不再提一次相同融合 |
| padded DN TMA + direct INV_PAD | c3–c8/c11/c12 | c5/c7 仍用动态输出尺度 |
| compact FP8 DN + packed-key 排序 | c9/c10 | 不要再把 packed sort 或 tile 连续权重当新增方向 |
| compact DN | c1/c2 | 主计算链已有多轮参数、布局、store 和累加方式探索 |

当前静态尺度为 `C[e,p]=1.02*max_h Σ_i |Dq[e,h,i]*Ds[e,h]|`，`folded=Ds/C`；DN 写 `FP8(acc*folded)` 和 `max(As*C,1e-12)`。这是完整现役数值链的一部分，不能在新候选中无意退回旧版本。

**源码导航，行号绑定本页 SHA：**

| 位置 | 行号 | 用途 |
|---|---:|---|
| `_run_replicated` | 6024 | 正式主链和家族分发 |
| `run_kernel` | 6529 | 全局 `_CALLN` / `_FL` / `_GA` |
| `_get_full_fp8_weights_lowmem_tiled` | 3790 | c9/c10 tiled 权重及 scale |
| `_fgs_tma1_kernel_gq_tiled` | 1639 | EP2/TP2 优先复用的 MD |
| `_fgs_tma1_host_tiled` | 1813 | MD 参数合同的正确参考 |
| `_prepare_moe_metadata` | 4118 | **最后生效**的 metadata 定义 |
| `_dn_tma2_f8_tiled_kernel` / `_host_tiled` | 4599 / 4657 | c9/c10 DN |
| `_gather_branch_sum_f8_kernel` | 4678 | 原 slot 顺序归并 |
| `_counting_sort_order_invpad` | 6875 | padded 排序 |
| `_get_down_static_scale` / `_pad_static_kernel` | 7122 / 7151 | v890 静态尺度 |
| `_csort_offsets_delta_kernel` | 7222 | 已实现的 PAD_DELTA 融合 |

文件中有重复定义，必须按 Python 最后绑定查调用。不要因函数名带 `split_cum` 就假设它是同一种数组。

## 5. 积分目标与工作量：决定预算的依据

### 5.1 本轮重算的范围

找到 7 个源码 SHA 精确等于 v890 的已有正式提交：149930、149935、150917、150926、150934、150942、150943。选最近五个 150917/150926/150934/150942/150943 的逐案 tk、tb 中位数作**描述性固定基准**。没有把它们说成一个新 A/B 实验。

本次恢复记录中，349 条状态 Accepted、tk>0、th=0 的逐案记录均满足：

```text
q_i = floor(100 * tb_i / (tb_i + tk_i))
raw = Σ q_i / 12
本账号新提交 net = raw - 10
```

公式只用于已观察到的 th=0 区间。官方说更快区间可使用对数计分、分数可超过 100；这里没有把 100 当作赛事永远不变的硬上限。

### 5.2 逐案表

所有 T、FLOPs 和字节数均为**每 GPU**。`M=T*k`，有用专家计算量约 `6MHI`，FP8 全专家权重容量约 `3EHI` 字节。容量不等于实测 HBM 流量；padding 和重复读会增加工作。

| 案 | T / H / E / I / k | tk 中位 ms | 固定 tb ms | 模型 q | 下一档需降时 | 最低已记 SQNR |
|---|---|---:|---:|---:|---:|---:|
| c1 | 16384 / 4096 / 8 / 8192 / 2 | 4.620 | 17.701 | 79 | 4.22% | 23.12 |
| c2 | 16384 / 4096 / 8 / 14336 / 2 | 7.839 | 28.748 | 78 | 2.51% | 23.14 |
| c3 | 16384 / 2048 / 32 / 2048 / 4 | 1.343 | 6.946 | 83 | 1.49% | 22.70 |
| c4 | 16384 / 2048 / 32 / 1024 / 4 | 0.784 | 5.011 | 86 | 4.49% | 22.98 |
| c5 | 8192 / 3584 / 64 / 2560 / 8 | 2.759 | 12.119 | 81 | 3.58% | 23.13 |
| c6 | 8192 / 3584 / 64 / 1024 / 8 | 1.247 | 7.489 | 85 | 2.23% | 22.93 |
| c7 | 16384 / 4096 / 96 / 2048 / 3 | 2.181 | 10.272 | 82 | 3.53% | 23.12 |
| c8 | 16384 / 4096 / 96 / 1024 / 3 | 1.232 | 7.164 | 85 | 5.34% | 22.90 |
| c9 | 4096 / 4096 / 256 / 2048 / 8 | 2.472 | 7.737 | 75 | 1.16% | 23.13 |
| c10 | 4096 / 4096 / 256 / 1536 / 8 | 1.881 | 6.690 | 78 | 5.46% | 23.11 |
| c11 | 65536 / 1024 / 32 / 1024 / 2 | 0.874 | 5.667 | 86 | 3.11% | 23.14 |
| c12 | 65536 / 1024 / 32 / 2048 / 2 | 1.472 | 8.153 | 84 | 2.26% | 22.90 |

合成中位耗时总和为 28.704 ms。c1/c2 占约 43.4%，但积分中每案等权。c9/c10 权重容量分别 6/4.5 GiB，平均只有 128 rows/expert；c1/c2 则是 0.75/1.3125 GiB、4096 rows/expert。两类问题不应使用同一优化策略。

### 5.3 raw 90 的量级

下表全部冻结本节 tb，只反映条件情景；**不是当前榜面成绩预测**。

| 假设 | 模型 raw | 模型 net | 相对该模型基线的增量 |
|---|---:|---:|---:|
| v890 描述性基准 | 81.83 | 71.83 | — |
| 仅 c9/c10 各降 30% | 82.75 | 72.75 | +0.92 |
| 仅 c9/c10 各降 50% | 83.50 | 73.50 | +1.67 |
| c9/c10 降 30%，其他案降 10% | 83.83 | 73.83 | +2.00 |
| 全案各降 30% | 86.50 | 76.50 | +4.67 |
| 全案各降 50% | 89.83 | 79.83 | +8.00 |
| c1/c2 不变，其余全降 50% | 88.25 | 78.25 | +6.42 |

同百分比降时模型达到 raw 90 的门槛约 **51.20%**。这不是“每个案必须降 51.20%”，也不是官方未来 tb 必须固定的假设；它说明当前可解释的结构性能距离目标仍很远。极端地令 c1/c2/c9/c10 的 tk 全为零、其余不动，此模型也只有 raw 89.33。这个极限计算仅针对当前 th=0 分支，不用于否定官方的对数计分。

若硬要求每案都到 q=90，则 c1/c2 需要 1.967/3.194 ms，对应有用整链吞吐约 3354/3614 TFLOPS；当前是 1428/1473 TFLOPS。没有相应能力与新算法证据，不能靠“把寄存器压低一点”承诺这种幅度。可以让不同案承担不同积分，但必须给出整题总账。

**判断：保留 raw 90 / net 80 作为冲刺目标；目前没有证据支持承诺达到。优先追求可复现的大机制收益，避免耗尽最后三天在 0.3% 的参数噪声上。** 固定 tb 的改善也不能直接加到 75.17 上宣称新榜分。

复算：[audit_v890.py](reports/audit_v890.py)、[逐案阈值](reports/2026-09-29-v890-thresholds.csv)、[情景表](reports/2026-09-29-v890-scenarios.csv)。

## 6. A1：EP2 失败中已定位的错误，先修这里

### 6.1 为什么旧诊断没有排除这个错误

9 月 27 日记录表明，pair 通信、过滤排序和 metadata 构建可分别运行，进入 MD 后失败，于是停在“需要 rank2 stderr / 单卡复现”的判断。[旧记录](experiments/2026-09-27/notes/r1_pair_ep2_probe.md)

本次检查发现，从 v858 到 v896 的多份候选，把：

```python
meta_expert_ids, counts, meta_split_cum,
```

传给 MD 的 `expert_ids, split_size, split_size_cum`，DN 也同样传错。真正需要的是：

```python
meta_expert_ids, counts, meta_tile_split,
```

`_prepare_moe_metadata` 最后生效定义的返回合同为：

| 返回位置 | 原变量名 | 实际语义 | 合法索引 |
|---|---|---|---|
| 0 | `meta_split_cum` | 每 expert 的 compact row 起点 | `expert`，长度 E |
| 1 | `meta_expert_ids` | tile→expert | `pid_m` |
| 2 | `meta_tile_split` | 每个 tile 所属 expert 的 compact row 起点 | **`pid_m`** |
| 3 | `meta_tile_num` | 每 tile 所属 expert 的 tile 数 | `pid_m` |
| 4 | `meta_tile_num_cum` | 每 tile 所属 expert 的 inclusive tile prefix | `pid_m` |
| 5 | `num_tiles_total` | 总 tile 数 | 标量 |

MD/DN 消费者都执行 `row_begin = tl.load(split_size_cum + pid_m)`。因此把位置 0 当作位置 2，并不是变量名不规范，而是实际地址错误。正确 replicated host 恰好传的是位置 2。

证明链：[生产 metadata](p1/kernel.py#L4118)、[本地 group_gemm 构建实现](work/repo/python/triton_dist/kernels/nvidia/group_gemm.py#L40)、[旧 v866 接线](experiments/2026-09-27/candidates/v866_c9c10_pair_ep2_halfweights.py#L7180)、[MD 读取](p1/kernel.py#L1664)。本地依赖版本不用于声称远端完全相同；生产显式返回结构和消费者索引已经足以定位参数类型不匹配。

### 6.2 最小反例

```text
counts = [129, 1, 0, 256]，BM=128
每 expert 起点：       [0, 129, 130, 130]
每 tile 正确起点：     [0,   0, 129, 130, 130]
```

tile 1 属于 expert 0，起点应仍为 0；旧代码读到 129。tile 4 则已超过长度为 4 的 per-expert 数组。正常 EP2 的 128 experts、平均 256 rows/expert 同样会产生多于 128 个 tile。

旧 v896 只检查 `counts.sum()==n_f` 和总 tile 数范围；这两者都正确，也不能证明传给 MD 的数组正确。v895 在 MD 后 synchronize 挂住的现象与错误访问相容，但**静态证据不能证明它是线上全部失败的唯一原因**。

本轮 CPU 检查覆盖 30 种 counts、962,946 个有效 row，包括空 expert、全偏斜、127/128/129 边界和随机分布；修复后的映射逐行唯一覆盖、容量约束通过。扫描命中 21 份历史候选，不应逐份重投。

### 6.3 已准备的修复

- [最小补丁](experiments/2026-09-29/candidates/ep2_v866_metadata_fix.patch)：相对 v866，仅更换 MD、DN 两个参数。
- [修复候选](experiments/2026-09-29/candidates/ep2_v866_metadata_fix.py)：SHA `b7079e0a239d1dfb01b03802cce45f7459da050f9df620060bc3f660c58652a8`。
- [CPU/AST 验证脚本](reports/prove_ep2_metadata_20260929.py)、[检查结果及受影响位置](reports/2026-09-29-ep2-metadata-proof.json)。

**这个候选保留 v866 的旧底盘、world all-gather、动态布尔过滤等开销，只供隔离验证错误。它不是基于 v890 的最终优化候选，不能直接覆盖生产。** 首个验证对象最好抽取成单卡 MD→DN 最小代理；若只能用正式 P1，则先移植到 v890 的仅 c9 分支，并保留完整正确输出。

改变量名建议：`expert_row_start`、`tile_expert`、`tile_expert_row_start`、`tile_count_in_expert`、`tile_end_in_expert`。六元 tuple 的每个字段写明维度。数学验证与性能改造分两步，避免一次同时改通信、tile 参数和数值格式。

## 7. A2：把修好的 EP2 做成真正有机会提分的实现

### 7.1 放置与上限

固定逻辑 pair `{0,1}`、`{2,3}`。每 pair 服务自己的两个源 rank；偶数 rank 计算 expert 0…127，奇数 rank 计算 128…255。

```text
owner = 2*(source_rank//2) + expert//128
local_expert = expert % 128
branch_code = ((source_rank%2)*T + token)*k + slot
```

每 owner 最多 `2*T*k=65536` branches，这是本两源模型的严格上限；平均 32768 不能拿来分配。tile 数满足 `Σ ceil(n_e/128) <= ceil(Mcap/128)+128`。无效 capacity 尾部不能作为有效 branch 读写。四源旧实现的相同容量不是四源最坏上界，不能混用。

第一版可保留全量静态权重缓存，只让计算读取正确的半份 view。性能成立后再做半份持久权重，避免同时改变所有静态预处理。tiled GU/D 的 expert slice 必须乘每 expert 的物理 tile 行数。

### 7.2 必须移除 v866 的两类动态开销

**通信：** v866 对 token、scale、ids、route weights 做 world all-gather，再只取 pair 对应两份；返回又 world all-gather。它不是预算中的单 peer 协议。应先在源 rank 完成本地 router、FP8 token、排序与 counts；仅把必要数据发给 partner。

**过滤：** `flat_ids[owned_mask]` 会产生动态尺寸，再以 `n_f` 建立缓冲/特化。正式快路径改为固定容量、GPU counts 驱动。现有 expert 排序已把同专家变为连续段，收端直接合并两源该 expert 的 ORDER，无需布尔筛选后再完整排序。

建议复用 [旧四卡 EP 的索引合并机制](experiments/2026-09-19/results/ep_build.md)，把源数从 4 改为 2，但重新证明正确性和容量：

```text
src 0 expert e 的段 → src 1 expert e 的段
ORDER 存 branch_code；MD 的 ORDER//k 寻址 [2T,H] token-only A
route weight 和 A_scale 按同一 source/token/slot 编码读取
```

保持所有 branch 恰好归属一个 owner，owner 内每 token 的缺失分支为零。partial 首次和每次复用均要完整覆盖；不能把上次输出当零值。

### 7.3 成本预算

| 每 rank 指标 | v890 replicated | 真 EP2 |
|---|---:|---:|
| 计算专家数 | 256 | 128 |
| 平均 rows/expert | 128 | 256 |
| 平均有用 branches | 32768 | 32768 |
| c9/c10 权重工作集 | 6 / 4.5 GiB | 3 / 2.25 GiB |
| token FP8 工作集 | 16 MiB | 32 MiB |
| 外发 token FP8 | 0 | 16 MiB |
| 外发 BF16 partial | 0 | 32 MiB |
| 外发 INT8 partial（可选后续） | 0 | 16 MiB + scale |

以 int64 ORDER、FP32 route weight、FP32 token scale、int32 counts 计，dispatch 辅助数据约 401 KiB/rank；INT8 return 每 token/256列一个 FP32 scale，约 256 KiB/rank。以上是 outbound payload，不再人为翻倍成“双向”；协议与额外 HBM copy 另计。

旧四卡 probe 曾得到约 126–154 GB/s 的大块发送量级，仅作敏感性参数。按这个范围估算，EP2 FP8 dispatch + BF16 return 的纯 payload 已要约 **0.33–0.40 ms**；加 INT8 return 可降到约 0.22–0.27 ms。未计 launch、索引合并、partial 归约和 rank 等待，也没有证明单 peer 能达到该范围。

两源并行还可能降低 padding。CPU 合法随机 top-8 模型中，replicated 平均 padded rows=48,640；EP2 两 owner 平均 41,152，约为原来的 84.61%。这解释了“有用 FLOPs 不变”仍可能减少实际执行量；**它是合成路由模型，不是对 OJ 路由的实测**。[验证](reports/2026-09-29-tp2-layout-proof.json)

判定公式：

```text
净节省 = replicated 计算/准备时间 - EP2 计算/准备时间
       - peer dispatch/return - partial 归并 - 新增最终合并 - rank 等待
```

如果没有合法阶段计时，就用相同 generated workload 分别测两个完整子链，且将共享准备成本留在两边；或者只用正式整案 A/B。不能从两个噪声很大的独立测量相减得出“精确通信时间”。

### 7.4 通信正确性

优先使用比赛明确允许的 utils 对称 tensor 管理和已有 `triton_dist.jit` 设备通信模式。第一版在当前 stream 按阶段执行，避免同时引入持久等待、跨 stream overlap、动态 subgroup。

协议必须明确四件事：发送方何时可以改写源 buffer；接收方何时能读到本轮数据；接收完成后何时能覆盖目标 buffer；下一轮如何区分 epoch。所有 rank 的对称分配仍需同序。`fence` 只保证相应顺序，不等于发送已完成；多 CTA producer 不能只靠某一个线程的 fence 为所有写入背书。[NVSHMEM ordering 说明](https://docs.nvidia.com/nvshmem/archives/nvshmem-221/api/docs/gen/api/ordering.html)

不要让满占 SM 的 wait 阻塞尚未运行的 producer，也不要以 rank0 最快时间代表四卡结束时间。NVSHMEM 的非本地依赖不会自动全部呈现给 CUDA 调度器，必须建立可见的阶段依赖。[CUDA/NVSHMEM 交互](https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html)

### 7.5 精度与验证门

EP2 owner 将自己拥有的各 slot 按固定顺序累加，返回 BF16 partial，再由源 rank 固定顺序合并。这会改变舍入和加法结合方式。首先用独立题意参考检查；与 v890 的差仅作诊断。INT8 return 等 BF16 版本成立后才研究，不把“回传格式变小”和“索引修好”混成一个实验。

建议门槛（工程预算，不是物理定理）：

1. **数学门**：修复后的单卡 MD、DN、final oracle 通过；同输入 byte exact；任意偏斜不越界。正式阈值 22 dB，研发优先保留至少 22.5 dB，且说明相对现役变化。
2. **计算门**：c9 计算代理若节省不到约 0.45 ms，BF16 return 版很难支付额外成本，暂停完整集成；c10 初筛约 0.35 ms。根据实测 peer 成本可更新此门，不能无依据降低。
3. **通信门**：用一次真实 payload 往返验证协议与预算，含完整完成/等待，不只测 launch。一个保守版本和最多一次机制修复即可。
4. **整案门**：c9 初筛至少两对 AB/BA，全案≥5% 信号才投入独立确认，目标≥8%；确认后才扩 c10。小于 3% 且误差/复杂度增加明显则止步。
5. **收口门**：合入 v890 后其他案无回退，任意调用路径合法，500 秒总预算通过；然后才晋升。

没有通过计算门时，不做半精度 return、overlap、更多 CTA 扫描。修复后的 world-all-gather 版本若很慢，只证明该诊断包装慢；要据上述成本表判断是否值得换成单 peer，不能直接否定也不能无限续命。

## 8. B：I 维 TP2，只在改变旧失败前提后复查

### 8.1 它已经有人做过

本次进一步检索找到了被较新交接遗漏的旧实验：

| 方案 | SID | 记录结果 |
|---|---|---|
| 四卡 I 维 TP | 145072 | Accepted；c9/c10 3.894/5.208 ms |
| 两卡 pair TP + world reduce-scatter | 145084 | Accepted；c9/c10 5.054/3.329 ms |
| pair-local P2P / subgroup | 旧 pair/pair2 | P2P 异常或取消，没有有效收益 |

来源：[09-16 SUMMARY](experiments/2026-09-16/SUMMARY.md)、[旧 TP2 源码](experiments/2026-09-16/candidates/tp256_pair_rs.py#L5752)。这些结果不支持原样重开。TP 本身是成熟的矩阵切分方式，不是本项目的新发明；[NVIDIA Megatron 的 column/row parallel 定义](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)仅作为数学背景，不引入其高层 GEMM 实现。

### 8.2 唯一值得比较的新前提

旧 `_run_tp256`：发送 BF16 X 给全部四卡、只取一对；两成员重复处理两源 router；使用旧非 tiled 权重；建立并清零 `[world*T,H]` partial，最后 world reduce-scatter。

本次允许复查的版本必须全部改为：

1. 本地 router 与 token FP8 只做一次，复用 A2 的单 peer 协议，发送 FP8 + route/scale/ORDER。
2. 每 rank 持全部 E 个专家的半个中间维度，使用当前 tile 连续权重。
3. 只处理本 pair 的 `[2T,H]`；partial 返回 partner 的 `[T,H]`，删除世界维度清零和 world reduce-scatter。
4. 两 rank 在静态 GU norm、权重 scale 上使用一致合同，避免切半后无意改变 ACT 量化尺度。

旧 BF16 world broadcast 每 rank 外发约 96 MiB，本设计的 FP8 peer 仅 16 MiB；这是实际改变旧成本结构的部分。仍需测得收益，不能仅靠字节比宣布会赢。

### 8.3 数学与布局

对 `I_s=I/2`，rank 的 `shard=rank%2`：

```text
G_s = G[:, shard*I_s:(shard+1)*I_s, :]
U_s = U[:, shard*I_s:(shard+1)*I_s, :]
D_s = D[:, :, shard*I_s:(shard+1)*I_s]
a_s = w * SiLU(x G_s^T) * (x U_s^T)
c = a_0 D_0^T + a_1 D_1^T
```

实数恒等式成立；分段 FP32 累加、分段 FP8/BF16 舍入不是 bitwise 等价，必须测最终 SQNR。首版保留原完整 D 的逐行 scale，只切 q 的 K 维；GU 按输出通道切 q 和 scale。BNORM 建议先保持完整专家的原值，使两半 ACT 的行尺度一致，少引入一个变量。

物理 tiled 切片不能直接 `dn_flat[:, :I_s]`。GU 为 `[E,2,R,KT,128,128]` 的 tile 序；DN 为 `[E,H/256,I/128,256,128]` 的 tile 序。GU 切 R，DN 切 K-tile，再重新连续化静态 shard。c9/c10 的 `I_s=1024/768` 均能被 128 整除。

本轮已做 **144,000 个 GU/D tiled 坐标检查**，并用 50 个小例检查 float64 分片恒等式，最大差 `2.78e-17`。这不覆盖 FP8、BF16、TMA 或通信。[脚本](reports/prove_tp2_layout_20260929.py)、[结果](reports/2026-09-29-tp2-layout-proof.json)

### 8.4 为什么只做备用

相对 EP2：TP2 每 rank 的分支数固定为 `2T*k=65536`，无需 owner 过滤，也没有 owner 间 branch 数偏斜；但它的 Down scratch 有 **256 MiB**，是 EP2 平均 128 MiB 的两倍，final 归并工作也更多。DN 的 K 减半，短 K 流水线可能更低效。激活总字节约相同，因为 M 加倍、I 减半。

| 指标 | EP2 | TP2 |
|---|---:|---:|
| 专家数 | 128 | 256 |
| 平均分支数 | 32768，最坏 65536 | 固定 65536 |
| 中间维 | I | I/2 |
| 有用 GEMM 工作 | 约与原每卡相同 | 约与原每卡相同 |
| 权重工作集 | 原来一半 | 原来一半 |
| ACT 容量（c9 平均） | 64 MiB | 64 MiB |
| Down FP8 容量（c9 平均） | 128 MiB | 256 MiB |
| 通信预算 | 单 peer 两腿 | 同样两腿 |

先用同一个计算代理比较 prepare+MD+DN+partial，**最多 2 次 custom**。TP2 若没有足以支付通信的节省，或比已修 EP2 还慢，不构建第二套正式通信。若 custom 镜像不同，结果只作筛选；临界优势不能据此晋升。

## 9. C：c11 专家对 DN 与确定归并

这条路线的作用是为 c9/c10 之外寻找结构机会，旧 v844 指引提出过，但当前目录未找到其 GPU 收益记录。不能称已证实的新优化。

c11：T=65536、E=32、k=2、H=I=1024。将 token 按无序专家对 `(lo,hi)` 分组，同一组的 token 共享两个 D。一个 CTA 为若干 token 计算两条 DN，然后在 CTA 内按确定顺序相加，唯一写入对应 output，不产生跨 CTA 浮点 atomic。

```text
pair_id = lo*(2*E-lo-1)//2 + hi-lo-1
row_lo[t] = compact_INV[2*t + slot_lo]
row_hi[t] = compact_INV[2*t + slot_hi]
```

**ACT 仍是 compact 布局，必须用 compact INV；INV_PAD 属于 Down，不能用于 ACT。** 固定恢复原 slot 顺序；若删除原 DN FP8 量化，则作为新数学链对独立参考验收，不能声称与 v890 逐字节相等。

可删除 c11 约 128 MiB Down FP8 写和 128 MiB读，以及 CSCL/fin 相应工作；新增 pair metadata、两路 ACT gather、第二套 accumulator 和 D 重读。D 总容量约 32 MiB；“小于 Hopper 家族 L2 容量”不意味着常驻，因为 A 与其他数据也竞争缓存。

已有 CPU 随机模型：496 对、平均约 132 tokens/pair；BM64 pair 需要 166,400 branch-row 等价计算，原 expert BM128 为 133,120，约多 **25%**。BM128 pair 约多 55%。[旧索引验证](reports/2026-09-27-pair-directions-proof.json)

因此第一版只做 c11、BM64、保守流水，不能一开始推广 c1/c2/c12。比较两条完整子链：

```text
基线：同一 ACT → 当前 v890 static padded DN → 当前 final
候选：同一 ACT → pair metadata → pair DN + 唯一输出
```

测试包含真实分布规模、全部 token/slot、空 pair、极偏 pair、127/128/129 边界。若 pair metadata 由路由侧生产，其每次动态成本也必须计入，不能缓存 warmup 路由。

预算最多 2 个 custom。没有≥10% 子链改善且有望转成≥3% c11 全案，就停止，不发 BM32/64/128 参数表。即使 c11 单案赢 10%，整题收益仍有限；这是一条互补实验，不是 raw 90 的充分方案。

## 10. 当前应该关闭或冻结的方向

| 方向 | 有效证据 | 本轮处理 |
|---|---|---|
| static DN + PAD_DELTA | v882/v890 已生产 | 保留，不重复派发 |
| 全 HIST/scan/metadata 融合 | v901 慢约 3%–9%；v902/v904 正常配对在噪声内 | 停止独立迭代 |
| v903 | 150928 TLE | 不能断言一定是编译；无新诊断不原样重发 |
| c6 static DN s2 | 149911，1.425 ms，明显慢 | 保持现役 |
| c3/c4 final BT64 | 149943 无收益 | 保持 BT32 |
| c1/c2 ACT TMA / 宽 B / gran1 GU | 149609/149612/149630/149636，整体负 | 不扫变体 |
| packed FP6/INT6 | 旧记录多倍慢、资源溢出 | 除非改变原生计算/解码机制，不重开 |
| 原生 INT4 探针 | 旧固定值有效吞吐不及 FP8 | 不凭位数更少推断速度 |
| FP16 累加大 tile | 精度可行但 lowering/寄存器代价负 | 没有同栈完整子链正证据不再扫 |
| E256 BM64 / sorted-A | 旧子链负 | 先改变 B 流量或算法，不重试同机制 |
| num_ctas=2 / 手写新版 Gluon WGMMA | 当前栈有编译断言或 API 不匹配 | 截止前不押编译器能力升级 |
| 旧四卡 EP | 计算节省不足付通信 | 只复用索引/通信知识，不重投原方案 |
| 原样 TP2 | 145084 已负 | 只考虑第 8 节改变通信/布局后的版本 |
| 路由剪枝、2:4 直接删权重 | 路由近均匀、精度余量有限 | 无误差与真实稀疏执行证据不投入 |
| 静态低秩、跨 expert 近似共享 | 未知权重是否有足够结构 | 无谱/残差与端到端证据，不在截止前展开大工程 |
| 参数遍历 / 重复同码碰窗口 | 不能解释结构收益 | 停止 |

来源：[最新 R2 记录](experiments/2026-09-27/notes/r2_static_dn_c11.md)、[最新排序记录](experiments/2026-09-27/notes/r5_sort_meta_probe.md)、[历史负结果](docs/HISTORICAL_NO_REPEAT.md)、[v844 指引](OPTIMIZATION_GUIDE_V844_2026-09-27.md)。旧文件中的推断语气不自动升级成事实；例如“无 SQNR=TLE 一定发生于编译”不成立。

## 11. P0：会影响赛后复测的实际代码问题

### 11.1 调用序号不应决定不同数学结果

v890 的 `_CALLN`、`_GA` 仍使 call1、call2、call3–5、call≥6 使用不同 router/量化/MD 路径。源码中有“为首次 JIT 预算延迟启用”的解释，但这不免除任意调用顺序的正确性和同输入确定性。

先按目标家族列出实际路径，再做同输入连续第 1/2/3/5/6/8 次和 A→B→A 的验证。不要把“每条路径分别 determinism OK”写成“路径之间相同输入必然相同”。也不要简单全开 call3 分支；历史全开有总预算 TLE。

新 EP/TP 只能依据 shape、权重生命周期和明确协议状态调度；不能探测评测阶段或调用次数后少算。epoch 用于同步，不用于区分检测/计时数学。若统一某家族路径，应作为单独正确性修复，有独立 SHA 和正式验证，不夹在速度候选里。

### 11.2 静态缓存合同

老缓存多按 shape，新 static-DN cache 按 `id(dn_q), id(dn_s)` 并持有引用。后者能避免自身对象 id 被回收复用，但不能修复其上游 shape-only 缓存。连续同 shape 不同权重、同对象原地更新、A→B→A 都需要被考虑。

题面仅保证同一测试点内权重固定，没有明确公开跨测试点的 tensor 身份/原地更新合同。旧文档“必须只按 shape”是历史经验，不能作为一般正确性依据；另一方面只按 id 也可能因包装对象变化反复预处理，超出 500 秒。先记录公开允许的身份/边界信号，再决定键和引用生命周期。若没有可靠失效信号，需要内容一致性验证或重新预处理，并计量成本。抽样 fingerprint 不是完全一致的证明，也不绕过被禁止的 `_version`。

静态缓存可以保存派生权重、布局、scale、通信容量；动态缓冲可复用容量，但每次必须重新生成本次 X 的 route、activation 和 output。

### 11.3 数值余量

c3 现役最低已记 SQNR 为 22.70 dB，只有 0.70 dB 余量。误差不能无条件按功率相加：若当前误差范数为 `e0`，新误差为 `e1`，最保守先用 `||e0+e1|| <= ||e0||+||e1||`，最终仍直接测独立参考。

新 partial return、分片 DN、静态尺度改动均需记录 finite、subnormal/饱和、所有 rank 的结果。测试量化时包含零值和极小量，尤其检查 static DN 中 `max(As*C,1e-12)` 与 q 的缩放是否仍匹配。仅日志同为 23.13 dB 不能证明输出 exact。

## 12. 只用 OJ 的实验方式

### 12.1 不额外消耗提交的准备

每候选至少附：父基线 SHA、候选 SHA、差异、触达 case、机制、预期节省及新增字节/launch、最坏容量、数值变化、继续/停止门。先 AST/语法检查，再读最后生效的调用链，确认修改真的命中目标；旧 v893 曾修改未命中的 MD host。

本地可做索引/代数/源码检查；本地没有 H800。它们不替代硬件测试。对同一问题不反复跑已经通过的静态检查，除非代码又变了。

### 12.2 custom 的证据范围

- 先查可用语言/模式，不假设 P1 distributed custom 可用。
- 单卡 custom 用于合法 generated workload 的算子 oracle 和完整被改子链；输入包括实际规模的权重/activation 工作集，避免小矩阵 L2 热循环冒充 6 GiB 工作集。
- baseline/candidate 必须完成相同任务、写完整输出、读本次输入；构造和计时边界一致。
- 若 Event 或内部 benchmark 被拒，使用平台报告的完整 workload tk 分开做 A/B，不绕开白名单。
- 记录镜像/版本可见信息；P2/P3 的 3.6 成功不能证明 P1 的 3.4 可用，资源/时间也不能直接移植。
- 如果没有兼容的 custom，改为一个家族、一个变量的正式 P1 初筛；不要为诊断而堆入所有新特化。

### 12.3 正式 P1

唯一执行者先查询自己的队列、令牌池和额度；冻结源文件，再使用 [SUBMISSION.md](docs/SUBMISSION.md) 的当前网页/PoW 通道。不要输出凭据，不取消未知任务。拿到 SID 后只跟踪该 SID；Pending、本地请求 timeout 不代表提交未发生，不据此重复发。

正式记录必须包含 sample、12 案、全部 rank 的平台正确性结论、determinism、SQNR、tk、tb、th、单案 displayScore 和总 displayScore。部分日志被截断时写“缺失”，不能把 rank0 或首段输出推广到全部。

### 12.4 固定的性能判定

1. 初筛 2 对 AB/BA，事先冻结目标 case、方向和停止阈值。
2. 有至少 5% 结构信号的候选再做新一批 3 对确认；不是持续追加直到某发赢。
3. 目标案看原始 tk、配对比和重复分布；未触碰案只作环境诊断，不用旧机器签名硬校正出收益。
4. 保留 `tk=0`、异常 tb 或未改案同步断崖的记录，标注不宜作性能证据；不能只删候选慢值。
5. 分别报告官方 raw/net 与固定 tb 模型收益。候选全案修改时没有 untouched control，不虚构控制组。
6. 合并两个胜者后重新对同一生产测量，不把各自百分比直接相加。

当前 `scripts/sqnr.py` 只读 dict 的 `data` 且按枚举标案，会漏读 `content` 或串案。`scripts/cases.py` 的 `score` 也可能是正确性分。新审计工具按 `tc=` 归案，兼容 `content/data`，保留性能 `displayScore` 和 schema 指标；后续 executor 应复用这个解析逻辑，而非照抄旧终端标签。

### 12.5 每次反馈模板

```text
机制 / 目标 case / 改变的历史失败前提：
baseline path + SHA / candidate path + SHA / diff：
数值、缓存、同步、最坏容量合同：
新增/删除 bytes、launch、JIT 特化与峰值显存估计：
CPU/custom/P1 各层的结果与限制：
CID/SID、anchor SID、sample/全部案/全部 rank 结果：
逐案 tk/tb/th/displayScore/SQNR/determinism：
异常标记、固定 tb 收益、正式 raw/net：
继续或停止的理由、下一个最便宜的消歧实验：
```

## 13. 可直接复制给其他 coding agent 的任务卡

### Agent A：EP2 修复与计算代理

> 先读本指南第 6–7 节和 AGENTS.md。生产基线是 v890、SHA 436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc。已定位旧 EP2 把 per-expert `meta_split_cum` 传给按 pid_m 索引的 MD/DN；修复候选和 patch 在 experiments/2026-09-29/candidates/。先复核两个参数及独立 metadata oracle，再抽取 c9 两源/128专家 MD→DN→partial 的单卡代理，比较等有用工作量的 v890 子链。覆盖 0/127/128/129 counts、最坏 65536 branches、正确 source/token/slot 与所有有效 row。不要同时改 tile 参数或回传格式。计算门通过后提出仅 c9 的完整 v890 候选，不覆盖生产。交 SHA/diff、完整子链成本、正确性及是否值得推进 peer 通信的判断；正式提交交唯一平台执行者。

### Agent B：真正的单 peer 数据路径；TP2 仅作成本对照

> 先审计旧 v866 与 145084 的 world 通信开销。实现本指南第 7 节的固定 pair FP8 token/scale/ORDER/weights 交换、两源 expert 段合并和 BF16 partial 返回。复用已验证的 utils/设备通信接口，所有 rank 对称分配同序，明确 epoch、可见性和 buffer 复用。不要使用已失败的 unbatched P2P/subgroup 路线，也不要让 world all-gather 冒充单 peer。首版无 overlap。给出实测完整往返成本及与 Agent A 计算节省的总账。若成本足够，可按第 8 节用同协议做 TP2 计算对照，但须保留旧负结果并解释 changed premise；不得直接重发 tp256_pair_rs.py。最多两个 TP2 custom 后决策。

### Agent C：c11 专家对 DN

> 从 v890 的 c11 static padded DN+fin 出发，只实现第 9 节的 pair-grouped DN。保留原 MD，ACT 用 compact INV，先 BM64；同一 CTA 计算两分支并唯一写 output，无浮点 atomic。比较的候选必须包含 pair metadata 和 ACT gather，记录 padding 与 B 重读，不只测 dot。对独立题意参考验 SQNR，对重复输入验 byte exact。预算最多两个 custom；没有≥10%完整子链信号就停止，不扫参数，不扩到 c1/c2/c12。产出候选、索引证明、成本表、结果和停止判断。

### 唯一平台执行者 / 集成人

> 冻结 v890，不使用禁止的脑暴技能。先修结果解析和按 tc 归案，记录当前榜分75.17、目标80与每次独立raw。串行跑已冻结且通过静态门的候选，不取消别人的队列、不重复提交未知SID。按2对初筛、3对独立确认执行。通信方案必须经过真实P1四卡，单卡结果不能晋升。明确反映失败发生在参数合同、oracle、编译、同步、数值还是性能，证据不足时写未知。最终同时验任意调用/缓存变化/全部case，输出最终文件、SHA、SID和回退文件。

## 14. 截止前的日程与实验预算

这是建议上限，不要求花完；可以提前因停止门收口，不为常规可逆实验增加审批环节。

| 时间（UTC+8） | 工作和交付 |
|---|---|
| 9/29 首个工作时段 | 完成 EP2 参数修复的 GPU oracle、计算代理；平台解析/缓存路径表同时完成 |
| 9/29 后续 | 只有计算门成立才测真正 peer；c11 pair DN 做最多两个 custom；TP2 先做成本表 |
| 9/30 白天 | c9 完整 A/B；通过后扩 c10；独立确认幸存方案，测组合而非加总百分比 |
| **9/30 18:00** | 若仍没有≥5%可复现整案信号，停止扩大新架构，转向保存已证实收益与正确性收口 |
| **10/1 12:00** | 停止引入新通信协议、大范围量化/缓存重写；选最终候选和回退 |
| 10/1 下午至晚间 | 完整P1复测、确定性/缓存核对、保存SID/SHA；给队列与500秒总预算留余量 |
| **10/1 23:59 前** | 完成最终正式提交；不把最后一分钟当首次测试机会 |

建议首轮总量：**6–10 个 custom、18–26 次正式 P1（包含控制）**；只有明确胜者才追加约 6 次组合/收口测试。metadata 接线错误不值得再消耗十几发盲诊断；TP2 若没有独立计算正信号，不占正式队列。

raw 90 需要广泛且显著的收益。若 9/30 时只有 c9/c10 改善，执行者应如实报告它对整题的最大贡献，继续争取成绩但不编造“已经有到 80 的确定路径”。比赛规则要求赛后复测，异常计时窗口不能作为工程收口证据。

## 15. 仍值得向用户/主办方索取的信息

截止和目标已经由用户给出；GPU、拓扑、3.4 版本和初始化接口也已有答复，不再重复询问。未决且真正影响执行的只有：

1. P1 跨测试点静态权重的失效合同：是否可能 same object/storage 原地换值，是否提供合法边界/版本标识。它影响最终缓存正确性与预处理成本。
2. 合法的逐阶段计时、PTXAS stderr、rank 失败日志；若没有，就按完整 workload/P1 测量，不能等待不存在的 profiler。

本次已发现具体 metadata 错误，所以“必须先拿到 stderr 才能动 EP2”不再成立。修复与最小 oracle 可以立即进行。本文没有对外发论坛帖、邮件或联系其他人。

## 16. 附件与复核命令

| 附件 | 用途 |
|---|---|
| [EP2 两处修复](experiments/2026-09-29/candidates/ep2_v866_metadata_fix.patch) | 最重要的可直接审阅改动 |
| [候选源码](experiments/2026-09-29/candidates/ep2_v866_metadata_fix.py) | 旧 v866 诊断底盘；未经 GPU 验证，不是生产 |
| [metadata 验证脚本](reports/prove_ep2_metadata_20260929.py) / [JSON](reports/2026-09-29-ep2-metadata-proof.json) | 反例、30组映射、受影响位置、SHA |
| [TP2 布局验证](reports/prove_tp2_layout_20260929.py) / [JSON](reports/2026-09-29-tp2-layout-proof.json) | 静态切片坐标和实数恒等式；不证明速度 |
| [只读刷新脚本](reports/refresh_20260929_readonly.py) | 自己的榜面、已有SID和custom可用性；不提交、不取令牌 |
| [平台状态](reports/2026-09-29-platform-readonly.json) / [逐案记录](reports/2026-09-29-submission-audit.json) | 当前事实和失败原始日志 |
| [评分重算](reports/audit_v890.py) / [审计结果](reports/2026-09-29-v890-audit.json) | 绑定 v890 SHA 的离线模型 |
| [阈值](reports/2026-09-29-v890-thresholds.csv) / [情景](reports/2026-09-29-v890-scenarios.csv) | 目标幅度与固定 tb 总账 |

```bash
cd /Users/sakimi/Desktop/xpuoj-p1
shasum -a 256 p1/kernel.py
python3 reports/audit_v890.py
python3 reports/prove_ep2_metadata_20260929.py
python3 reports/prove_tp2_layout_20260929.py
```

后两项只运行 CPU。旧 `reports/audit_v844.py` 绑定旧 SHA，面对当前生产报错是预期行为，不要修改其断言伪造旧基线。

交付时生产 SHA 仍为 `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc`。下一次晋升必须同时更新生产、README、manifest 与对应正式 SID，避免后续 agent 再从过期入口出发。
