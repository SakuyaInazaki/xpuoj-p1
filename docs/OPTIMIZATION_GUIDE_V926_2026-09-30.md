# P1 MegaMoE：v926 接手结论与下一轮优化任务书

> **历史快照，已被 [最后冲刺指引](OPTIMIZATION_GUIDE_FINAL_SPRINT_2026-09-30.md) 取代。** 截至 9/30 22:34，生产已是 v12，榜分 79.08；本文 A 已晋升、B 已试无收益，清理版 SHA 与待办不再适用。下文保留原貌用于追溯。

本轮榜面只读核验：**2026-09-30 05:00，Asia/Shanghai**；连续异常队列补全时间 **05:36**。目标仍是 **2026-10-01 23:59 前争取 raw 90 / net 80**。本文接替原 9 月 30 日指引；原指引中的 slot DN 主线已经实测关闭。

**用户补充明确：要追的是平台 tk 极低的得分异常。** 本轮为此恢复连续 **187 次提交**：125 次完整 AC 中 7 次极低 tk，全部属于本段首次 SHA（7/58）；重复 SHA 的 67 次 AC 中为 0。新增优先任务 **P：区分源码身份、JIT 冷暖和计时采集异常**，具体证据及最多 6 发的对照设计见 [平台极低 tk 专项指引](PLATFORM_TIMING_ANOMALIES_2026-09-30.md)。这是可检验的关联，尚无可稳定触发异常的办法。

结构优化保留 **A：把 MD 行参数计算移到现有 GQ，先 c11；B：c9 保留 tile 连续布局的交错 GU 单累加器**。两条都尚未取得 GPU 正收益，不能承诺 raw 90。稳定调用与缓存合同列为独立正确性工作；主办方没有新回复，不因等待答复阻塞其余分析。已失败的 EP、slot DN、metadata 合并和配置扫描不继续占主队列。

本轮完成了代码审计、最新结果恢复、外部资料核查、CPU 索引/标量证明和工作区整理。**新正式提交 0 次、custom 0 次；没有新增 GPU 提分结论。** 用户自行安排其他 coding agent；本轮没有代为派发，也没有使用被禁用的技能。

## 1. 接手必须先核对的三个版本

| 对象 | 路径 | SHA-256 / 状态 |
|---|---|---|
| 已测 v926，性能与数值对照锚 | [`p1/references/kernel_v926_measured.py`](../p1/references/kernel_v926_measured.py) | `2633cc995eb2563e22c1a212df1256141bdc1b520fe841d4dd4c15c0d382e7bf`；SID 151806 / 151807 全部 AC |
| 当前可读源码 | [`p1/kernel.py`](../p1/kernel.py) | `eb5ffee8db4012570f083f6cf85acaf36bf1297b880fd55942f02061ead1fc26`；v926 的保守清理版，未重新上 OJ |
| v890 回退锚 | [`p1/references/kernel_v890_measured.py`](../p1/references/kernel_v890_measured.py) | `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc` |

清理版从 **7,363 行 / 188 个函数定义**变为 **5,331 行 / 140 个函数定义**，删除 10 个被后续定义覆盖的旧定义、38 个保守调用图不可达定义，以及过时/重复注释。所有保留函数的 AST、所有非函数顶层语句及执行顺序均与原版相同；没有顺手修改计算、启动参数、调用次数分支或缓存策略。证明见 [cleanup.json](../reports/2026-09-30-v926-cleanup.json)，生成工具见 [clean_kernel.py](../scripts/clean_kernel.py)。

这证明的是**保留代码的静态结构一致**，不是对装饰器、JIT 缓存键或冷编译耗时的运行时证明。清理会改变源码哈希；首次完整确认仍须核对 500 秒总预算。做历史性能比较使用已测快照，不要把清理后的 SHA 写成已有 AC。

开工命令：

```bash
cd /Users/sakimi/Desktop/xpuoj-p1
shasum -a 256 p1/kernel.py p1/references/kernel_v926_measured.py
python3 -m py_compile p1/kernel.py
python3 scripts/check_jit_globals.py p1/kernel.py
python3 reports/prove_v926_directions.py
```

本目录不是 Git 仓库。每个候选使用独立文件、SHA 和 diff，不能依赖不存在的 Git 回退。最终 [交付验证记录](../reports/2026-09-30-handoff-verification.json)包含清理复现、语法/JIT 全局检查、1556 项迁移 SHA 和 125 次完整 AC 的评分公式核对。

## 2. 最新实测事实：先关闭过期任务

当前 P1 榜分 **75.17**，最佳 SID **149493 / raw 85.17**；P1 累计提交 **3426**，罚分按当前 10 分上限估算。账号综合排名为 4，这不是 P1 单题名次。最近十条列表均为终态，但下一次执行者仍须重新检查并发。

| 实验 | 最新证据 | 决策 |
|---|---|---|
| c10 static DN，v926 | A1 151805：1.881 ms；B1 151806：1.850 ms；B2 151807：1.851 ms；A2 151808：1.890 ms。两对分别 −1.65% / −2.06%；SQNR 23.11 → 22.69 dB | 已晋升；不再扫 1.02 系数，不重复寻找同一收益 |
| c11 slot 分组、第二路 DN 融合 fin，v932 | 151793：0.891 ms；151804：1.014 ms；同窗锚 0.876 / 0.871 ms | 当前实现实测负，关闭 |
| slot tile-offset 改进，v933/v934 | 151796 / 151802 总 TLE，无目标性能数据 | 未验证，不等于速度失败；本冲刺不盲目重发 |
| EP2 metadata 修复 / single peer / BM256 | 150964：c9/c10 2.843/2.441；150965：2.705/2.310；151332：2.950/2.737 ms | 已修对但整案仍慢，不再派“修 EP2 就能提分” |
| offsets + metadata 融合，v904 | 第二对 150945 对 150943：c11 −0.34%、c12 +0.27% | 收益不可分辨，不再作为新主线 |
| c1/c2 单宽 accumulator / gran=1 | 149630 / 149636 已 AC，c1 明显慢；gran=1 两案均慢 | 不把 B 的 c9 试验自动扩到 c1/c2 |

来源：[最新平台快照](../reports/2026-09-30-v926-platform-readonly.json)、[最近 9 次提交详情与源码 SHA](../reports/2026-09-30-v926-submission-audit.json)、[上一轮 A/B 结论](../experiments/2026-09-30/notes/slot_dn_A_negative_b_promoted_20260930.md)、[metadata 关闭记录](../experiments/2026-09-27/notes/r5_sort_meta_probe.md)、[历史结构候选](../experiments/2026-09-27/SUMMARY.md)。

151793 的 c12 `tk=0`，149493 的 c11/c12 `0.325/0.000 ms` 不代表可复现的执行速度，但它们对平台分数确有影响，已作为单独的提分研究对象。149493 还叠加 c5 的 `tb=48.349 ms` 异常抬高。新增连续样本、同 SHA 对照和受控复现方案见 [极低 tk 专项](PLATFORM_TIMING_ANOMALIES_2026-09-30.md)。正常性能估计剔除这些异常；平台成绩与异常统计完整保留，不混为同一结论。

## 3. 题目、环境与执行边界

完整数学约定在 [1-full.md](../1-full.md)。每 rank 输入 X[T,H]，本地只持 E/4 个专家；router 在四卡相同。BF16 logits → FP32 softmax/top-k/归一化；专家执行 gate/up、SwiGLU、路由加权、down，最后按 token 在 FP32 合并并写 BF16。官方参考在中间指定了 BF16 舍入，生产的 FP8 路线属于误差预算内的近似。

必须满足：最终 SQNR ≥22 dB、输入只读、输出有限且完整、同输入独立调用输出逐字节一致。至少一次预热后会换 X；只能缓存静态权重派生数据与可复用容量，不能缓存旧 X 的路由、activation 或输出。

用户给出的最新讨论已冻结为 [addinfo 原文副本](../reports/2026-09-30-addinfo-source.md)。其中官方回复支持以下事实：

- P1 使用 Triton / triton-dist **3.4 系**；精确 fork/commit 未知。P2/P3 的 3.6、CUDA stream 回答不套用到 P1。
- 硬件为 **4×H800 80G SXM，默认 700 W，任意两卡 NV8**。NV8 不是本应用实际通信带宽。
- torch.distributed 与 NVSHMEM 已由评测器初始化。通信使用允许的 `triton_dist.utils`，不再次 init/finalize；设备端 dist 调用放在 `triton_dist.jit` 中。
- **500 秒覆盖一次 torchrun 整批评测**，包含编译、预热、检查等。不是每个 kernel 各 500 秒。官方帖子写十个点，而现存详情有 12 案，使用详情的真实案号。
- 本次 `triton-dist` custom availability 仍为 **false**。单卡 custom 可用于其允许的完整 generated workload，但需先确认题目模式、接口和版本；它不能验证 P1 四卡行为。
- 没有拿到合法逐 kernel profile 新接口；材料明确提到 Event 被拒，不能绕过。后续验证只用 OJ 正式提交/允许的 custom，不要求用户准备本地 H800。

当前 12 案全部进入 replicated 路线：静态预处理收集全专家权重，稳态每卡只算自己的 token。因此稳态已经没有 token dispatch/return 通信，不能重复把“去通信”列成待赚收益。

## 4. 当前分数账与资源账

下表使用 v926 的 151806/151807 逐案 tk 均值，以及四发 ABBA 的逐案 tb 中位数。这是 **n=2 的描述性基线**，不是未来 A/B 的替代品。冻结 tb 后，当前观察区间 `th=0` 使用 `q=floor(100·tb/(tb+tk))`；整题 raw 为十二案 q 的平均。官方说明其他区间可转对数计分，不把 100 当绝对上限。

| 案 | T / H / E / I / k | tk ms | 固定 tb ms | q | 下一档需降时 | 当前最低 SQNR |
|---|---|---:|---:|---:|---:|---:|
| c1 | 16384 / 4096 / 8 / 8192 / 2 | 4.6300 | 17.6800 | 79 | 4.54% | 23.12 |
| c2 | 16384 / 4096 / 8 / 14336 / 2 | 7.9035 | 28.7370 | 78 | 3.35% | 23.14 |
| c3 | 16384 / 2048 / 32 / 2048 / 4 | 1.3420 | 6.9645 | 83 | 1.15% | 22.70 |
| c4 | 16384 / 2048 / 32 / 1024 / 4 | 0.7860 | 4.9490 | 86 | 5.92% | 22.98 |
| c5 | 8192 / 3584 / 64 / 2560 / 8 | 2.7435 | 12.1140 | 81 | 3.07% | 23.13 |
| c6 | 8192 / 3584 / 64 / 1024 / 8 | 1.2450 | 7.5670 | 85 | 1.06% | 22.93 |
| c7 | 16384 / 4096 / 96 / 2048 / 3 | 2.1780 | 10.3115 | 82 | 3.03% | 23.12 |
| c8 | 16384 / 4096 / 96 / 1024 / 3 | 1.2330 | 7.1950 | 85 | 5.01% | 22.90 |
| c9 | 4096 / 4096 / 256 / 2048 / 8 | 2.4545 | 7.7255 | 75 | 0.61% | 23.13 |
| c10 | 4096 / 4096 / 256 / 1536 / 8 | 1.8505 | 6.7115 | 78 | 3.59% | 22.69 |
| c11 | 65536 / 1024 / 32 / 1024 / 2 | 0.8720 | 5.6570 | 86 | 3.06% | 23.14 |
| c12 | 65536 / 1024 / 32 / 2048 / 2 | 1.4685 | 8.1535 | 84 | 2.02% | 22.90 |

模型 raw **81.8333**；它与官方最佳 raw 85.17 是不同统计对象。不能把模型差值直接加到榜面最佳值。

| 条件情景，固定本节 tb | 模型 raw | 相对模型基线 |
|---|---:|---:|
| c11/c12 各降 5% | 82.0000 | +0.1667 |
| c9/c10 各降 5% | 82.0000 | +0.1667 |
| c9/c10 各降 30% | 82.7500 | +0.9167 |
| 全案各降 30% | 86.4167 | +4.5833 |
| 全案各降 50% | 89.7500 | +7.9167 |

同百分比改善下，模型达到 raw 90 约需 **50.94% 降时**。这是目标差距说明，不是不可达定理；现有证据没有支持幅度接近它的新机制。**本轮应争取可复现的增量，不能把两条百分之几的假设包装成已经找到 4.83 榜分。**

每 rank 有用专家 FLOPs 为 `6·T·k·H·I`，FP8 权重容量为 `3·E·H·I` 字节。c1/c2 分别约 6.60 / 11.54 TFLOPs；c9/c10 权重分别 6 / 4.5 GiB。容量不是实测 HBM 流量，tile 重读可能命中 L2；数学工作量也不含 padding、路由和量化，不能用容量/tk 反推出真实带宽。

完整表与情景：[CSV](../reports/2026-09-30-v926-score.csv)、[JSON](../reports/2026-09-30-v926-score.json)。

## 5. 源码与正确性审计发现

用户所指的极低 tk 平台异常另见 [专项指引](PLATFORM_TIMING_ANOMALIES_2026-09-30.md)。以下是源码中的独立问题，不替代平台异常研究。

### 5.1 同一源码在不同调用序号执行不同算法

`run_kernel` 的全局 `_CALLN` 每次加一，不按输入 shape 或测试点重置。`_GA` 只在调用 3–5 且匹配家族时为真。源码上至少存在：

| 家族 | call 1 / 2 | call 3–5 | call ≥6 |
|---|---|---|---|
| c1/c2 | 路由和部分 GEMM/ACT 路径不同 | 融合 router、当前 FP8 MD/DN | 主稳态路径延续 |
| c3–c8 | 旧路由及 ACT 路径 | token-only GQ + gather MD | sorted-copy GQ +另一 MD 变体 |
| c9/c10 | 路由前两次不同；tiled MD 已生效 | token-only GQ + tiled MD/DN | 同一 tiled 主体，不受旧 GA 窗口限制 |
| c11/c12 | 旧 ACT/DN 路径 | sorted GQ + TMA MD + padded static DN | 主稳态路径延续 |

不能把“call 3–5 必然就是正式计时”当题面合同；题面也没有保证每个测试点新建 Python 模块。当前 AC 只说明平台已测调用通过，不能证明跨 2→3、5→6 和跨 shape 的字节一致性。

**正确性任务：** 建立每个目标家族从第一次调用就固定的计算图，仅允许静态缓存初始化与复用有区别；同一 X 在调用 1/2/3/5/6/9、不同 X 交替、shape A→B→A 时比较输出。编译总预算须实测。不要为了“避开检查/计时”保留或新建调用阶段闸门。

固定图未必更快：历史 stable-all 版本曾总 TLE；c5/c7 的 stable 子集已有 AC。复用其中实现思想时仍须按 v926 重做 diff，不覆盖静态 DN、tiled B 等已晋升机制。旧阶段时间不等于当前瓶颈比例。[历史固定图实验](../experiments/2026-09-08/notes/stable_profile_c5_c7.md)

### 5.2 静态权重缓存只用 shape，失效不完整

full weights、FP8、BNORM、interleave 等缓存并非都绑定源对象。相同 shape 的新权重可能命中旧结果。只给最外层加 id 不够，派生缓存应共享同一权重版本标识。

可先修复“新对象替换”情形：保留源 Tensor 强引用，检查 `is` 及 dtype/device/stride、shape、rank/world、topk、量化布局。强引用防止 id 被重用。但同对象跨点原地改写仍不可见；若每次调用又由代理生成新对象，identity guard 可能每次重建，性能合同同样不清楚。

**分布式额外约束：** 若只有一个 rank 判失效，而它独自进入 all-gather，其他 rank 命中缓存直接返回，会造成集合通信失配。任何修复必须说明四卡如何达成共同重建决策。不能只给本地 dict 换键后宣称完成。

用户已回复：主办方没有新答复。当前把“对象替换安全”和“支持任意原地改写”分开报告；不通过被禁的 `_version` 或其他拦截接口规避限制。详细历史边界见 [cache contract](../experiments/2026-09-12/notes/native_weight_cache_contract.md)。

### 5.3 MD 行尺度计算在每个 N tile 重复，并且同一地址多写

MD 的 `bound / bexp / s / inv` 只依赖当前 token 的 X scale、静态专家 BNORM、当前 branch 的 route weight，与 I 方向 tile 无关。但每个分支重复 **I/128 次**，`SCL[row]` 也被不同 tile 同值写回。

这不是 ACT 缓存机会；每次 X 都要重算这些量。它是**循环不变量外提与单写者重构**机会，具体方案见第 6 节。不能把同值多写直接等同于已观测的确定性故障，也不能把写请求字节全部算成 HBM 流量。

### 5.4 c2 存在稳态未使用的大分配

`_run_replicated` 的 direct-GQ c2 分支仍分配 BF16 `tokens_sorted[M,H]` 和 `_gateup_shadow[M,2I]`。该分支后续使用 FP8 q，shadow 无读取。c2 两块合计约 **2 GiB 的分配请求**，其中 shadow 1.75 GiB。

适合作为独立 host 清理候选移除，并验证相关调用路径。`torch.empty` 不等于实际写满 2 GiB，也不能因此宣称节省 2 GiB HBM 传输或 1 ms。当前清理版为了保留 AST 没有改这两条语句。

### 5.5 static DN 的 scale floor 与 q 没有同步补偿

当前静态链为 `q=FP8(acc·Dscale/C)`，`s=max(A_scale·C,1e-12)`。当 floor 激活且 q 非零，重建值不再对应 `acc·Dscale·A_scale`。

独立反例：`acc·Dscale=2, C=4, A_scale=1e-14`，q=0.5 可精确表示；存储 scale=1e-12，重建 5e-13，未 clamp 的物理值是 2e-14，相差 25 倍。**这是代数边界问题，不是已测 OJ 失败，也不是当前提速瓶颈。**

若修复，应让量化分子与实际使用的 s 配对，例如加入逐行 `A_scale·C/s` 补偿；只在 floor 活跃时走慢支，正常路径不改变。重新验证极小非零输入和最终 SQNR。仅全零输入无法暴露它，因为 q 也为零。

### 5.6 并列 top-k 与资源异常要各自归因

融合 router 使用最小 expert index 打破并列；PyTorch 不保证 tied top-k 索引稳定。历史样本兼容不是通用规则。固定图的正确性集要含 BF16 舍入后出现并列的 logits，并区分“路由选择改变”与“FP8 链误差”。不要只凭最终 SQNR 猜测误差全来自 Down。[PyTorch topk 文档](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)

`ptxas exit 255` 无 stderr 时不能直接判定 spill，TLE 无分段日志时不能判定某个 kernel 卡编译。保存原始错误、出现阶段、完整源码 SHA，最多一次针对明确原因的修补，不原样赌重发。

## 6. A 方向：GQ 生成行参数，MD 专注矩阵乘和逐元素收尾

### 6.1 首版只触达 c11

c11 的 `_gq1p_tm_kernel` 已经一次读取每个 token，量化成 FP8，并按 INV 写到两个 sorted branch 位置。它已经拥有当前 X scale；route ids/weights 和 BNORM 也已可取得。利用这次 launch 同时生成每条 branch 的 MD 参数，不新增一个“预计算 kernel”。

当前 c11 MD 对每条 branch 重复计算 8 次；c12 重复 16 次。c11 当前全部 MD scale 写请求约 4 MiB，单写约 0.5 MiB；c12 分别约 8 / 0.5 MiB。主要假设是减少收尾标量运算、同值 store 和寄存器活跃值，**不是减少主 GEMM FLOPs**。

选择 c11 首发因为它已有固定的 sorted GQ/TMA MD 主体且 K=1024，收尾更可能显露；c12 仅在 c11 有正信号后验证。c3–c8 要先处理 GA 路径差异。c1/c2 已有“收尾代数重结合变慢”的负记录，不先碰它们。

### 6.2 张量合同与公式

`branch=t*k+j`，`e=flat_ids[branch]`，`m=INV[branch]`。新 producer 写 sorted row `m`，不是 padded row，也不是 token row：

```text
a  = 当前 GQ 算出的 X row scale
w  = flat_weights[branch]
bn = BNORM[e]
bound = (((a*a)*bn)*bn)*abs(w)        # 保持原 FP32 运算顺序
bexp  = (bits(max(bound,1e-30)) >> 23) & 255
s     = float_from_bits((bexp+10) << 23)
inv   = float_from_bits((244-bexp) << 23)
AH[m] = a * 0.5
WI[m] = (w*a) * inv                  # 对应 c11 已折叠的收尾
ACT_SCALE[m] = s
```

AH/WI 各一个 FP32[M]；ACT_SCALE 复用当前 MD 输出 scale 的位置。每个 branch 恰好由一条 token/slot 路径写一次。`INV` 必须完整置换，空专家不写任何 branch。

MD 保留原 `acc*B_SCALE → reshape/split → h=gr*AH → tanh → silu → q=FP8(silu*u*WI)` 的运算顺序和当前 tanh 格式。移除 MD 中 BNORM/route weight/scale 指数计算及 SCL store。**不能顺便改变 tanh、权重量化、tile、stages、flatten 或 DN。**

第一版可以保留原 `w_sorted=weights[order]` 分配但不作为晋升版遗留；确认它不再有其他消费者后再去掉。更好的初版直接去掉 c11 对它的需求，并在 diff 中单列这是被 A 消除的依赖。

静态 BNORM 预处理提前到 GQ 之前，只移动依赖顺序，不重复计算。ACT_SCALE 在 GQ 前分配并传入 MD host；不要让 host 又分配一个空 scale 覆盖它。GQ→MD→DN 使用同一当前 stream，不加跨 CTA 旗标、自旋或浮点 atomic。

### 6.3 为什么不是旧失败方向

- 不是再次“把 scale load 从 K 循环前挪到后”；那是同一个 tile 内的生命周期调整。这里把每个 branch 重复 R 次的计算移出 N tile 循环。
- 不是 `_q8_blk2row`，不增加 ACT 全量读/写，不在 DN 的 K 循环中插入逐块 scale。
- 不是缓存旧 X 的 amax；所有动态量每次由本次 X 与路由重新生成。
- 不做 c1/c2 已失败的 V716 代数折叠；首版只对本来已采用折叠式的 c11。

在已查的当前源码、结果摘要和审计文档中未发现这个具体 producer/consumer 改造的实测记录；这不构成“历史所有文件都绝无同类代码”的保证，执行者立项前仍可按名称/公式检索。

### 6.4 成本与停止条件

增加 GQ 的 ids/weights/BNORM 查表、AH/WI 两块标量输出；删除 MD 重复的 bound/位操作/scale 写与旧 weights 重排依赖。GQ 的 CTA 一次只处理一个 token，新增标量散写可能效率低，不能仅按省下的运算条数推导加速。

前置 CPU 证明已经覆盖 **20 个 E/topk 组合、6205 条 branch 的 sorted-row 参数位模式**；见 [proofs.json](../reports/2026-09-30-v926-direction-proofs.json) 与 [脚本](../reports/prove_v926_directions.py)。它不证明 GPU 的 FMA、编译器重结合或实际 MD 输出 exact。

GPU 验收分三层：

1. 允许的 custom 中直接比较 AH/WI/SCL 与旧公式的全部有效行位模式；覆盖零 route weight、极小 scale、非整块尾行、空专家。不能只比较少量点。
2. 相同真实规模的 GQ→MD→DN→fin 全链比较输出，SQNR、finite、写覆盖、跨调用一致性全部过关。若当前 custom 不能容纳这种 workload，直接构建仅触达 c11 的完整 P1 候选，不把借用其他题接口当成豁免。
3. 两对独立 AB/BA 完整 P1，目标 c11 **≥1.5% 降时**，两对同向且未触达案无系统回退。若在 ±0.5% 内、方向相反或 GQ 成本吃掉 MD 收益，关闭。只在正信号出现后扩 c12，扩展单独确认。

至多两版实现；第二版必须修明确的索引、寄存器或生产者成本问题，不用扫参数延长该路线。对于新的全案正确性检查失败，先修复正确性，不能用更多 AC 次数抵消失败。

## 7. B 方向：c9 的 tile 连续、逐通道交错 GU

### 7.1 源码不对称与待验证机制

c3–c8/c11/c12 使用交错 GU 的单 accumulator；c9/c10 的 `_fgs_tma1_kernel_gq_tiled` 每个 K tile 仍执行两次 `[128,128]` B TMA load 和两次 dot，分别累加 gate/up。

旧 `merge_gq_c910` 在 v833 上约为噪声；旧 `c910_tma2_blockinterleave` 同时引入 sorted A 并只有 TLE。它们没有给出“**保留当前 token-only A 和 tile 连续 B，仅合并 GU 数据布局/累加器**”的有效性能结论。[旧实验范围](../experiments/2026-09-19/results/micro_pairs.md)

新假设是减少 TMA 请求/调度与双 accumulator 依赖管理，而不是减半权重字节。尤其 c9 权重仍是 6 GiB；如果主瓶颈是 HBM，本方向可能无收益。c1/c2 同类合并已慢，因此 B 只配小预算，不能把 wider dot 当成普适结论。

### 7.2 保留物理 tile 连续性

固定 BM=128，logical BN=128，BK=128，R=I/128，KT=H/128。将 GU 从：

```text
旧：[E, 2, R, KT, 128, 128]
新：[E, R, KT, 128, 2, 128]        # 倒数第二轴为 gate/up
```

新布局使一个 K tile 的 gate/up 共 256 个 B 行连续；scale 对应 `[E,R,128,2]`。它不是把未打包的 `[E,2I,H]` 直接交错后退回长步长 TMA。

对坐标 `(e,n,h,gu)`，定义 `r=n//128, ni=n%128, kt=h//128, ki=h%128`：

```text
flat_index = (((((e*R+r)*KT+kt)*128+ni)*2+gu)*128+ki)
B_DESC 视图：[-1,128]，block_shape=[256,128]
load_row = ((e*R+pid_n)*KT+kk)*256
acc[128,256] += dot(A_tile, B_DESC.load([load_row,0]).T)
scaled = acc * B_SCALE[e, pid_n, :, :].reshape(256)
g,u = split(reshape(scaled,[128,128,2]))
```

静态打包先保持原 `_quant_weight_fp8` 的量化 q/scale；只是换布局。BNORM 必须沿原物理权重定义计算，不能在打包后误把末两轴作为 K。DN 权重与 c9 动态 DN 完全不动。first version 保持当前 EPP=2、同 K 累加次序和后续 scale 算法。

预热只分配一个新 GU 最终布局，按原四 rank 的 gate/up all-gather 顺序逐段填充对应 gu 轴，避免同时保留原/新完整 GU 两份造成峰值显存上涨。切换缓存布局时使用不同布局键；不能命中旧布局 Tensor。

本轮对 I=1536/2048、H=4096、E=256 做了 **80,000 个打包/反解坐标检查**，包括 tile 起点与容量；这不证明 TMA 运行或 wide WGMMA 正确。[CPU 证明](../reports/2026-09-30-v926-direction-proofs.json)

### 7.3 资源账不能省略

合并前后每 CTA 累加器均为 32768 个 FP32 值；按 8 warps 分摊，仅 accumulator 即平均 128 个 32-bit 寄存器/线程。每 stage 的 A 为 16 KiB、GU 为 32 KiB，四级约 192 KiB，尚未计编译器额外 scratch、barrier 与布局转换。不要额外放大 BM/BN/BK 或添加 epilogue 缓冲。

保持现役 grid=132、w8/s4/maxnreg=232、GROUP_M=32。实际寄存器/smem、WGMMA lowering 和流水线结果由编译器决定，纸面相等不代表二进制相等。Hopper 的资源上限可参考 [NVIDIA 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)，不能把 H100 的参数表直接当本比赛实测吞吐。

### 7.4 验收与预算

先只 c9，验证 packed q/scale 与原权重逐坐标相等，再比较 MD、最终 output。所有有效行/尾行/零专家、原 token/slot 归并都必须正确。不要将两种 accumulator 的小数位相同 SQNR 当成位相同。

在允许的同版本 custom 获得完整子链正信号后，或 custom 不支持时直接用一个完整 P1 候选，做两对 AB/BA。**端到端 c9 至少 2% 降时、两对同向**才继续；随后 c10 单独确认，保留它已经晋升的 static DN。

如果 B load 请求减少但全案中性/变慢，则该假设被否定；不继续调 gran/warps/stages。首次 TLE/ptxas 失败只允许一次能解释原失败的最小修补。没有 stderr 时不要声称只是编译冷缓存，也不要反复投相同源码。

## 8. 探索额度如何分配

预算是本轮建议的停止纪律，不是用户必须花满的额度：

| 项目 | 最大试验范围 | 继续条件 |
|---|---|---|
| P：平台极低 tk | A/B/C 三种源码对照，最多 6 发；复用清理版确认 | 识别源码身份/JIT 变化与首次异常的关联；无新信号不扩量 |
| 清理/固定路径及合同验证 | 清理基线确认可并入 P；有合同证据才做缓存修复；固定图先一个家族 | 任意调用正确、无集合通信失配、编译总预算可用 |
| A | 最多两版；c11 最多两对，正信号才加 c12 两对 | 第 6.4 节门槛 |
| B | 最多两版；c9 最多两对，正信号才加 c10 两对 | 第 7.4 节门槛 |
| 最终合并 | 一个候选、两对完整确认；若已有分支失败不硬合 | 组合完整输出正确且增益未被抵消 |

若只有正式 P1 可测，一对 A/B 即两次提交。第一次完整正确性提交如为已冻结同一 SHA 且实验顺序预先确定，可以计入配对；不要重复制造“初筛池”和“确认池”来消耗次数。若版本改动，重新记录新的比较对象。

阶段诊断优先使用当前允许的完整 custom workload tk。只有完整 P1 时，可以有限使用“多执行一次同一纯写覆盖阶段”的输出保持探针，且始终计算完整结果；差分只叫**重复阶段的有效增量**，含缓存/调度干扰，不能当物理 kernel 时间。禁止跳过检查调用、故意返回旧输出或通过异常文本绕过 Event 拦截。

任何路线如果预期只有 0.2% 却需要几十对才能分辨，就不符合剩余时间约束。结构 A/B 都失败时，冻结可靠版本；平台异常 P 依据自己的预设预算收口，不用微扫或无期限重复投递替代新证据。

## 9. 提交与判断协议

唯一平台执行者遵循 [SUBMISSION.md](SUBMISSION.md)。先查在途任务、令牌池与 custom 模式；只读查询不消耗提交额度。提交前冻结路径/SHA、触达案、diff、预期机制、停止条件。已有持续上传授权，不例行追加确认。

建议每条结果按如下字段留档：

```text
label, source_sha, anchor_sha, touched_cases, mechanism,
SID_or_CID, status, submit_time, raw, per_case_tk, per_case_tb,
per_case_min_SQNR, determinism, errors, paired_control, decision
```

分数与 tk 分开看：同窗顺序 A-B-B-A；原始 tk 差与未触达案变化一起报告。不能因 raw 更高就判赢，不能使用多年旧机器系数强行修正。样本少时报告逐对数据和范围，不编造显著性。

异常处理：`tk<=0`、不合理突降、未触达案同步巨变、非 AC、缺案，先标异常并保留原数据；不纳入提速估计。Canceled 是取消；本地轮询超时需查同 SID，不能据此再提交。

数值验收至少含：当前 X 改变、相同 X 反复、不同调用序号、零与极小值、路由并列、每专家 count=0/1/127/128/129、高度倾斜、全部输出写入、输入不变。测试点 static 参数变化也要验证；**未取得缓存合同的部分明确保留为限制**。

10 月 1 日 18:00 后不引入新内核体系；21:00 前冻结候选，留出正式队列和总编译预算。每次晋升更新 README 与实际 SHA，不能只改旧指引中的一句“当前”。

## 10. 查阅资料后的可复用结论

外部资料用于机制和接口边界；性能数据不直接移植为 H800 OJ 预测。比赛页本次网页正文读取失败，题目与官方答复以保存原文和只读 API 为准。

| 一手资料 | 本项目可复用的内容 | 本轮不据此假设的能力 |
|---|---|---|
| [Triton v3.4 persistent matmul](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/tutorials/09-persistent-matmul.py) | persistent 循环、descriptor、前言/收尾生命周期、epilogue 分块的具体代码；可以按 tag 对照，而非抄 main | 该版本代码对自动 warp specialization 有架构限制；不能直接当 P1 Hopper 的可用提速开关 |
| [v3.4 Gluon Hopper 导出](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py) | 复核 TMA、mbarrier、fence 接口边界 | 不能据最新版 Gluon 教程要求线上具有显式 WGMMA 或 async_task |
| [NVIDIA Hopper 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | TMA 可减少寄存器/SM 数据搬运负担；共享内存与寄存器共同约束并发 | “硬件支持 cluster”不等于该 fork 可正确编译生产 kernel |
| [CUTLASS Hopper builder 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) | mainloop/epilogue/scheduler 需一起选择；解释为何不能只按 load 次数判断 B | 不在提交中调用禁止的高层 GEMM，不另建 CUDA 工具链 |
| [CUTLASS Hopper FP8 grouped/blockwise scaling](https://github.com/NVIDIA/cutlass/blob/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling.cu) | grouped 布局和 scale 是共同合同；可参考静态打包与 shape 分发方法 | 不把 Blackwell 的原生 block-scaled 路线套到 H800 |
| [TMA-Adaptive FP8 Grouped GEMM 论文](https://arxiv.org/abs/2508.16584) | 尾块通过 descriptor pool 与分阶段访问处理，说明“padding 消除”需要完整访问设计 | 本项目 BM64/tail 已有负结果，论文不构成重开理由；deadline 前不移植整套 descriptor pool |
| [PyTorch topk](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html) | tied indices 无稳定性承诺，需包含并列路由检查 | 不能据“usual behaviour”宣称最小索引与 torch.topk 永远一致 |

当前最明显的限制是已有 FP8、融合量化、padded DN、tiled B、短 K flatten 已经吃掉很多常规空间；缺乏 profile 并不自动意味着尚有一倍速度。上表帮助排除不适配的版本路线，而不是用更多论文标题替代成本账。

## 11. 给其他 coding agent 的任务卡

### Agent P：极低 tk 平台异常

> 用户明确要求研究平台极低 tk 以提分。先读 [专项任务书](PLATFORM_TIMING_ANOMALIES_2026-09-30.md)，复算 187 发连续样本及 4 组同 SHA 对照。重点检验首次源码 7/58、重复源码 0/67 的关联，不把首次 SHA 当已证冷 JIT。可按专项第 6 节最多 6 发 A–B–C–C–B–A，区分只改模块头部与保守清理后的首次/重复效果，保持完整数学和输出。结果中分别报告平台保留分数、异常命中和正常性能，失败零值排除；无新信号即停。平台操作统一交唯一执行者。

### Agent A：行参数 producer/consumer 改造

> 阅读 README、AGENTS.md 与本指南第 5–6 节。当前清理版 SHA eb5ffee8db40…，已测锚 p1/references/kernel_v926_measured.py 的 SHA 2633cc995eb2…。先只 c11，把现有 GQ 的本次 X scale、INV、route ids/weights、BNORM 转成 sorted AH/WI/ACT_SCALE，并让 MD 消费；保持原逐语句运算顺序、TMA 布局、tile、stages、tanh 和 DN。每条 branch 单写 ACT_SCALE，不新增整量 ACT 读写，不用调用次数识别评测阶段。先运行 CPU 证明，再给 GPU 全链正确性与两对 AB/BA；c11 降时不到 1.5% 或两对不同向即停。不要覆盖生产，提交独立候选、SHA、diff、触达案、producer 成本与端到端结论。平台提交统一交唯一执行者。

### Agent B：c9 tile 连续交错 GU

> 阅读本指南第 7 节及历史 merge_gq/c1-c2 负结果。只改 c9 GU 为 [E,R,KT,128,2,128]，scale 同步交错；保持 token-only A、原量化/BNORM、DN、K 次序、BM128/logical BN128/BK128、w8/s4/maxnreg232。一个 [256,128] B load 与一个 [128,256] accumulator 取代两路；不要退回未打包 B 或切 sorted A。先验证 80,000 坐标证明，再比较完整 MD→DN→fin 与最终 P1。两对 c9 不到 2% 或无法复现则停止；有正信号再单独扩 c10，保留其 static DN。最多一次针对明确错误的修补，不能无日志反复赌 TLE。交独立文件、SHA、内存峰值、正确性及完整耗时，不覆盖生产。

### 正确性审查与平台执行者

> 先核对源 SHA 和平台最近终态，严禁多个任务同时取提交令牌。审查 _CALLN/_GA 的跨调用一致性，cache 全派生失效与全 rank 重建协议、static DN floor 补偿，以及新布局索引。官方未确认的合同不能写成保证。所有性能候选按完整 run_kernel 验收；保存异常 tk 原始记录但不据此晋升。统一维护候选/锚配对与停线预算，冻结后只合并已确认正收益。

这些任务卡供用户分派，不表示已经启动对应 agent。

## 12. 工作区索引与回退

- `p1/`：当前清理源码、两个实测 reference、说明文件；不再混放上千份实验。
- `experiments/YYYY-MM-DD/`：独立候选、结果、note；本轮 CPU 分析放 `reports/`，没有虚构 GPU 结果目录。
- `docs/`：当前任务书、平台异常专项与操作约定；STATE/CODE_MAP 已改为当前摘要，原文完整保存在 `archive/historical-docs/`。
- `archive/p1/`：原 p1 中 **1552 个历史文件**；内容保持原样，逐文件 SHA 已复核。
- `archive/guides/`：4 份旧优化指引原文，保留历史决策；不是当前指令。旧文内相对路径基于当时根目录，查文件以迁移索引为准。
- `archive/WORKSPACE_MOVES_2026-09-30.csv`：**1556 个迁移条目**，记录旧路径、新路径、SHA、大小、时间。未删除历史候选或原始测量。
- `logs/`、`benchmarks/`、旧实验与 `work/repo/` 保留，`.secrets/` 未改、未输出。

检索旧候选：

```bash
python3 scripts/find_candidate.py kernel_v800
python3 scripts/find_candidate.py 436f0a227678
```

回退单个历史文件：先查迁移清单和 SHA，确认旧目标路径不存在，再移回；不要覆盖新工作。回退生产可以从已测 reference 复制到 `p1/kernel.py`，随后复核 SHA，并更新 README。完整恢复迁移前目录不是日常接手所需。

## 13. 尚需主办方提供的信息

最关键的是**测试点边界和静态权重失效合同**：跨点是否换 Tensor/存储；同点是否稳定对象；是否允许同一对象原地改写；有没有公开版本或边界信号。这个答案关系到正确缓存与多卡 collective 的共同决策，不只是性能细节。

用户确认主办方对此尚无回复。其次是精确 Triton/triton-dist fork、PTXAS stderr、合法分段 profile，以及异常专项所需的采集完整性/worker 信息。没有这些仍可按本指南做完整 OJ 对照；不为等待 profile 停止所有工作，也不把不存在的诊断接口写进执行前提。
