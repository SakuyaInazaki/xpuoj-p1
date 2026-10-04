# P1 最后冲刺优化指引：以 v12 为锚，分开验证结构收益与计时异常

> 2026-10-01 更新：待办已由 [283 发证据版任务书](OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)替代。本文件保留历史推导；R1/R3 已有新失败记录，R2 需覆盖 MDg，新增 J、S1 和 I constexpr 实验请读新任务书。

证据截点：**2026-09-30 22:34–22:37，Asia/Shanghai**；**23:03 交付复核，分数与最新 SID 未变**。连续队列至 **SID 152586**。本文件取代旧 v926 指引的“下一步任务”，供用户分派其他 coding agent。截止目标：**2026-10-01 23:59 前争取 raw 90 / net 80**。

## 1. 决策摘要

**目前距离榜面目标只差 0.92 分，但正常性能距离 raw 90 仍很远。** 当前最高 SID 152238 为 **raw 89.08 / net 79.08**；同一源码的正常复测 SID 152241 为 raw 82.00。最高记录包含 c8–c12 极低计时，不能把 89.08 当作可稳定复测的算子性能。

本轮建议按以下顺序投入剩余时间：

| 顺位 | 任务 | 首发范围 | 交付与继续条件 |
|---|---|---|---|
| P0，立即 | 停止重复旧 A/B 和注释／改名抽样，冻结 v12 与新证据 | 全项目 | 旧 A 已晋升；旧 B 已无收益；44 次中性源码探针零命中 |
| R1，主线 | persistent GEMM 分开维护读入与写出 tile 计数器，缩短跨 K 循环的索引依赖 | **c11 的 static DN** | 两对正常结果同向，目标案 ≥1.5% 降时；有信号优先扩 c6，见积分账 |
| R2，主线 | MD 写入 padded ACT，DN 直接读取该布局，消除 MD 尾块条件 store | **c4** | 先布局版，再有条件做短 K flatten 版；整案 ≥2%，至多两种实现 |
| R3，备用 | 把 static DN 的 DSCL 表分解成动态行 scale 与静态专家/chunk scale，由 fin 重建 | **c11** | 保持量化与 floor 不变；整案 ≥1.5%；只在 R1/R2 无信号或有余量时执行 |
| P，伴随工作 | 记录真实结构候选的首次／重复、JIT 函数文本与行号变化，解释 AC 异常 | 现有提交与上述候选 | 不另开无限“刷新 SHA”队列；出现异常后同 SHA 正常复测一次，分别入账 |

三条结构方案都是**尚未取得 GPU 正收益的可检验假设**。R1 有同版本 Triton 官方代码范式支持；R2 有本项目 padded DN 的成功经验支持；R3 有当前 static DN 的明确代数冗余支持。它们比再扫 tile/stages 更具体，也不能承诺补足 0.92 榜分。

本次接手完成了只读平台核验、257 发连续样本分析、当前源码审计、历史排重、上游资料核查和 CPU 合同验证。**新增正式提交 0 次，custom 0 次；生产 kernel 未改；没有代替用户启动其他 agent。**

## 2. 基线、最新状态与交接纠错

### 2.1 只认文件 SHA 与正式记录

| 对象 | 路径 | SHA-256 / 证据 |
|---|---|---|
| 生产 v12 | [`p1/kernel.py`](../p1/kernel.py) | `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9` |
| 同内容冻结候选 | [`p1_dirA_c34_v12_far_meas.py`](../experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py) | 同 SHA；152238 异常 AC，152241 正常 AC |
| 老实测锚 v926 | [`kernel_v926_measured.py`](../p1/references/kernel_v926_measured.py) | `2633cc995eb2563e22c1a212df1256141bdc1b520fe841d4dd4c15c0d382e7bf`；用于回溯 A，后续性能对照首选 v12 |
| 回退 v890 | [`kernel_v890_measured.py`](../p1/references/kernel_v890_measured.py) | `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc` |

22:34 的平台只读快照：P1 **79.08，3496 次提交，最佳 SID 152238**。账号总分 250.08、总榜第 3；这不是 P1 单题名次。最近十条均为终态。P1 `triton-dist` custom 仍 `available=false`。见[平台原始快照](../reports/2026-09-30-late-platform-readonly.json)。截点后的队列变化须由执行者重新核对。

**23:03 交付复核**：上述分数、提交数、最新 SID 和 custom 状态均未变，见[交付平台快照](../reports/2026-09-30-sprint-delivery-platform-readonly.json)。

### 2.2 旧指引中不应再执行的待办

1. **旧 A：GQ 生成 AH/WI/ACT_SCALE 已完成。** c11/c12 有约 1.6%–2.4% 的受控收益，之后扩至 c3–c8。但 `_dir_a` 仍要求 `_GA==0`，c3–c8 的 call 3–5 并不走新 pre 支路。不能把扩展后的所有调用都算成 A 已生效。当前 c2 无用分配清理也已经完成。
2. **旧 B：tile 连续交错 GU 单 accumulator 已做。** v14 SID 152485 首次 TLE，152488 同码 AC，c9/c10 为 2.448/1.869 ms，没有可分辨收益。不要再次把它交给 agent 当新主线。[记录](../experiments/2026-09-30/notes/dirB_wide_negative_20260930.md)
3. **A 扩到 c1/c2 的 v13 未晋升。** 152289 高分含异常，152295 正常复测没有支持这条扩展取代生产。不要用它的 87.58 分证明 c1/c2 更快。
4. **最新 c9 pre v15 也不是未做资产。** SID 152541 TLE，152547 AC，c9/c10 2.465/1.891 ms；后续中性变体未给出明确收益。只读证据不足以做严格两对判负，但不值得原样扩投。[候选](../experiments/2026-09-30/candidates/p1_c9_pre_v15_meas.py)
5. 旧异常指南的 **A–B–C–C–B–A 六发计划已经过时**：新版已有大量源码等价探针，且“模块头部注释只改整文件身份”的前提有问题，见第 4 节。

## 3. 题目与执行约束

[完整题面](../1-full.md)要求四卡 MoE：BF16 router logits、FP32 softmax/top-k/归一化；专家 gate/up、SwiGLU、路由加权、down；分支以 FP32 累加，最终写 BF16。SQNR ≥22 dB、输入只读、输出有限且完整、同输入独立调用逐字节一致。每案至少一次不计时预热，随后更换 X；只能复用静态参数派生数据与缓冲容量。

现役十二案都走 **replicated weights**：预热收集全专家权重，稳态每卡只处理本卡 X。稳态已经没有 token 跨卡 dispatch/return，不能把“去掉通信”再次计入收益。

用户提供的[最新讨论原文](</Users/sakimi/Desktop/addinfo/addinfo.md>)明确 P1 为 Triton/triton-dist **3.4 系**、4×H800 80G SXM、默认 700 W、任意两卡 NV8。精确 fork/driver 未知。500 秒覆盖整批 torchrun，不是每个 kernel 500 秒。官方答复说“十个点”，正式记录实际有十二案，按真实案号分析。

NVSHMEM 已初始化；允许的通信入口走 `triton_dist.utils`，设备端通信用 `triton_dist.jit`，不得重复 init/finalize。P2/P3 的 Triton 3.6 和 CUDA stream 回答不能套到 P1。当前不依赖 Event、自建远程 sandbox 或本地 H800；验收只用正式 OJ／平台实际开放的 custom。

赛事原文规定截止后会复测性能与正确性，实时榜与最终结果可能不同。**保留最高分提交，同时保留正常性能与可解释的源码记录**，是异常分析的实际交付要求。[赛事原文](../比赛信息/xpuoj-d-31-deliverable-20260806/full.md)

## 4. 极低 tk：新证据、重要纠错与剩余价值

### 4.1 连续队列，不只取高分样本

只读恢复 SID **149486–152586，共 257 次 P1 提交**：186 AC、39 WA、21 Canceled、11 TLE。旧 187 发快照保留不动，本轮新增 70 发。

极低门保持旧口径：单案 Accepted、`pass=true`，且 `tk < v926 对应案正常值的 50%`。失败案的零值完全排除；整题异常成功必须十二案完整 AC。

| 分组 | 完整 AC | 含极低 tk | TLE |
|---|---:|---:|---:|
| SHA 在本段首次出现 | 109 | 11 | 11 |
| 本段已有相同 SHA | 77 | 0 | 0 |

本日新增的 **44 个 `anom_probe` 中性源码变体均 AC，极低值命中为 0**。这些变体包含模块注释、JIT 内注释和局部重命名；不是完全同一种干预，不能当作 44 次严格独立同分布试验。它们足以否定“继续随便改名就很可能提分”的工作优先级，不能证明异常概率严格为零。

[连续原始日志](../reports/2026-09-30-late-cohort-readonly.json) · [逐发台账](../reports/2026-09-30-late-ledger.csv) · [可复算分析](../reports/2026-09-30-late-strategy-analysis.json)

### 4.2 本日四个完整 AC 异常

| SID | 变更 | raw | 明显异常 tk，ms | 同源码复测 |
|---|---|---:|---|---|
| 151895 | v7：A 的 flatten/dummy-store 兼容版 | 85.58 | c10=.658，c11=0，c12=0 | 151899 正常 |
| 152205 | v11：non-flatten TMA pre | 87.75 | c9=.859，c10=.063，c11=0，c12=0 | 152219 正常 |
| 152238 | v12：far regular-A pre | **89.08** | c8=.468，c9=.062，c10=0，c11=0，c12=0 | **152241 正常** |
| 152289 | v13：c1/c2 pre | 87.58 | c9=.856，c10=.062，c11=0，c12=0 | 152295 正常 |

151885、152193 虽然某些通过案也出现零值，但整题 WA，不算成功获取异常分数。日志中的 SQNR 和 determinism 行仅证明对应检查通过，不证明零时间是真实性能。

异常从旧 c10–c12 扩展到了 **c8–c12**。出现过 `.06 ms`、约一阶段残余、随后零值的梯度，这更像计时采集或汇总异常，而非真实加速。案号顺序与返回 JSON 顺序不是 GPU 时间线，不能据此断言固定秒数之后 profiler 停止。

### 4.3 新发现：函数行号也是 JIT 身份的一部分

上游 **Triton v3.4.0** 的 `JITFunction.cache_key` 使用函数源码和依赖哈希，并追加 `starting_line_number`；该行号来自 `inspect.getsourcelines(fn)[1]`。本地 triton-dist wrapper 委托给 Triton 的 JITFunction。**因此函数 AST 相同、函数内部文本相同，都不足以证明缓存键相同。** 线上精确 fork 未知，此处是上游实现证据。[3.4.0 源码](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/runtime/jit.py) · [本地 wrapper](../work/repo/python/triton_dist/jit.py)

当前文件的实际行号变化：

| 版本 / SID | `_dn_tma2_f8_pad_static_kernel` 装饰器行 | tiled static DN 装饰器行 |
|---|---:|---:|
| v9 / 151912 | 7313 | 7406 |
| v11 / 152205 | 7406 | 7499 |
| v12 / 152238 | 7496 | 7589 |
| v13 / 152289 | 7647 | 7740 |

新增 MD 函数插在模块中段，后面的 DN 即使内容未改，行号也会变。这是“修改一处却多个家族首编译”的合理混杂因素。它解释了为什么不能只记录整文件 SHA，也不能把“c9/c10 数学未改”当作“它们没有编译变化”。

**但行号变化不是已证触发器。** v12 的头部注释变体 152248、JIT 注释变体 152251，以及其他多个移行变体均正常；v14/v15 新结构首发只 TLE、复测正常。新 JIT、位置变化、源码体积、调用路径、worker、编译耗时均可能混杂，现有数据没有因果结论。

### 4.4 接下来怎么研究，哪些不值得再做

**默认不再单独提交注释或改名探针。** 用 R1/R2/R3 这些真实完整计算候选作为后续观测来源，每发记录：

```text
整文件 SHA；每个 JIT 的文本哈希、AST 哈希、起始行；
实际被修改的家族；新增启动配置／constexpr；首次还是重复；
12 案 tb/tk/th/pass、双 SQNR、determinism、终态、SID；
正常性能结论；异常最早案号与零值集合。
```

构建候选时优先把新增 helper **追加到文件尾部**，控制未改函数的行号漂移；已有 helper 的修改可保持行数或用可审计的空行调整，但不用大量无意义内容制造冷编译。追加定义必须确认没有覆盖旧名字，模块加载后引用才执行。性能确认重复提交完全相同的候选文件，途中不做格式清理。

本轮源码索引的文本哈希取 AST 所界定的 `def` 至函数末尾，起始行另含装饰器位置；它是可审计的比较指纹，**不是线上 JIT cache key 的完整复刻**。依赖函数、全局常量、启动选项和编译器版本仍需一起记录。

若主办方愿意协助，最有辨别力的是让其对 **152238 / 152241 同 SHA** 提供：每案每 rank 的有效 kernel activity 数、零时间戳数、丢记录数、编译/预热/计时边界、实际运行次数与汇总规则、worker 和 CUPTI/driver 版本。NVIDIA 文档明确存在活动记录丢失和起止时间戳均为零的情况；这只是诊断候选，未证实 XPUOJ 使用哪种采集实现。[CUPTI 时间戳定义](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel9.html) · [丢记录接口说明](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)

只有得到新诊断线索后才值得另配一个最多 **4 发**的受控实验。若只是区分文件身份与 JIT 身份，文件身份对照必须在**文件尾部、所有函数外**加注释，并证明各 JIT 文本与起始行都不变；不要再用头部插行。既有证据已经不支持重复原先的六发计划。

保持每次完整计算当前 X、同一语义与正常 stream 依赖。不要用额外编译负载、睡眠、采集器干预、故意异步早返回、旧 activation/output 缓存或调用阶段识别制造低值；这些没有真实结构收益，也无法提供可解释的复测结果。

## 5. 分数账：0.92 分到底差在哪里

当前日志的 `th=0`、正常计分区间可复算为 `q=floor(100*tb/(tb+tk))`，raw 为十二案 q 的均值。本轮已核对全部 **186 个 AC** 的公式与显示分，均在两位小数舍入范围内吻合。官方说明更高区间可以超过 100，本表不外推那些区间。

正常参考取 **152241、152248、152251** 的逐案 tk/tb 中位数。这三份文件完整模块 AST 相同；注释与行号不同，**n=3 仅用于描述分数门槛，不替代未来候选的同窗对照**。

| 案 | T / H / E / I / k | tk ms | tb ms | q | 下一整数档需降时 | 最低 SQNR |
|---|---|---:|---:|---:|---:|---:|
| c1 | 16384/4096/8/8192/2 | 4.600 | 17.682 | 79 | 3.90% | 23.12 |
| c2 | 16384/4096/8/14336/2 | 7.857 | 28.736 | 78 | 2.78% | 23.14 |
| c3 | 16384/2048/32/2048/4 | 1.348 | 6.938 | 83 | 1.96% | 22.70 |
| c4 | 16384/2048/32/1024/4 | .795 | 4.943 | 86 | 7.09% | 22.98 |
| c5 | 8192/3584/64/2560/8 | 2.760 | 12.071 | 81 | 4.00% | 23.13 |
| c6 | 8192/3584/64/1024/8 | 1.246 | 7.471 | 85 | 2.39% | 22.93 |
| c7 | 16384/4096/96/2048/3 | 2.180 | 10.252 | 82 | 3.68% | 23.12 |
| c8 | 16384/4096/96/1024/3 | 1.232 | 7.157 | 85 | 5.43% | 22.90 |
| c9 | 4096/4096/256/2048/8 | 2.439 | 7.724 | 76 | 5.41% | 23.13 |
| c10 | 4096/4096/256/1536/8 | 1.857 | 6.679 | 78 | 4.39% | 22.69 |
| c11 | 65536/1024/32/1024/2 | .858 | 5.668 | 86 | 1.29% | 23.14 |
| c12 | 65536/1024/32/2048/2 | 1.446 | 8.134 | 84 | .73% | 22.90 |

模型正常 raw **81.9167**。全案各降 5% →82.6667，降 10% →83.3333，降 30% →86.4167，降 50% →89.75。小优化不能直接让普通窗口达到 90。[完整成本/门槛 CSV](../reports/2026-09-30-late-score.csv)

最佳 152238 的整数分为：

```text
c1…c12 = 79,78,83,87,81,86,83,93,99,100,100,100
合计 1069；raw 90 需要合计 1080；差 11 个单案整数分。
```

| 条件情景 | raw | 意义 |
|---|---:|---|
| 保持最佳那一发的其他值，仅再令 c8=0 | 89.6667 | c8 还能赚 7 个整数分，仍不够 |
| 同上，c8+c9 都变 0 | 89.7500 | 两案只剩 8 个整数分空间，仍差 3 |
| 同上，仅再令 c7=0 | 90.5000 | 更早案出现异常才足以显著越线；尚无这种完整 AC 实证 |
| 采用正常中位 tb，令 c8–c12 全为 0 | 89.4167 | 没有最佳那发的其他有利读数，也不够 90 |
| 正常中位 tb，令 c7–c12 全为 0 | 90.9167 | 仅是假设上界情景，不是触发方案 |

所以“已经 89.08，再把最后一个零点补齐就到 90”不成立；c10–c12 已经都是零。正常收益不能直接加到历史最佳分数上，未来提交必须在同一发里同时体现所需改善。数值以原始 tk/tb 重算，不能按四舍五入后的显示分简单累加。

**这也约束扩展顺序。** c11 首发只是低成本验证结构机制；若未来重现最佳那类异常，c11/c12 已经100分，继续只优化这两案不会额外增加当发榜分。R1 有信号后优先移到同样 short-K、GM8/s4 的 **c6**，再考虑 c4/c8；c12 是改善正常性能的可选扩展。R3 同理。每单案一个整数档只值 **1/12 原始分**，不要把一处2%正常降时包装成0.92分的确定解法。

## 6. 瓶颈判断：把力气花在尚未吃掉的成本上

每 rank 有用专家 FLOPs 为 `6*T*k*H*I`，FP8 专家权重容量为 `3*E*H*I` 字节。按完整 tk 折算，c1/c2 约 **1.43/1.47 PFLOP/s**；c9/c10 的权重各 **6/4.5 GiB**。这说明大案已不是“尚未用上 tensor core”的底盘。

| 家族 | 当前结构与主要压力 | 本轮选择 |
|---|---|---|
| c1/c2 | 大 K、双 GU accumulator、长 DN；约 6.60/11.54 TFLOPs；A 扩展和宽 GU 已无收益 | 暂不主攻；不要用通用大 tile、再合并 dot 作为新路线 |
| c9/c10 | 每专家平均 M=128，权重体量大，tiled B 已启用；compact packed 路线已胜 padded DN | 不再做 B、EP2、BM64；保留正常锚与异常观测 |
| c3/c4 | TMA-A MD，当前 pre_nf 含整块/尾块 store 分支；DN 已 padded | R2 先 c4，目标是消除控制流障碍，而非再包装一次 DN padding |
| c6/c8/c11/c12 | short-K DN 与 epilogue 占比更可能显露；static scale、flatten、TMA 已在 | R1/R3 先 c11，再逐案扩展 |
| c5/c7 | DN 仍用动态输出量化，先前 static DN 未晋升 | 不把 R3 自动扩到这两案，避免消耗精度余量 |

权重容量/tk 不是实际 HBM 带宽；FLOPs/tk 不是单 kernel tensor-core 利用率。当前没有合法在线 profile，不能用这些估算编造阶段比例或“物理不可能性证明”。新的方案必须解释减少什么执行成本，以及新增了什么成本。

## 7. R1：读写 tile 计数器分离，先只改 c11 static DN

### 7.1 为什么值得做

当前 [`_dn_tma2_f8_pad_static_kernel`](../p1/kernel.py:7497)用同一套 `tile_id → expert/row_begin/local_m/pid_n` 索引贯穿 K 循环及 epilogue。它已经 `flatten=True`，却没有采用 Triton v3.4 官方 persistent matmul 的**独立 epilogue 计数器**。官方范式旨在打断 prologue 与 epilogue 的索引依赖；它不要求换编译器、Gluon、warp specialization 或 cluster。[同版本官方实现](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/tutorials/09-persistent-matmul.py)

假设：让下一 tile 的读入索引不必等待当前 tile 的写出索引生命周期，减少 flatten 生成的依赖／寄存器负担。**不是“再开一次 flatten”或“外层 num_stages=2”。** 历史 v351 外层 stages 无效、v369 手工合并 K/tile 循环慢约 15%；本方向既不把 metadata 放入每个 K 迭代，也不改变 K 循环和累加顺序。

本轮按 `tile_id_c`、prologue/epilogue、重算 metadata 检索现役及 archive/p1，未发现同一独立计数器实现；这是限定范围的排重结论，不是全历史完备证明。

### 7.2 最小实现

从冻结 v12 复制为独立候选，新增 c11 专用 DN helper，原 helper 保持供其他案调用。host 门控 `(T,H,E,I,k)==(65536,1024,32,1024,2)`；这是几何分发，不新增调用阶段门控。

```python
# 伪代码：不是可直接提交的 Triton 实现
write_tile = pid - num_pid
for read_tile in tl.range(pid, total_tiles * Ntiles, num_pid, flatten=True):
    read_meta = decode(read_tile)    # 原专家查表 + 原 swizzle
    acc = original_K_loop(read_meta)

    write_tile += num_pid
    write_meta = decode(write_tile)  # 独立重建，只在 K 循环结束后
    original_epilogue(acc, write_meta)
```

`decode` 必须完全复制原规则：未 swizzle 的 `pid_m` 查 `expert_ids/tile_num/tile_cum/split_size_cum`，计算 local_m，再用 `tl.swizzle2d(local_m,pid_n,t_num,Ntiles,GROUP_M)`。写出所有地址、`B_SCALE`、`C_SCALE`、`A_SCALE` 和 mask 都使用 write_meta；不能混用 swizzle 前后位置。

首版保持 **BM128/BN256/BK128、w8、s4、GM8、FP8/F32、现有 static scale/floor**。仅重建写出所需索引；不提前搬 AH/WI，不加新 metadata kernel，不改 tiled weights，不扫资源参数。

对每个 CTA，第 j 次迭代都有 `write_tile == read_tile == pid+j*num_pid`。CPU 已验证 120 组含空专家、0/1/127/128/129/255/256/257 行、尾 group、不同列块与 grid 的映射，273,840 个 tile 访问一致。**CPU 不能证明编译器真的解除依赖。**

### 7.3 代价、验收与止损

代价是第二套标量索引运算和 metadata load。若编译器把它们 CSE 回原依赖链，就可能完全无效；若保留过多值，也可能更慢。不能只看源代码变短或寄存器理论数宣称收益。

- 首轮候选 B / 生产 A 各一次；异常数据不作性能样本。
- 若 c11 正常降时 ≥1.5%，再做反向 A/B，要求两对同向且未触达案无系统回退。
- 若在 ±0.5% 附近或变慢，关闭；没有 IR/编译诊断就不要换 10 种计数器拼写继续投。
- 新 SHA 首测 TLE 可预留**一次相同 SHA**复测；第二次仍 TLE 且无新错误线索则停。
- 正信号按榜分目标先扩 c6，再单独验 c4/c8；c12 可作为正常性能扩展。c11 的整数档距离小、实现单纯，适合作为机制 canary，不能因此无限消耗队列。

## 8. R2：MD 输出 padded ACT，令 MD→DN 布局连续

### 8.1 当前空缺与收益来源

当前 `_fgs_t1i_mdq_tma_pre_nf_kernel`（c3/c4）及 pre kernel（c11/c12）对完整专家块使用 TMA store，对尾块使用带 mask 的 pointer store。原因是 ACT 仍按 compact row 存放，尾块整块写会覆盖下个专家。DN 的输出已经 padded，但 **DN 的输入 ACT 没有 padded**。

方案是在 MD 输出端就使用已有 tile-padded row 编号；DN 随即从这一位置读取 ACT。目标是把每个 MD tile 的收尾改成一致的 TMA store，给短 K 的 pipeline 留出更简单的控制流。它不减少 GEMM FLOPs，也不消除 ACT 的整量物化。

### 8.2 地址合同

设专家 e 有 `count[e]` 行：

```text
R[e] = sum(count[x], x<e)                 # compact 行首
B[e] = sum(ceil(count[x]/128), x<e)       # padded tile 首
compact = R[e] + local_m*128 + lane
padded  = (B[e]+local_m)*128 + lane
```

现有 metadata 中 `B[e] = t_cum-t_num`；已经存在的 `INV_PAD` 供 final 读 DN。**无需再建 PAD_ROW、无需新一次扫描/拷贝、无需 CPU `.item()`。**

首发 c4，只做以下四处变化：

1. MD host 分配 `ACT[P,I]`，`P=128*ceil((M+127*E)/128)`，其中 M=T*k；GQ 的输入 q、AH/WI 和 ACT_SCALE 仍是 **compact**。c4 从 64 MiB ACT 增至最多 68 MiB。
2. MD 保留从 compact `a_row` 读取 A/AH/WI，保留所有运算顺序；输出改为 `ACT_DESC.store([padded_row,pid_n*128],q)`，去掉 full/tail 条件分支。整个 tile 被写入自己的 padded 块。
3. DN 的 `A_DESC.load` 改为 padded `a_row`；**A_SCALE 继续按 compact offs_m 读取**。B 权重、输出 padded row、CSCL 和 fin 的 INV_PAD 均保持原合同。
4. host 的 `T` 始终从原 X/output 得到，禁止从 padded ACT/Down 的第一维除 topk 推回 T。真实 M 与容量 P 分开传参。

**producer 与 consumer 必须成对切换。** 首版只改 c4 的 pre_nf producer；用 host 布局标记（例如 `act_is_padded`，默认 False，仅在新 producer 返回后设 True）选择 DN 的 ACT 读址。c4 的既有 `_GA` 分支和早期调用仍可能产出 compact ACT，必须继续配 compact DN 输入；不能只按 c4 shape 把所有 DN 输入改成 padded。这项标记描述本次 tensor 布局，不新增评测阶段判断，也不宣称 R2 已覆盖 c4 的所有调用路径。

尾行从未进入有效输出，但其 ACT 存储必须有定义，不能让有效 DN 行读取未写 scratch。优先明确写零 padding；若为避免新增 select 保留计算出的 padding 值，必须证明它们仅进入对应无效 M 行且永不被 fin 读取。空专家没有 tile，不得给它硬造一块。

安全容量上界来自 `sum(128*ceil(count/128)) <= M+127*E`，不是路由均匀假设。c3/c11/c12 对应额外 ACT 容量上限为 8/4/8 MiB；c9 则接近再加 63.5 MiB，本轮不推广 c9/c10。[CPU 映射验证](../reports/2026-09-30-sprint-direction-proofs.json)

### 8.3 两版实验，明确区分新前提

**R2a：只改布局，保持 c4 的非 flatten 循环及启动配置。** 这验证 ACT 合同和完整链成本，不预设一定更快。不要混入 R1 或改变 BNORM/量化。

**R2b：仅在 R2a 正确、且未明显变慢时，尝试该 branch-free c4 MD 的 flatten。** 这与历史“原含尾块分支的 MD 直接 flatten”不同。最多这一版兼容尝试；仍用 s4/w8/maxnreg232，保持 K 循环。不因一个 PassManager 错误开始扫全参数。

保留当前必要的编译兼容结构。v12 的 `tl.store(DROP+0,0)` 被所有 CTA 同址写，不是理想的 scratch 所有权；如果必须修兼容 scratch，应单列为变化，并改成每 CTA 独有槽位、每次完整写入。**不能未经 OJ 就断言移除 dummy 或改为独有槽位仍可编译。**

验收按 sort/GQ→MD→DN→fin 全链。目标 c4 正常两对同向 ≥2%；只看到 MD 理论上少分支不算通过。R2a 明显慢、R2b 无法编译或仍无收益就关闭，不扩 c1/c2。若有效，先 c3 或 c11/c12 分别验；不要一次把十案切换。

## 9. R3：消除 static DN 的二维 scale 中间表

### 9.1 可消除的依赖

static DN 已预先把 down channel scale 除以静态 `C[e,chunk]`。其量化输出 q 与动态 A_scale 无关；动态信息只用于写：

```text
DSCL[row,chunk] = max(FP32(A_SCALE[compact_row] * C[e,chunk]), 1e-12)
fin: sum_j FP32(q_j) * DSCL_j
```

因此 DSCL 可不物化，fin 用本次 A_SCALE 和静态 C 重建同一个 FP32 值。目标是让 DN epilogue 只做 B scale、FP8 conversion、TMA 输出，删除动态行尺度 load、乘法、floor 和二维 scalar store；fin 增加少量行级索引/算术。

这是**当前 static DN 已成立之后**才有的机会，不是重试“static DN 系数取 1.02”，也不是 BF16 output/atomic fin。c5/c7 使用动态 rowmax，不适用。c1/c2、c9 的动态 DN 也不在首版范围。

### 9.2 最小实现和数值顺序

首发只 c11，新增独立 DN/fin helper：

```text
branch = t*k+j
compact = INV[branch]
padded = INV_PAD[branch]
expert = flat_ids[branch]
a = A_SCALE[compact]
c = C[expert, h_chunk]
s = max(FP32(a*c), FP32(1e-12))
acc = 原 fin 的逐 j FP32 累加(q[padded,h] * s)
```

不要写成 `acc += (q*a)*c`；这改变 FP32 舍入、floor 和潜在 FMA。保持现有 fin 的 j 顺序与最终 BF16 舍入。C 是用 **DN 输出 H//256** 索引，不是 MD 的 I//128。

第一版直接复用已有 compact INV 和 flat_ids；它们无需新增 producer。fin 虽多两次依赖索引，但避免读取大 DSCL。下一版只有在明确被依赖加载吃掉收益时，才考虑由现有 GQ 给原始 branch 写一维 A_SCALE；不能无证据连续改造整个 producer。

host 显式标记本次 Down 是否由新 factorized-scale DN 产生，并据此配对新 fin。当前末尾仅用 `down.dtype` 和 `inv_pad` 分发；删掉 `_dscl` 后直接落入原 fin 会读空参数。未走新 DN 的调用仍用原 DSCL/fin，不要只按 c11 shape 全面替换。

c11 的 compact DSCL 仅 **2 MiB**，padded 容量约 2.0625 MiB。删表节省的纯字节很小，**收益假设主要是 DN 收尾简化**；新增 fin 查表完全可能抵消。不要宣称这是百 MiB 级流量优化。

CPU 在专家顺序物化原 DSCL，再从乱序 branch 的 INV/INV_PAD/expert 路径重建，比较 40,000 个 FP32 scale 位模式，覆盖空专家、padding、零值、极小值与 floor 激活。它验证索引和给定运算顺序下的尺度一致，不证明 GPU FMA、完整输出或 SQNR。

### 9.3 编译风险与停止条件

当前编译器曾在删除最后一条普通 `tl.store`、只剩 descriptor store 时失败。R3 可能遇到同类问题。若发生，允许一版每 CTA 独有 scratch store 的兼容修补；不得多 CTA 写同址，不加跨 CTA 自旋。兼容 scratch 及其开销必须算入候选。

要求完整 AC、同窗两对 c11 ≥1.5% 改善。若无可辨收益，关闭；有收益按积分账优先考虑 c6/c4/c8，c12 为正常性能扩展。扩到 c10 要保持其 tiled static DN 独立布局，不能直接复用非 tiled B descriptor。

本方案保留现有 `max(...,1e-12)` 行为，并**不修复**旧 static DN 极小 scale 与 q 不配对的代数边界。那是独立正确性问题，不能在此顺手改变输出量化后仍称纯搬移。

## 10. 本轮明确不重开哪些方向

| 方向 | 已知证据 | 本轮处理 |
|---|---|---|
| c9/c10 宽 GU / 交错单 accumulator | 152488 AC，无可辨收益 | 已完成，关闭同实现 |
| A 扩到 c1/c2 / c9 | v13/v15 正常结果没有支持晋升 | 不因首次高 raw 扩投 |
| EP2、single-peer、BM256 | 已修正确，完整链仍慢于 replicated | 仅“NV8 很快”不构成重启理由 |
| slot DN / pair regroup + fin | 最近 slot 方案两发慢；重排/尾块/权重复读代价明确 | 不能把省一次 fin 当净收益 |
| BM64 全替换或 ragged 双 launch | 历史权重复读、额外 launch 成本大 | 新 R2 保持原 tile 数与布局调度 |
| FP16 accumulator、大 tile、低位权重解包 | 已有明显慢结果和编译资源问题 | 没有新 lowering 证据不试 |
| warp specialization、cluster、Gluon 新接口 | 线上 3.4 系，生产 kernel 曾 TTGIR/CTA planner 失败 | 不把 main 分支教程当线上能力 |
| c1/c2 ACT TMA store、fin TMA store | 已有正常结果不优 | R2 只 c4，目标是消除既有尾分支 |
| metadata 合并、小参数全扫 | 完整两对中性或负；收益落噪声内 | 只保留明确依赖削减的 R1 |
| 动态 activation/路由结果跨调用缓存 | X 在正式运行会变 | 不作为提分资产 |

历史入口：[负结果索引](HISTORICAL_NO_REPEAT.md)、[9/27 审计](../reports/2026-09-27-history-audit.md)、[9/27 实验](../experiments/2026-09-27/SUMMARY.md)、[9/19 受控微实验](../experiments/2026-09-19/results/micro_pairs.md)。旧“所有路线物理不可能”的句子不是本指南结论；负结果必须绑定实际几何、工具链与实现。

## 11. 数值与调用合同：每位执行 agent 都要知道

这些是现役风险，**完整 AC 不代表普遍合同已被证明**：

- `_CALLN` 全局计数及 `_GA` 在 call 3–5 切换算法；题面没有保证这些序号对应哪个阶段。新候选不得新增这种识别。做同一 shape 的固定实现，初始化只影响静态缓存，不能影响算子含义。若移除现有门控，单独成候选，保留冷编译预算，不混到 R1–R3 的性能归因中。
- 多个静态派生缓存只按 shape，可能误用同 shape 新权重。对象强引用/身份可以覆盖对象替换，但不能解决同对象原地更新；四 rank 若失效决策不一致还会 collective 失配。主办方缓存边界答复仍重要。
- static DN 的极小 A_scale floor、router tied top-k 与数值近似余量须单独审查。c3/c10 最低 SQNR 仅 22.70/22.69，优先选择不改量化的结构方案。
- 输出 q/scale/SQNR 打印相同不是“新旧输出逐位相同”；平台 determinism 是同候选重复调用一致，和 baseline/candidate exact 比较是两件事。
- 所有地址在 swizzle 后配对；compact / padded / token / branch 四种索引必须在参数名和注释里明确，不能混用。新 tensor descriptor 的 leading stride 及基址必须满足对齐要求。[Triton descriptor 约束](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)

CPU 证明覆盖数学和索引，可以筛掉低级错误；正式 OJ 才覆盖当前 H800 编译、SQNR、determinism 和全链性能。如果 custom 未开放，不把另一个题目的接口当作通用 P1 测试入口。

## 12. 唯一平台执行者的协议与剩余时间预算

### 12.1 每发记录与判定

1. 重新核对生产 SHA、最近终态、是否已有其他 P1 提交作业；只留一个在途任务。用户自行分派 agent，本指南不授权各 agent 各开提交队列。
2. 候选写到独立路径，附 diff、触达 case、数值合同、成本假设和回退 SHA。通过 Python 语法及 JIT globals 静态检查。不要覆盖生产。
3. 初筛 B/A；有 ≥1.5%–2% 正信号再反向 A/B。raw 只记账，判效看逐案 tk 和未触达案，不用旧 mnorm 系数强行消噪。
4. 任意完整 AC 极低值保留为异常事件，随后同 SHA 正常复测；异常和 TLE 不充当性能样本。连续窗口中首次 TLE 与新 SHA 有关联，允许一次同 SHA 复测，不当作必然冷编译诊断。
5. 预先冻结确认次数。两对翻号或均值接近 ±0.5% 则停；不为 0.2% 的假收益追加几十发。
6. 只有正常性能和完整正确性都过门，才晋升；把最高分 SID 与可复现性能 SID 分别写入 README。多个成功改动合并后，重新测合并版，不能直接相加单项收益。

### 12.2 建议预算

| 工作 | 首轮预算 | 追加条件 |
|---|---:|---|
| R1 | 2 发筛选 + 2 发确认；最多另 1 发处理首 TLE | 明确正信号后才扩一案 |
| R2 | R2a 最多 2 发；R2b 最多 2 发筛选 + 2 发确认 | a 正确且非明显负；b 有信号才确认 |
| R3 | 2 发筛选 + 2 发确认 | 排在前两项之后 |
| 异常专用重投 | **默认 0 发** | 新平台诊断线索才最多 4 发 |
| 合并版与最终复测 | 预留 4 发 | 不借给无信息增益的扫描 |

这不是要无条件用完 20 发。明显负结果立即停止，把队列留给更有解释力的方案。当前已在罚分上限，提交的主要成本是时间、冷编译、排队和错误归因，仍不是无限免费的搜索。

建议时间窗：9/30 夜间完成 R1/R2 候选及索引审查；10/1 上午先筛 R1，再筛 R2；14:00 前决定是否启用 R3；18:00 后不引入新内核体系；21:00 前冻结合并源码，留下约三小时给正式队列与回退。不要在截止前一分钟依赖新 SHA 首次编译。

## 13. 可直接复制给其他 coding agent 的任务卡

### Agent 1：R1，独立读写索引

> 工作目录 `/Users/sakimi/Desktop/xpuoj-p1`。先读 AGENTS.md、README 与本指引第 2、5、7、11、12 节。生产锚 SHA 为 `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`。只对 c11 static padded DN 增加独立 write_tile 计数器，在 K 循环后重新构造 epilogue 索引，采用 Triton v3.4 官方 persistent matmul 的依赖分离方式。K 顺序、量化、BM/BN/BK、stages、warps、GM、其他案全部保持。新增 helper 尽量追加尾部，并记录每个 JIT 起始行变化。先复算 CPU proof；提交独立候选、diff、SHA、触达案、索引证明和成本说明。平台交唯一执行者做 B/A 后反向 A/B，两对同向且 c11 ≥1.5% 才优先扩 c6；c12 为可选的正常性能扩展，中性或慢即停。不得覆盖生产、引入评测阶段识别或自行并发提交。

### Agent 2：R2，padded ACT

> 从同一 v12 SHA 出发，只改 c4。按第 8 节把 pre_nf MD 的 ACT 输出改成 tile-padded row，用显式 host 布局标记让 DN 读取相同 padded row；未走新 producer 的 `_GA`/早期调用仍按 compact 读址。A_SCALE/AH/WI 仍 compact，Down/CSCL/INV_PAD/fin 保持现有合同。不新增映射 kernel 或 ACT copy。容量使用 M+127E 安全上界，不使用 `.item()`，不把容量除 topk 当 T。先交 R2a 布局版；正确且非明显慢才交 R2b branch-free short-K flatten 版。保持现有数值顺序和 launch 配置，至多两版。列出全 tile 唯一写者、空专家、尾块、padding 与有效行隔离的证明。全链正常两对 ≥2% 才继续，不复活 c1/c2 TMA store 或 c9 padding。独立候选交统一平台执行者。

### Agent 3：R3 备用与数值审查

> 阅读第 9、11 节。只 c11 static DN，删除二维 DSCL 物化，在 final 用 INV/INV_PAD/flat_ids、本次 compact A_SCALE、静态 C[e,h//256] 重建同一 FP32 s=max(a*C,1e-12)。用 host 标记把新 DN 与新 fin 成对分发；未走新 DN 的调用保留原 DSCL/fin。不重结合成 q*a*C，不改变 j 累加顺序和 floor，不把动态 DN 的 c5/c7 混进来。先算新增 fin 依赖与删掉的 DN store 成本；2 MiB DSCL 不能吹成大带宽收益。删普通 tl.store 若触发已知编译问题，只允许一版每 CTA 独有 scratch 的兼容修补。交独立候选与 CPU scale/index oracle；由统一执行者按 ≥1.5%、两对同向筛选。任务优先级低于 R1/R2。

### 平台与证据 agent

> 先读第 4、5、12 节。刷新当前榜单和最近提交，统一管理一个在途 P1 作业；复算257发连续队列与44个中性探针零命中。不要执行旧A/B任务或旧六发注释计划。按源码SHA、JIT文本、函数起始行记录候选，保留所有正常/异常/TLE/WA。遇完整AC极低值先保存原始日志，再同SHA复测；正常收益与异常分数分开，不把首次SHA直接标注冷JIT。控制候选/锚两对预算，冻结后只合并已证收益。保持最高分152238与正常性能锚152241均可追溯。

这些任务卡供用户派发，不表示已经启动任务或获得 GPU 结果。

## 14. 本轮一手资料与可复用边界

外部检索围绕 Hopper persistent GEMM、grouped/ragged layout、FP8 收尾、JIT 缓存与活动计时。比赛网页正文无法直接读取，以本地题面/讨论原文及授权只读 API 为平台依据。以下资料提供具体设计约束，不把别处跑分搬成 OJ 收益。

| 一手来源 | 本项目采用的内容 | 不直接搬用的部分 |
|---|---|---|
| [Triton v3.4.0 persistent matmul](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/tutorials/09-persistent-matmul.py) | 独立 epilogue 计数器、TMA/persistent 结构、收尾分块范式 | main 分支新特性和 Blackwell 专用调度 |
| [Triton v3.4.0 JIT](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/runtime/jit.py) | 函数源码、依赖与起始行参与缓存身份 | 不据此断言线上 fork/worker 磁盘缓存完全相同 |
| [Triton v3.4.0 swizzle2d](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/language/standard.py) | 尾 group 的列优先 swizzle 规则，用于 CPU 合同 | 不以 CPU 索引正确替代 GPU 编译正确 |
| [Triton v3.4.0 grouped GEMM](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/tutorials/08-grouped-gemm.py) | 固定 CTA 数、设备端分组调度、descriptor 基本用法 | 例程自身的 full-tile 假设不能套到任意 count |
| [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | TMA、寄存器/共享内存/L2 与并发约束 | 不由硬件支持推定当前编译器可用 cluster |
| [CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md) | mainloop、epilogue、寄存器、调度需共同设计 | 不在 P1 调用高层 GEMM；不直接移植 CUDA runtime |
| [CUTLASS Hopper mixed INT4/FP8](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu) | 低位存储仍需完整数据转换与执行管线 | 不是 H800 原生 FP4 能力，不能抹掉本项目解包负结果 |
| [TMA-Adaptive FP8 Grouped GEMM](https://arxiv.org/abs/2508.16584) | ragged 边界需要 descriptor/访存合同共同处理 | deadline 前不移植 descriptor pool，也不假定消 padding 免费 |
| [NVIDIA MoE grouped GEMM 说明](https://developer.nvidia.com/blog/accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine/) | 动态 expert 分组、避免 host count 同步的设计原则 | 本项目已经 grouped；通信去除不能重复计收益 |
| [CUPTI Kernel activity](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel9.html)、[Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html) | 零时间戳、丢记录、flush 边界是需要平台提供的诊断字段 | 未证实本赛计时实现；不能从论文猜出稳定异常触发器 |

## 15. 复算入口与需要用户协助的唯一高价值信息

本轮报告均在 `reports/`，旧报告未覆盖：

```bash
# 纯本地复算，无网络、无 GPU
python3 reports/analyze_strategy_20260930.py
python3 reports/prove_sprint_directions_20260930.py

# 可选：只读补全平台证据；不提交、不取验证码令牌
python3 reports/refresh_strategy_evidence_20260930.py

shasum -a 256 p1/kernel.py
```

CPU proof 完成 **120 个调度场景、273,840 个 tile 访问、3,384,912 次有效行映射检查、40,000 个 FP32 scale 位模式比较**。结果见[验证 JSON](../reports/2026-09-30-sprint-direction-proofs.json)；明确不包含 GPU lowering、TMA 顺序、最终 SQNR 或性能。

**当前不需要用户准备显卡、提供凭据或批准每发代码，其他工作可以直接按本文件推进。** 若用户能向主办方追问，最有价值的是下列两项，不再询问已经回答的硬件和主版本：

1. 同 SHA 的 **152238 与 152241** 为什么 c8–c12 的 tk 不同，能否提供逐 rank 有效计时记录数、实际运行次数、零时间戳/丢记录数及 flush/计时边界？这直接决定异常能否被解释和复现。
2. P1 跨测试点的静态权重是否更换 Tensor/存储，是否允许同对象原地改写，是否有公开缓存失效信号？这决定缓存与多 rank 重建的正确实现。

这两项是尚缺的平台信息；等待答复不阻塞 R1/R2 的独立结构验证。本轮没有代发站内消息。
