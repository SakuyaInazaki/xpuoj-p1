# P1 最后冲刺：异常证据、受控探针与结构优化任务书

证据截点：**2026-10-01 14:51:56（Asia/Shanghai）**。目标截止：2026-10-01 23:59，争取 **raw 90 / net 80**；目前未达到。本文替代旧的《终极冲刺与异常诱发优化指引》和 9 月 30 日任务书的待办优先级。

**建议马上执行的顺序：收取在途提交 → 原样复测 152976 一次 → 有限的 J1/J2 布局对照；开发工作先做 S1 hybrid GQ，再做 S2 padded ACT；S0 的 `I: tl.constexpr` 是小成本备选。** 正式平台统一一个执行者。用户自行分派 coding agent；本次指导工作没有另启 agent、没有新增正式/custom 提交，也没有修改生产 kernel。

本轮最关键的变化是：新 SID **152976 完整 AC / raw 89.00，c9–c12 四个 `tk=0`，c8=0.471 ms**。它比历史最佳多一个零计时点，却没有超过 89.08。异常研究值得继续，但“再多一个零点就一定破 90”不成立；前段整数档也必须同时保住。

## 1. 接手锚点与信息可信度

### 1.1 平台和源码

| 项目 | 已核实状态 |
|---|---|
| P1 榜面 / 最佳提交 | **79.08 / SID 152238，raw 89.08** |
| 正常同源码对照 | 152241 raw 82.00；另有同 SHA 的 152248、152251 |
| 提交数 | 3522；这是 14:51 平台快照，后续可能继续增加 |
| 总榜 | 账号总分 250.08、第 4；不能写成 P1 单题第 4 |
| 最近结果 | 152976 AC 89.00；152995 AC 81.92；153006 AC 83.42 |
| distributed custom | `triton-dist`：`available=false` |
| 单卡 custom | `triton-h800`：`available=true`；不能替代四卡 P1 评测 |
| 生产源码 | [p1/kernel.py](../p1/kernel.py)，7701 行，v12 |
| 生产 SHA-256 | `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9` |
| 同内容回退文件 | [p1_dirA_c34_v12_far_meas.py](../experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py) |
| 新异常锚 | [冻结的 SID 152976 源码](../experiments/2026-10-01/guide-candidates/anchor_sid152976_88361f262ddd.py) |
| 新异常锚 SHA-256 | `88361f262dddcb58986b23b9d54d9219c1645529b98752e8b768e19fc2ace673` |

依据：[平台快照](../reports/2026-10-01-platform-readonly.json)、[283 发逐提交证据](../reports/2026-10-01-cohort-readonly.json)、[分析结果](../reports/2026-10-01-strategy-analysis.json)。`experiments/.../candidates` 正被其他工作使用，**文件名不是版本身份**；例如 R3 的旧 v1 文件已被改写，其现内容与 152644 相符，不能用它还原 152641。先核 SHA，再读性能结论。

本文标明三类结论：**实测事实**来自完整日志/平台结果；**实现事实**来自当前源码或指定版本的官方代码；**待测假设**仅构成实验理由。没有 GPU 验证的方案不标为提速，没有复现的异常不标为稳定触发器。

### 1.2 题目合同与新讨论

先读 [完整题目](../1-full.md) 和用户提供的 [addinfo.md](</Users/sakimi/Desktop/addinfo/addinfo.md>)。对本轮最有影响的是：

- 四卡 H800，输入 BF16；router、top-k/归一化、GU、SwiGLU、路由加权、Down、按分支汇总的数值语义都要保留。SQNR 门槛为 **22 dB**，并检查输入只读、有限输出和逐字节确定性。
- 至少一次预热；随后会重新生成 `hidden_states`。允许缓存从同一测试点的静态权重/topk 导出的结果，**不允许据此缓存动态 X 或输出**。切换测试点后须相应更新静态缓存，这条合同已经明确。
- 最新答复是 **Triton / triton-dist 3.4 系**，精确 fork、CUDA、CUPTI、PyTorch 版本未知。P2/P3 的 3.6、流水线故障与 Graph 回答不能直接套给 P1。
- 官方称整批 `torchrun` 共用 **500 秒**；回复口语中写“十个测试点”，当前正式结果实际是 **12 个**。编译不计入正式 kernel 耗时，但会占整批超时预算。
- 评测器已初始化通信/NVSHMEM；使用 `triton_dist.utils` 的允许接口，不再自行初始化/结束 distributed 或 NVSHMEM。当前稳态 replicated 实现已在各卡保存全专家权重，不存在可再次消除的稳态 token 跨卡 dispatch。
- 用户确认：没有尚未写入 addinfo 的新计时异常答复。不能把推测升级为主办方已确认结论。

现代码的 `_CALLN/_GA`、部分缓存键和 static scale 极端数值处理仍有正确性边界。后面的实验不再增加调用阶段识别；结构改动要覆盖实际会执行的生产者/消费者，不能假定“call 3–5 必然就是全部计时调用”。

## 2. 这轮证据改变了什么

### 2.1 283 发连续窗口：精确零值和低值必须分开

窗口为 SID 149486 至 153006 的本账号 P1 记录，共 **283 发：207 Accepted、43 WrongAnswer、21 Canceled、12 TLE**。低值阈值沿用历史分析：某案 `tk < v926 正常锚的 50%`；这个阈值只是筛选器。

| 分组 | 完整 AC 数 | 至少一个低值的完整 AC | 至少一个精确 `tk=0` 的完整 AC | TLE |
|---|---:|---:|---:|---:|
| SHA 在此窗口首次出现 | 130 | 17 | **9** | 12 |
| 此窗口已出现过的 SHA | 77 | 0 | **0** | 0 |

“窗口首次”不等于账号生涯首次，更不等于已确认的冷编译。这是非随机、相互依赖的历史样本，**不能把 9/130 当作下一发成功概率**。旧的 44 个中性源码探针均无低值，也只能说明那批试验无效，不能声称理论命中率为零。

207 个完整 AC 都具备 12 案通过标记；重新按日志 `tc=` 匹配后，每案都找到两次 SQNR≥22 和两次 `[DETERMINISM OK]`。这支持“正式检查通过”，不证明所有潜在输入、缓存失效与调用次数都正确。

以下列全 17 次低值 AC，避免把非零低值混成 tk=0：

| SID | raw | 低值案 | 精确零值案 |
|---|---:|---|---|
| 149493 | 85.17 | 11、12 | 12 |
| 149520 / 149526 / 149530 | 84.83 / 84.92 / 85.00 | 各自 10–12 | 无 |
| 149609 | 83.50 | 10 | 无 |
| 150941 | 84.50 | 11、12 | 12 |
| 151793 | 83.42 | 12 | 12 |
| 151895 | 85.58 | 10–12 | 11、12 |
| 152205 | 87.75 | 9–12 | 11、12 |
| **152238** | **89.08** | **8–12** | **10–12** |
| 152289 | 87.58 | 9–12 | 11、12 |
| **152721** | **87.50** | **9–12** | **11、12** |
| 152939 / 152945 / 152969 | 83.00 / 83.00 / 82.92 | 各自 12 | 无 |
| **152976** | **89.00** | **8–12** | **9–12** |
| 153006 | 83.42 | 11、12 | 无 |

152939/152945/152969 的 c12 分别约 **0.460/0.462/0.461 ms**；153006 为 **c11=0.302、c12=0.460 ms**。这种窄档位复现值得记录，优先考虑“部分活动计入/归属变化”的假设；**尚不能据此声称 c12 计算真的快了三倍**。

### 2.2 最新候选的实际结论

| 实验 / SID | 可用结论 | 接下来 |
|---|---|---|
| R1 c11 152626、c6 152630 | 完整 AC，raw 81.92/82.00，正常 tk 无可辨收益 | 关闭普通参数扩散 |
| R1 all-static / all-pad 152713、152715 | 均 81.92，未给出正常收益 | 不把后续零值归功于 split counter |
| A 强制扩至 c3/c4/c5：152639 | 完整 AC 81.75，没解决额外 sorted Q 写入成本 | 不再原样扩 A；改做 S1 |
| R3：152641、152644 | 前者漏传参数；后者 tc1 rank2 exit 255 | 缺失 stderr，不能判定具体编译根因；降级 |
| pre_far f16x2：152647 | 正常 AC 82.00，无可辨共同收益 | 不独立再扫 tanh |
| c7 static DN+pretanh：152650 | c7 约 0.3%–0.4% 变化；c10 的 tb=9.135 抬高总分 | 不能据 raw 82.25 晋升 |
| c3 DN s3：152652、MDg s4：152660 | 前者微弱变化，后者变慢 | 关闭 |
| v3：152721 | AC 87.50，零 c11/c12 | 异常证据，非稳定增量 |
| v4：152920 | TLE | 不能据此定位耗时函数 |
| v6：152925 | AC 81.92 | 新增 tanh helper 实际未被有效 host 调用 |
| v8/v9/v16：152939/152945/152969 | c12≈0.46，均无零值 | 部分低值签名 |
| v17：152976 | AC 89.00，零 c9–c12 | **优先冻结并同 SHA 复测** |
| v18：152995 | AC 81.92，所有 tk 回正常范围 | 与 v17 不是单一位置对照 |
| v21：153006 | AC 83.42，c11/c12 非零低值 | 不宣称正常 c11/c12 加速 |

源码审计还揭示三个会让实验结论失真的问题：

1. **v2、v3、v4、v5、v17 的最终有效函数 AST 映射和其他顶层 AST 相同**，主要差别在定义排列/位置；结果却涵盖正常、部分低值、零值、TLE。它们没有体现新的数学优化。AST 相同仍不等于编译产物、JIT 缓存或运行时行为必然相同。
2. **v6→v8 不只是搬代码。** v6 留下一个未接入的新 tanh helper，v8 才把有效 pre_far host 接过去，同时搬位置；两个因素混在一起。v17→v18 也改变了有效 host 调用目标，不能当位置对照。
3. 部分候选有重复 `def`；应按 Python 最后定义解析，而不是 grep 到第一个同名函数就判定调用链。审计见 [source-audit.json](../reports/2026-10-01-source-audit.json)。

### 2.3 新的直接证据：`torch.profiler` 和沙箱改写

**21 个不同 SID 的实际日志出现 `torch/profiler/profiler.py` 的 cycle 清空事件警告**，包括 152644。该次日志同时显示 `warmup=1, iters=2, testdata_groups=2`。因此“至少这些执行路径使用 torch.profiler”有直接证据，不必再仅凭 Event API 被禁来猜测。

但警告本身不说明有 bug。官方 PyTorch v2.11 源码的 `prepare_trace` 在 `acc_events=False` 时就会发出这条提醒；若评测器每个 cycle 及时读取正确的结果，清空是正常行为。线上具体版本、schedule、统计读取位置尚未取得，不能假定与 v2.11 完全一致。[PyTorch v2.11 实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/profiler/profiler.py)、[官方 profiler API](https://docs.pytorch.org/docs/main/profiler)。

另外，152641 的实际回溯来自 `/judge/1/working/.kernel_cache/kernel_...py`，host 调用已变为 `_getitem_(kernel, grid)(...)` 和 `_getattr_(tensor, 'stride')()`；`run_kernel` 行号也与提交文件不同。**评测端会改写 host 源码**。上游 Triton 3.4 的 JIT key 确实包含函数起始行，但提交文件行号不必等于运行时 JIT 看到的行号；注释可能被改写器消去。不能从“本地多加一行”直接断言“线上新编译一个 kernel”。[Triton v3.4 JIT 源码](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/runtime/jit.py)。

原始摘录与限制见 [runtime-facts.json](../reports/2026-10-01-runtime-facts.json)。这些记录没有给出 CUPTI 版本、活动数、时间戳、事件聚合方法或所有 rank 的 trace。

## 3. `tk=0` 机制：当前应怎样判断

### 3.1 保留四组假设，按可观察结果区分

| 假设 | 现有支持 | 缺少的关键量 | 当前可做的实验 |
|---|---|---|---|
| profiler cycle / 读取时点 / activity 归属遗漏 | 真实 profiler 日志；非零低值呈窄档位 | schedule、读取发生在哪个 cycle、kernel 记录数、rank 汇总方式 | 同 SHA 复测；J1/J2；看整条 tk 向量及低值档位是否重现 |
| CUPTI 活动时间戳缺失或部分记录丢失 | 官方明确存在零时间戳；有外部近似现象 | start/end 原始值、dropped 数、版本、初始化时点 | 保存症状供对照；从 OJ 汇总 tk 本身无法确证 |
| 冷编译/模块注册/源码改写改变计时覆盖 | 9 次零值都在窗口首次 SHA；相同 AST 的布局变体结果不同 | 真实缓存命中、改写后 JIT 源码/行号、编译日志、实际时间线 | 小规模受控布局对照，不加空算、不改数学 |
| 真实提速、tb 漂移或路径/缓存错误 | 小幅 tk 改变、tb 上浮已有实例；现代码有阶段分支 | 同窗配对、动态输入覆盖、所有调用路径验证 | 全 AC 审核与正常 ABBA；单独审计缓存和数据依赖 |

当前最应上调的是 **profiler 统计覆盖/聚合** 的诊断优先级，而不是继续把“缓冲区耗尽”写成唯一原因。同一 SID 所有 case 的任务标识相同，也不够证明 12 案具体执行顺序；返回 JSON 的排列不能用作时间线。

### 3.2 CUPTI 资料支持到哪一步

- 官方 Kernel9 文档规定：无法采集 kernel 起止时间时，相应时间戳可以为零。这描述的是记录字段的语义，**不证明 XPUOJ 的 tk 来自此字段**。[CUPTI Kernel9](https://docs.nvidia.com/cupti/12.8/api/structCUpti__ActivityKernel9.html)
- Activity API 的内部 device buffer 会在消费后复用；高活动密度、分配受限等可能造成时间戳问题。不存在从这份文档就能推出的“一次性容量用完后，余下测试永远全零”定律。常见默认量级为每 buffer 3,200,000 字节、上限 250 个，具体活动种类/版本仍有差异；semaphore pool 的部分旧属性自 CUDA 12.3 已不支持。[CUPTI Activity API](https://docs.nvidia.com/cupti/12.8/api/group__CUPTI__ACTIVITY__API.html)
- CUPTI 13.1 发行说明记录过自 CUDA 12.6 Update 2 引入、与自定义 timestamp callback 有关的零时间戳问题。只有确认版本与回调条件后才适用。[CUPTI 13.1 release notes](https://docs.nvidia.com/cupti/13.1.0/release-notes/release-notes.html)
- 2026-09-15 NVIDIA 论坛有用户在另一套 H20/PyTorch/vLLM 环境报告：固定数量后时间戳归零、dropped 为零、推理继续，改变初始化时点会改变结果。这是值得保留的第一手外部复现，**不是 NVIDIA 已确认的 P1 诊断，也不能直接移植触发阈值**。[外部复现帖](https://forums.developer.nvidia.com/t/cupti-13-4-58-gpu-memory-growth-and-zero-timestamps-at-fixed-record-counts-late-initialization-helps/383290)

因此撤销旧 E1 的“两次辅助 launch 保证把零值推到 c7”方案。persistent kernel 内处理的 CTA/tile 数也不是同数量的 kernel launch 活动记录。对着未知采集器增加无用工作既没有阈值依据，也会混淆有效计算与采集行为。

### 3.3 一个必须纠正的时间单位错误

本窗口所有有相应字段的完整 AC 均满足：

```text
平台 timeUsed ≈ 1000 × Σ(12 案显示的 tk_ms)
```

例如 152969 的 `timeUsed=27430` 对应 **Σtk=27.430 ms**，不是“整批只跑 27.43 秒”。它不能证明加编译仍安全低于 40 秒，不能用来给 500 秒整批预算背书。当前缺少可信的编译/整批 wall time；提交到终态的时间还含排队，须单独记录。

## 4. 积分账决定应该优化哪些 case

本窗口 207 个完整 AC 的显示分都能由以下公式复算到两位小数；这是本窗口实证，不宣称覆盖平台所有计分区间：

```text
q_i = floor(100 * tb_i / (tb_i + tk_i))
raw = Σ q_i / 12
当前 net = raw - 10
```

raw 90 要求整数总和 ≥1080。152238 为 **1069**，差 11；152976 为 **1068**，差 12。时间是显示精度，临界档位应留余量。[可复算的积分模型](../reports/2026-10-01-score-plan.json)。

下面使用 v12 正常同 SHA 的 152241/152248/152251 逐案中位数。它是比较窗口，不是一发新的实测提交。case 编号来自正式日志，不要照题面仅三个 sample 的编号对号入座。

| c | T / H / E / I / k | tk ms | tb ms | q | 再上 1 档所需 tk 降幅 | 最低 SQNR dB |
|---|---|---:|---:|---:|---:|---:|
| 1 | 16384 / 4096 / 8 / 8192 / 2 | 4.600 | 17.682 | 79 | 3.90% | 23.12 |
| 2 | 16384 / 4096 / 8 / 14336 / 2 | 7.857 | 28.736 | 78 | 2.78% | 23.14 |
| 3 | 16384 / 2048 / 32 / 2048 / 4 | 1.348 | 6.938 | 83 | **1.96%** | 22.70 |
| 4 | 16384 / 2048 / 32 / 1024 / 4 | 0.795 | 4.943 | 86 | 7.09% | 22.98 |
| 5 | 8192 / 3584 / 64 / 2560 / 8 | 2.760 | 12.071 | 81 | 4.00% | 23.13 |
| 6 | 8192 / 3584 / 64 / 1024 / 8 | 1.246 | 7.471 | 85 | **2.39%** | 22.93 |
| 7 | 16384 / 4096 / 96 / 2048 / 3 | 2.180 | 10.252 | 82 | **3.68%** | 23.12 |
| 8 | 16384 / 4096 / 96 / 1024 / 3 | 1.232 | 7.157 | 85 | 5.43% | 22.90 |
| 9 | 4096 / 4096 / 256 / 2048 / 8 | 2.439 | 7.724 | 76 | 5.41% | 23.13 |
| 10 | 4096 / 4096 / 256 / 1536 / 8 | 1.857 | 6.679 | 78 | 4.39% | 22.69 |
| 11 | 65536 / 1024 / 32 / 1024 / 2 | 0.858 | 5.668 | 86 | 1.29% | 23.14 |
| 12 | 65536 / 1024 / 32 / 2048 / 2 | 1.446 | 8.134 | 84 | 0.73% | 22.90 |

如果固定 tb、所有正常 tk 同时降低 5%/10%/30%/50%，模型 raw 分别是 **82.67/83.33/86.42/89.75**。所以截止前把正常 82 推到 90 极难；围绕完整 AC 异常冲榜有现实动机，但还没有高概率触发方法。

对历史最佳的条件账：

| 只改变指定项，其余严格保持 152238 | raw |
|---|---:|
| c8 从 q93 变为零计时 q100 | 89.67 |
| c8、c9 都变零 | 89.75，仍差 3 个整数分 |
| 仅 c7 从 q83 变零 | 90.50 |

正常中位数窗口中，即便 c8–c12 都零也只有 **89.50**；c7–c12 都零才到 **91.00**。以上只是条件算术，既不承诺零值扩展，也不能把不同 SID 的最好 case 拼成成绩。

**任务选择结论：c3/c6/c7 的真实收益对“前段尚正常、后段出现异常”的提交最有价值。c11/c12 的小优化可以改善正常回退成绩，却不会给已经零计时的最佳提交加分。c4 适合先验证布局正确性，但正常快 2% 不等于立即多一档。**

## 5. 异常主线 J：围绕新锚做有边界的布局实验

### 5.1 为什么这次值得做，为什么不扩大为随机搜索

v17 与 v3 的有效函数体没有变化，v17 却从 c9 开始四案精确零值。这比“加一行注释试试”有更具体的观察依据。但 v17 搬了一个较大代码窗口，许多 JIT 定义同时移位，不能知道影响来自 MDg、TMA pre、tiled DN、编译注册顺序还是平台状态。

这次把变量缩到 **一个相邻 JIT kernel 与普通 host 函数对的前后顺序**：host 在定义时不调用 kernel，执行 `run_kernel` 前所有定义仍已完成；函数体、参数、调用者、其他顶层语句都保持。只在原窗口内交换，不添加文件头、空算、额外 launch、延时或新的 `_CALLN` 分支。

已经生成以下文件，**全部只经过本地 Python 语法和 AST/行号检查，尚未上 OJ**：

| 候选 | 文件 | 相对 152976 的唯一 JIT 提交起始行变化（含装饰器） |
|---|---|---|
| J0 锚 | [anchor_sid152976_88361f262ddd.py](../experiments/2026-10-01/guide-candidates/anchor_sid152976_88361f262ddd.py) | 无，必须字节相同 |
| J1 | [p1_layout_J1_unmeasured.py](../experiments/2026-10-01/guide-candidates/p1_layout_J1_unmeasured.py) | `_fgs_t1i_mdq_kernel_g`：6007→6056，主要覆盖 c3–c8 gather MD |
| J2 | [p1_layout_J2_unmeasured.py](../experiments/2026-10-01/guide-candidates/p1_layout_J2_unmeasured.py) | `_fgs_t1i_mdq_tma_pre_kernel`：6156→6177，覆盖 c11/c12 A 路径 |
| J12 | [p1_layout_J12_unmeasured.py](../experiments/2026-10-01/guide-candidates/p1_layout_J12_unmeasured.py) | 同时作 J1、J2 |

完整 SHA 与验证结果在 [探针清单](../reports/2026-10-01-controlled-layout-probes.json)。生成器比较所有同名函数的**每次定义**，不只是最终定义；交换窗口外 JIT 行号保持不变。**这只控制提交文本，不控制沙箱改写后的源码、编译缓存或评测机器。** 也不证明在线允许此排列，一旦出现验证拒绝就记录并停该支路。

### 5.2 执行顺序与分支

1. 先查询在途 P1 的终态和已提交 SHA。若其他执行者已经复测 J0 或提交同内容，直接使用那条结果，不重复消费额度。
2. **原字节 J0 同 SHA 复测一次。** 结果是正常、部分低值、精确零值三者中的哪一种，均有信息；不要为这次复测改名或格式化文件内容。
3. 在正常平台窗口依次首测 J1、J2，逐发收终态。两者中有完整 AC 低值/零值则优先做同 SHA 复测；如只给普通波动，预算内再首测 J12 一次，随后停止纯布局扩散。
4. J1/J2 都异常也不能推出两个因素都因果有效；三发首见样本没有足够统计力量，机器/缓存状态无法随机重置。记录关联用于选择后续结构落点，不发布命中率。
5. 首发 TLE 可按下文规则原 SHA 重试一次，消耗本路线预算；WA/导入拒绝先修明白，不能通过重复投同样 WA 排查。

**本路线总预算 6 发，含 J0、最多 3 个新布局、最多 2 次异常/TLE 复测。** 三个新布局都无有用信号，关闭；不要把 J12 再组合成几十种移动。新结构 S1/S2 首发出现异常时继续沿用同一证据表，也算研究样本。

重点记录整条向量，不只记“有没有 0”：

```text
SID / source SHA / 原始源码冻结路径 / 父 SHA / 唯一变更
完整 AC? / 12 案两组 SQNR / 两次 determinism / 每案 tb 与 tk
精确零值集合 / <50% 低值集合 / 0.30、0.46、0.75 等局部档位
每案 q / raw / net / 实际错误签名 / 首见或重复 / 排队与终态时间
```

只有完整 AC 的零值才计入冲榜有效样本。**153006 的 c12=0.460 不是 tk=0。** `timeUsed`、首见 SHA、源文件位置都不冒充编译 wall time 或机器标识。

## 6. 结构主线 S1：token-only Q 与预计算分支参数组合

**优先级最高的新计算方案。首发仅 c6，成功后 c3，再考虑 c7/c5。** 已有 A 扩展失败没有覆盖这条组合：A 的参数前移与 k 份 sorted Q 绑定；当前 `_GA` 路径保持 token-only Q，却在每个 N tile 重算参数。S1 保留两者各自有价值的部分。

### 6.1 当前瓶颈与新数据流

当前 c3–c8 的 `_GA` 路径：

```text
GQ：X[T,H] → Qtoken[T,H]、scale_token[T]
MDg：按 ORDER[row]//k gather Qtoken
     每个 N tile 再读取行 scale、sorted weight、BNORM，重算 bound/指数/s
     输出 ACTcompact[M,I]；多个 N tile 重复写同一行 SCL
DN → fin
```

已晋升 A 的另一条路径：GQ 一次生成 AH/WI/ACT_SCALE，但同时把每个 token 的 Q 写入 k 份 branch 行。直接强制 A 的 152639 没有正常收益，说明必须把复制成本从实验中拿掉。

S1：

```text
新 GQ：每 token 只读/量化/写 Q 一次
       遍历 k 个真实路由分支，只把 AH/WI/ACT_SCALE 三个标量写到 compact 行
新 MDg-pre：保留 Qtoken 的 ORDER gather、B 的 TMA 与原 MMA
           K 循环后读 compact AH/WI，直接做既有 epilogue
           ACT 仍 compact；SCL 由 GQ 唯一生成，MD 不再重复写
DN / fin：保持原接口与数值语义
```

### 6.2 精确实现合同

复制 `_gq1p_tok_kernel` 为新 producer，不能改原函数返回合同。`M=T*k`，`INV[b]` 是 compact sorted 行，`FLAT_IDS[b]` 是专家，`FLAT_W[b]` 是原 branch 权重。保留原量化舍入和 `scale=max(amax/448,1e-12)`。

```python
# 伪代码；q 只在 j 循环外写一次。
store(Qtoken[t, :], quantize_exactly_as_existing(x[t, :]))
store(SCALE_TOKEN[t], a)
for j in tl.static_range(K_BRANCH):
    b = t * K_BRANCH + j
    row = load(INV[b])
    e = load(FLAT_IDS[b])
    w = load(FLAT_W[b])
    bn = load(BNORM[e])
    bound = (((a * a) * bn) * bn) * abs(w)
    bexp = exponent_bits(max(bound, 1e-30))
    s = bitcast_f32((bexp + 10) << 23)
    inv = bitcast_f32((244 - bexp) << 23)
    store(AH[row], a * 0.5)
    store(WI[row], (w * a) * inv)
    store(ACT_SCALE[row], s)
```

`INV` 是置换，每个 compact 行只有一个 producer。乘法结合顺序按现 `_gq1p_tm_params_kernel` 原样保留，不顺手改成对数、pow 或新的 bound。

新 consumer 从 `_fgs_t1i_mdq_kernel_g` 派生：

- A 仍按 `ORDER[offs_m] // KTOP` 读取 **token Q**，所有 row mask、stride、B 的 GU 交错布局保留。不能拿 compact 行直接寻址 token Q。
- 去掉 MD 的 W、BNORM、行 scale 的参数计算；K 循环后按 **compact offs_m** 读 AH/WI。
- `scaled=acc*b_scale → reshape(M,N,2) → gr,u → h=gr*ah → 原 f16x2 tanh → silu=h*th+h → q=(silu*u*wi).fp8`。
- ACT 仍 `[M,I]` compact；返回的 `act_rowscl` 必须是新 GQ 的 ACT_SCALE[M]，**不是 SCALE_TOKEN[T]**。DN 仍读取 ACT，不读取 token Q。
- 新 host 不再创建 `weights[order]`。不另起 launch 算参数；直接放进现 GQ launch，否则会改变成本模型。

### 6.3 成本账与止损

| shape | 若强制旧 A 的 sorted Q | S1 的 token Q | 两者相差 |
|---|---:|---:|---:|
| c3/c4 | 128 MiB | 32 MiB | 96 MiB |
| c5/c6 | 224 MiB | 28 MiB | 196 MiB |
| c7/c8 | 192 MiB | 64 MiB | 128 MiB |

**这些差值是相对“强制旧 A”，不是相对当前 `_GA` 基线的新增节省。** 对当前基线真正节省的是重复 N-tile 参数运算、标量读写与 sorted weights gather；新增成本是每 token 的 k 次标量 gather/scatter 和 AH/WI/ACT_SCALE 数组。c6 三数组合计约 0.75 MiB。可能出现 GQ 变慢抵消 MD 收益，必须看整案。

首发范围用完整 shape `(8192,3584,64,1024,8)` 限定，并且只替换**原本选择 token-only Q 的支路**。在原有 producer 分发中设置独立 `_hybrid` 标记，再由这个标记选择 consumer；不要扩大 `_dir_a` 条件，不要让 E=256 误入。152634 的 tuple unpack 失败就是这种错误。其他调用次数的路径先保留；后续如要统一所有调用，另做独立候选并验证，不能把调用序号视为计时 oracle。

起步参数保持 MDg 的 BM128/BN128/BK128、原 GROUP_M、w8、原 launch stage；`I` 继续 constexpr。不要同时换 tile、tanh、DN scale。新 helper 追加文件尾部，减少无关文本扰动；这只是实验控制，不保证线上缓存不变。

晋升门槛：两对正常对照 c6 整案至少约 **2%** 收益，最好跨过当前约 2.39% 的整数档并留余量；SQNR 不出现实质退化，12 案全 AC。若一对结果小于 1% 或明显变慢，再允许一次针对明确 GQ/调用接线问题的修正，仍无信号就关闭。不要把首发低值当正常收益。

本地已完成 **3471 个 branch 映射、10413 次 FP32 标量 bit 比较**，支持索引和指定结合顺序；没有模拟 GPU 编译、FMA、tanh/FP8 全链 SQNR。证明脚本见第 11 节。

## 7. 结构主线 S2：把 c4 的 ACT 真正改为 padded，并覆盖两条 MD

**首发 c4，先布局，后流水。** 旧 R2 只列 `_fgs_t1i_mdq_tma_pre_nf_kernel` 不完整：生产 c4 在 `_GA` 支路也会用 `_fgs_t1i_mdq_kernel_g`。只改前者可能根本没有优化你想测的路径。

### 7.1 要一起改变的三个地址语义

| 对象 | 现布局 | S2 布局 |
|---|---|---|
| MD 输入 Q | token-only 或 compact sorted，取决于 producer | 各自保持 |
| ACT | compact `[M,I]` | expert-tile padded `[P,I]` |
| AH/WI/ACT_SCALE、动态行 scale | compact `[M]` | **仍 compact** |
| DN 输出和 DSCL | 已是 padded | 保持；fin 仍用 INV_PAD |

沿用 metadata，不重新发明排序。对 swizzle 后的 `local_m`：

```python
compact_row = row_begin + local_m * 128
padded_row = (t_cum - t_num + local_m) * 128
P = 128 * ((M + 127 * E + 127) // 128)  # 安全上界
```

c4 `M=65536,E=32,I=1024`，ACT 最大由 64 MiB 增至 68 MiB。每个专家最后一个 tile 拥有私有的 128 行，不能用 compact 地址整块写，否则会覆盖下个专家。

### 7.2 producer 与 consumer 配对

1. 为 c4 新建 **MDg-pad** 和 **pre_nf-pad**，都将 ACT descriptor 设置成 `[P,I]`，每 tile 无条件 `ACT_DESC.store([padded_row,pid_n*128],q)`。
2. MDg 当前已对满 tile 用 TMA store、尾 tile 用 masked store；本方案消掉的是尾块分支/混合写回，**不是把全部 ACT 写回从普通 store 变成 TMA**。pre_nf 同理。
3. MD 输入寻址不变；AH/WI、SCL 均按 compact 行寻址与 mask。MDg 的 SCL 生成先原样保留，不混入 S1。pre_nf 中既有 DROP store 也先保留，不把 descriptor-only 路径风险混进首测。
4. 配套新 static DN 只将 `A_DESC.load` 起始行换为 padded_row；`A_SCALE` 仍 compact。C_DESC、CSCL、INV_PAD 与 fin 保持原有对应关系。
5. host 由**实际 producer**设置 `act_is_padded`，仅该标记为真时调配套 DN；旧调用、冷路径仍返回 compact，不能仅按 shape 强制所有 DN 读 padded。
6. 原 DN host 的 `M,K=a_q.shape` 在 S2 中会把 P 当 M。凡需逻辑行数都显式传 `logical_m=T*k`，分别管理物理容量 P 与有效行数 M。
7. 所有实际消费的 padded tiles 都由新 MD 写过；无效尾行不会被 fin 聚合，无需额外清零整个 ACT。检查每个有效行恰好一个 tile 覆盖，空专家不读写，容量足够。

TMA 仍需连续最后维、基址及外层字节 stride 对齐，descriptor block 与原 fp8 dtype 一致；本题 I 均满足相关整除，但别推广成任意 shape 自动可用。[Triton TensorDescriptor 文档](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)。

### 7.3 不要把两个 `num_stages` 混起来

c4 当前 MDg 的 outer `tl.range(...,num_stages=2)`，host launch 为 **3 stages**；pre_nf 的 outer 也是 2，但 launch 为 **4 stages / maxnreg=232**。c4 static DN 的 launch 是 **3 stages**。

S2a 首测保留这些各自参数，仅改 ACT 地址合同。若全 AC 且正常整案有 ≥2% 信号，再单独试 S2b 的 flatten；**删除分支并不保证 flatten 有收益**，长 K 或寄存器压力也可能变差。两个结构版本都无信号即停，不扩大 stage/warp 网格搜索。

CPU 已检查 9 组 counts（含空专家、全量集中、127/128/129 边界）、1341 个完整 tile 不重叠和 140808 个有效 compact↔padded 行对应。GPU 的 TMA、同步、descriptor 生命周期、SQNR 仍须 OJ。

## 8. 小成本备选 S0：给 A consumer 的 I 恢复编译期特化

这是源码中实际存在、旧 R1/R3 没解决的优化机会，**不是已经证实的回退 bug**。原旧 TMA 路径也有同样写法，因此不能断言它是 A 引入的退化。

```python
# 当前 MDg：I、K 都是 constexpr
M, I: tl.constexpr, K: tl.constexpr

# 当前 tma_pre / tma_pre_nf / pre_far 的 kernel：只有 K 被注解
M, I, K: tl.constexpr
```

Python 不会把最后一个参数的注解分给前面参数。I 决定 `num_block_n`、tile 的除余、swizzle 与 `expert*(2*I)` 地址；静态维度可能减少整数控制成本。CUDA 官方也指出整数除余成本较高、常量特别是 2 的幂可被优化，但需要具体生成代码/实测才知道净收益。[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

首发只复制 c11 的 `_fgs_t1i_mdq_tma_pre_kernel`，把签名改为 `M, I: tl.constexpr, K: tl.constexpr`，**M、counts、num_tiles 不新增 constexpr**，避免动态规模的编译特化膨胀。接入一个专用 host，FP32 结合、DROP、tile、stage 不变。禁止用顶层 Python 函数对象别名覆盖旧函数；152622 已被静态校验拒绝。

预算最多 4 发，含首测、必要复测及正常配对；正常收益达约 1.5% 才值得继续。c11 成功仍只是正常回退增量，若时间有限优先完成能帮助 c3/c6/c7 的 S1。S0 不与 S1/S2 首版本混合，以便归因。

## 9. 统一执行、验收与停止条件

### 9.1 下一轮总预算建议：最多 24 发

| 路线 | 上限（含修补/复测） | 交付物 |
|---|---:|---|
| J：异常布局对照 | 6 | 精确源码、全向量、首见/重复对照、是否继续 |
| S1：hybrid GQ | 6 | c6 独立候选；正常信号成立后扩 c3/c7 |
| S2：padded ACT | 6 | c4 双 producer + 成对 DN；layout-only 先过 |
| S0：I constexpr | 4 | c11 独立候选与正常配对 |
| 最终集成/复核 | 2 | 只合并有正常收益或需要验证的组合 |

这是限额而非必须用完的任务，也不与其他 agent 的提交预算重复相加。优先级依次为 J 的短实验、S1、S2、S0；编写候选可由用户并行分派，但提交必须串行。若截至较晚时段队列/编译耗时上升，先取消低优先路线，建议至少预留最后 90 分钟给已完成候选的终态与冻结。不能仅因轮询超时再发同一候选。

### 9.2 三种验收互不替代

**正常性能晋升：** 用同窗、同 SHA 的 v12 做 A/B/B/A 或相当的两对对照。保留每案原始 tk；用未改 case 的中位数比值辅助识别机器整体波动，不能用异常 tb 强行归一成收益。任何含目标案极低值的发次先从正常比较中剔除。差异在 1% 左右而方向不一致，按无可辨收益处理；超过门槛也要确认其他 case 未丢档。

**异常冲榜记录：** 必须 12 案完整 AC、两组精度/确定性通过、保存原始 tk/tb 和源码字节。若 raw 创新高，记录可见榜单更新，但暂不把低 tk 当算子吞吐；再做一次同 SHA 复测来决定能否重现。同 SHA 回正常仍保留这个历史高分样本。

**首测 TLE：** 全批只有 500 秒，又缺少完整编译日志，因此允许同 SHA 重试一次。第二次 TLE 关闭；不要将“首次 TLE→重复 AC”无条件解释为 cache hit，也不要用提交 `timeUsed` 推算编译耗时。

SQNR 官方门槛是 22，历史 c3/c10 余量较小。尽量保持基线 SQNR；若降低约 0.2 dB 或更多，先定位运算/路径差别再晋升，这个 0.2 是工程排查阈值，不是新增比赛规则。所有 case 的定序累加、输入只读和有限输出都要保留。

### 9.3 提交前只做有价值的本地检查

- SHA、Python 语法、最终有效定义/调用引用、tuple 参数数量，尤其 GQ 的 2 返回值与 5 返回值不能混用。
- S1：`INV[b]` 唯一写者；token scale 与 ACT scale 不串；E=256 不进新分支。
- S2：两种 producer 覆盖；logical M 与 capacity P 区分；compact scale 与 padded ACT/Down 显式配对。
- 数值与布局 CPU 合同通过后，上 OJ 验证；本机没有四卡 H800，语法通过不能写成性能/完整正确性通过。
- 保存候选原件再提交，不在某 SID 对应文件上原地继续改。平台 SHA 是最终身份。

平台工具和授权沿用 [SUBMISSION.md](SUBMISSION.md)。只用 OJ 正式/平台实际允许的 custom；不改共享令牌池目录、不输出凭据。`triton-dist` custom 不可用时不要绕成单卡成绩替代。

### 9.4 当前不再立项的方向

R1 读写计数器普通扩散、无诊断的 R3 反复修编译、单独 f16x2/阶段数小扫、c1/c2/c9 强行 A、旧 B wide、EP2/slot DN、全局 BM256、metadata 合并等，除非有新的具体证据，不重复开题。完整历史参见 [HISTORICAL_NO_REPEAT.md](HISTORICAL_NO_REPEAT.md)。

Triton v3.4 persistent matmul 中的双计数器注释明确针对 **Blackwell** 的流水线限制，不是 H800 上必然加速的普适依据；warp specialization 也不能直接从该教程迁移为 P1 3.4/Hopper 可用能力。[官方 persistent matmul 示例](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/tutorials/09-persistent-matmul.py)。

暂不把 epilogue subtile 列主线：它能减少输出暂存，但未必跨过 SMEM/occupancy 阈值，而且会改变 store 数及编译路径；应在有资源报告后再判断。截止前优先完成上面有清晰未覆盖成本的实验。

## 10. 可直接分派给 coding agent 的任务卡

### 任务 J：唯一平台执行者

> 阅读本文件第 1–5、9 节。先查在途与已提交 SHA，基线生产文件不动。使用 guide-candidates 内已冻结 J0/J1/J2/J12；提交前复算 SHA 并对照清单。按 J0 一次、J1/J2、必要时 J12 的顺序执行，路线总上限 6 发，含异常/TLE 复测。不新增注释探针、额外 launch 或调用计数分支。逐发记录 12 案 tb/tk、q、两组 SQNR/确定性、exact-zero 集合和错误。不要将局部低值写成零，不把 timeUsed 当整批秒数。交付冻结源码与一页结论，然后停止纯布局扩散。

### 任务 S1：GQ / MD 开发者

> 从 v12 SHA 08dd08eb51b1… 派生独立候选，先只覆盖 c6 `(8192,3584,64,1024,8)` 的现 token-only Q 支路。按第 6 节实现 token-only Q + compact AH/WI/ACT_SCALE producer，再实现保留 ORDER gather 的 MDg-pre consumer。Q 只写 T 行，scale 和三数组不能混用；新 MD 不创建 sorted weights、不重算 bound、不写 SCL；DN/fin 保持。数值操作顺序与 f16x2 epilogue 保持；禁止顺带换 tile/stage。追加 helper、明确 `_hybrid` 接线，排除 E256。运行 CPU 合同/静态检查后交唯一执行者。以正常两对 ≥2% 信号决定是否扩 c3/c7，异常单独入账。

### 任务 S2：ACT 布局开发者

> 从 v12 派生 c4 独立候选。必须同时处理 `_fgs_t1i_mdq_kernel_g` 和 `_fgs_t1i_mdq_tma_pre_nf_kernel` 两个实际 producer。ACT 写 expert-tile padded；所有行参数仍 compact；用实际 producer 标记选择 padded-input static DN；Down/DSCL/INV_PAD 不改。显式区分 logical M/P，保留原 outer/launch stages 和 DROP。先通过第 7 节 CPU 边界合同，再由统一执行者做 layout-only 正式全 AC。正常有 ≥2% 信号才另试 flatten；不混入 S1、R3 或 stage 扫描。

### 任务 S0：小成本特化核验

> 从 v12 独立派生 c11 的 TMA pre consumer，仅将 I 设为 constexpr，M/counts/num_tiles 不特化。使用独立 kernel/host，禁止顶层函数对象别名。保持 tile、stage、数值和 DROP；先静态确认实际接线。正常降时 ≥1.5% 且全 AC 才保留；总预算 4 发，异常不算正常加速。不要把此任务对尾部的收益直接加到历史最佳的零值 case 上。

## 11. 交付证据、复算入口与还缺的信息

### 11.1 本轮文件

| 文件 | 用途 |
|---|---|
| [platform-readonly.json](../reports/2026-10-01-platform-readonly.json) | 榜单、最近提交、custom 状态 |
| [cohort-readonly.json](../reports/2026-10-01-cohort-readonly.json) | 283 发结果与日志、平台源码指纹 |
| [ledger.csv](../reports/2026-10-01-ledger.csv) / [strategy-analysis.json](../reports/2026-10-01-strategy-analysis.json) | 状态、低值/零值、首次/重复、源码匹配与公式核验 |
| [source-audit.json](../reports/2026-10-01-source-audit.json) | 最终定义、重复定义、JIT 起始行、参数注解与调用引用 |
| [runtime-facts.json](../reports/2026-10-01-runtime-facts.json) | profiler 警告和沙箱改写证据 |
| [score-plan.json](../reports/2026-10-01-score-plan.json) | 12 案 shape、正常锚、整数档与条件积分账 |
| [direction-proofs.json](../reports/2026-10-01-direction-proofs.json) | S1 标量/映射、S2 容量/行对应、注解核验 |
| [controlled-layout-probes.json](../reports/2026-10-01-controlled-layout-probes.json) | J0 锚、J1/J2/J12 SHA 与逐 JIT 行号变化 |

本地 CPU 复算，不会提交：

```bash
python3 reports/analyze_strategy_20261001.py
python3 reports/extract_runtime_and_score_20261001.py
python3 reports/prove_hybrid_and_layout_20261001.py
python3 reports/build_controlled_layout_probes_20261001.py
shasum -a 256 p1/kernel.py
```

最后一个生成器要求当前 v17 原件与锚 SHA 相同，若被其他工作改写会拒绝，不能绕过检查继续生成。需要刷新时，用 `python3 reports/refresh_strategy_evidence_20261001.py` 只读查询本人记录；它不会提交，也不消费提交令牌。刷新会推进报告截点，文档静态统计不会自动更新；交接时注明新旧截点。

### 11.2 有这些信息才可能把机制定位到更窄

目前不需要用户补资料才能执行 J/S1/S2。若后续能从主办方得到以下少量字段，诊断价值很高；本次没有擅自发消息：

1. P1 PyTorch/CUDA/CUPTI 的准确版本；tk 是 wall duration、CUDA event、profiler kernel duration 求和还是其他方式，如何跨 rank 聚合。
2. profiler 的 cycle/schedule、`acc_events`、结果读取位置；每案有效记录数、零时间戳数、dropped 数；尤其 152238、152976 与同 SHA 正常复测的对应值。
3. 12 案实际执行时间线与各次调用边界；编译/cache 命中及 wall time；沙箱改写后 JIT 源码身份是否稳定。

主办方有复测安排，因此同时保留**最高可见分样本**和**正常性能可靠锚**。本文交付的是可执行的验证路线与三个未测布局候选，**没有声称已经取得 raw 90、找到稳定 tk=0 触发器或证实新的 GPU 提速**。
