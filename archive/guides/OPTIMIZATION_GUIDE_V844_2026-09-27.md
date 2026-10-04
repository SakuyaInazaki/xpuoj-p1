# P1 MegaMoE：v844 之后的优化方向与执行任务书

适用时间：2026-09-27，UTC+8。工作区：`/Users/sakimi/Desktop/xpuoj-p1`。投入策略：用户选择**积极突破，接受较多失败，争取明显提分**。本文件取代当天凌晨以 v842 为起点的任务安排；旧指引保留为历史证据。

**建议把主要研究预算投入两项：c9/c10 的两卡分组专家分片，以及 DN 输出量化的静态尺度。前者改变权重和通信的成本结构；后者尝试去掉当前仍存在的逐行归约。排序和 metadata 只安排有限收尾预算。c1/c2 暂不继续扫 GEMM 参数。**

这不是“已经获得提速”的报告。此次完成题面、比赛讨论、现役源码及历史结果审计，只读恢复 19 个已有 SID 的详情，重算评分与工作量，并实现了新方向的 CPU 索引验证。**本轮正式提交 0 次、custom 0 次，生产内核未修改。** 后续 GPU 结论均需要由用户安排的 coding agent 实测。

## 1. 当前起点：先核对 SHA，再做实验

| 项目 | 本轮核实结果 |
|---|---|
| 生产文件 | `p1/kernel.py` = `p1/kernel_v844_packed_compact_c910.py` |
| SHA-256 | `d1f1e0697c3e99a86f3d34bea6509755be6f70470a2373ec0bd355819c671f9e` |
| 当前 P1 榜分 | **75.17**，best SID **149493** |
| 当前账号综合排名 | **2**，三题总分 243.67；不是 P1 单题名次 |
| P1 正式提交数 | **3308**；已经处于单次新提交扣 10 分的上限 |
| 最新已有提交 | **149665**，Accepted，display 82.08，源码 SHA 精确等于 v844 |
| 最近队列 | 本轮查询最近 60 项未见 Pending/Running；开工时重新检查，不能视为永久无在途 |
| 分布式 custom | 本轮只读查询仍为 `available=false` |
| 可用实验资源 | OJ 正式四卡 P1；平台允许的单卡 custom。无本地 H800、无可假定的 profiler |

来源：[只读平台快照](reports/2026-09-27-v844-platform-readonly.json)、[19 个 SID 的脱敏逐案详情](reports/2026-09-27-v844-cases.json)。查询发生在本轮生成快照时；JSON 保存精确 UTC 时间。

75.17 对应的 SID 149493 **并不是 v844**。它的源码 SHA 为 `c23394ec00de…`，c11/c12 的 tk 是 **0.325/0.000 ms**，显著脱离正常 v844 的约 0.885/1.487 ms。原始记录保留；不能把这些计时作为结构收益、可复现性能或后续优化目标。异常的具体原因尚未定位。

已找到三个源码 SHA 精确等于 v844 的已有提交：**149603、149604、149665**。它们的 display 为 82.42、82.42、82.08，扣 10 分后分别为 72.42、72.42、72.08。这三个样本只足以提供新的描述性基线，不足以证明千分之几的增益。

## 2. 比赛约束与实现事实

### 2.1 题意与允许的优化

每个 rank 输入本地 `X[T,H]`，router 权重 `Wgate[E,H]` 和本地 `E/4` 个专家的 G/U/D 权重。路由 BF16 matmul 后按 BF16 保存，再在 FP32 做 softmax、top-k 与归一化。专家执行 gate/up、SwiGLU、路由加权、down，最后按原 token 合并到 BF16 output。

题面参考链有多处 BF16 舍入。近似计算最终必须满足：**SQNR≥22 dB、输出有限且完整、输入不变、相同输入独立调用逐字节确定**。不能用“与 v844 接近”替代与独立题意参考的 SQNR。

同一测试点权重和 topk 从预热到正式运行固定，允许缓存它们的派生表示。`hidden_states` 在预热后会更换；不能缓存输入相关的 route、activation、输出。切换测试点必须正确更新静态缓存。[本地题面](1-full.md)，原题链接：[P1](https://xpuoj.com/contest/13/problem/1)。此次公网工具未取得可读题面，题意以本地完整抓取和用户补充说明为依据。

### 2.2 新增官方答复已经解决的问题

用户提供的 [addinfo.md](/Users/sakimi/Desktop/addinfo/addinfo.md) 明确：

- P1：Triton-distributed **3.4**，对应 Triton **3.4**；确切 fork commit 未给出。
- 硬件：**4×H800 80G SXM**，默认 700 W，卡间均显示 **NV8**。不能把拓扑标记直接换算成应用实测带宽。
- **500 秒是一次 torchrun 批量执行多个测试点的总预算**。官方答复写“十个”，当前正式详情有 12 案；数量不一致不影响“不是每个新 kernel 独享 500 秒”的结论。
- 通信与 NVSHMEM 已初始化。通过 `triton_dist.utils` 使用 `is_shmem_initialized`、`nvshmem_create_tensor(s)`、`nvshmem_free_tensor_sync`、`nvshmem_barrier_all_on_stream`，不要重复 init/finalize。
- 含分布式设备操作的 kernel 使用 `triton_dist.jit`；不能在普通 `triton.jit` 中直接使用 dist 操作。
- 逐阶段 profile 入口尚未有确定答复。其他参赛者报告 `torch.cuda.Event` 被拒；本项目历史 Event 读数不等于当前仍允许。被拒后不能绕过限制。

旧 `docs/SUBMISSION.md` 的“远端版本未知”、旧方法学的“没有 SQNR 就是编译爆炸”，均需要按上述证据修正。**仅有 TLE，原因仍未定；明确编译断言才是编译证据。**

官方 Triton [v3.4 Hopper Gluon 入口](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py) 没有显式 `warpgroup_mma` 接口。不要从最新版教程拼装当前环境不存在的 API。Hopper 的 TMA、共享内存和 L2 结构可参考 [NVIDIA 架构指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)，其 H100 描述用于架构背景，不能替代评测 H800 的实测资源和带宽。

### 2.3 v844 真正执行什么

12 案均进入 **replicated** 路径：预热收集全专家权重，各卡独立计算本地 token；稳态不做 token 跨卡 dispatch/combine。不是四卡重复计算全部四卡 token。

```text
静态：跨卡收集专家 → FP8、scale、布局预处理 → 缓存
动态：路由 → 稳定排序、metadata → 输入 FP8
    → MD（gate/up + SwiGLU + route weight + ACT FP8）
    → DN（down GEMM + 输出 FP8、每 256 列 scale）
    → inverse gather + FP32 分支求和 → BF16 output
```

v844 已经拥有：token 量化消重、GU/激活/量化融合、TMA 权重读取、c9/c10 tile 连续权重、FP8 Down scratch，以及：

- c3–c8、c11/c12：direct `INV_PAD` + padded DN TMA store。
- c9/c10：compact DN + packed-key 局部稳定排序；已撤回这两案的 padded 输出。
- c1/c2：原 compact 输出，padded 实测未获益。

不要将这些写成新提案。“换成 FP8”“融合 SwiGLU”“复制静态专家”“用 TMA”“减少排序 one-hot”都必须先指明**相对 v844 还改了什么**。

### 2.4 现役源码导航

行号只对本文件冻结 SHA 有效；Python 后定义覆盖前定义。

| 入口 | v844 行号 | 后续用途 |
|---|---:|---|
| `run_kernel` | 6514 | `_CALLN`、`_FL`、`_GA` 与家族分发 |
| `_run_replicated` | 6024 | 主链，排序/量化/MD/DN/fin 接线 |
| `_get_full_fp8_weights_lowmem_tiled` | 3790 | c9/c10 静态 tiled 权重 |
| `_fgs_tma1_kernel_gq_tiled` | 1639 | c9/c10 token-gather MD，EP2 优先复用 |
| `_fgs_tma2_int_pm_q8_kernel` | 5057 | c1/c2 MD |
| `_fgs_tma1_intq_host` | 5844 | 其他 MD 变体分发 |
| `_dn_tma2_f8_kernel` / `_tiled_kernel` | 4520 / 4599 | compact DN、输出量化 |
| `_dn_tma2_f8_pad_kernel` | 6814 | 现役八案 padded DN |
| `_gather_branch_sum_f8_kernel` | 4678 | 原 slot 顺序、反量化和累加 |
| `_csort_offsets_kernel` | 733 | counts 前缀，metadata 融合候选 |
| `_prepare_moe_metadata` | 4118 | 最后生效的 metadata host |
| `_counting_sort_order_invpad` | 6860 | padded 排序链，仍有独立 pad_delta launch |
| `_counting_sort_order_packed_compact` | 7054 | c9/c10 当前排序链 |

完整最后绑定定义表：[audit JSON](reports/2026-09-27-v844-audit.json)。

## 3. 工作量与积分：什么程度的优化才值得做

以下是**每卡**工作量。`M=T·k`，专家 GEMM 的有用运算量约 `F=6MHI`（不含路由、padding 和辅助计算）；FP8 全专家权重存储量约 `W=3EHI` 字节（不含 scale）。W 是容量，不是已经测得的 HBM 读数。

| 案 | T/H/E/I/k | 平均每专家行数 | v844 tk 中位 ms | F，TFLOP | W，GiB | 主要研究价值 |
|---|---|---:|---:|---:|---:|---|
| c1 | 16384/4096/8/8192/2 | 4096 | 4.626 | 6.597 | 0.750 | 大 GEMM，已高度调优 |
| c2 | 16384/4096/8/14336/2 | 4096 | 7.919 | 11.545 | 1.3125 | 绝对耗时最大，常规路线接近饱和 |
| c3 | 16384/2048/32/2048/4 | 2048 | 1.365 | 1.649 | 0.375 | DN epilogue 可扩展对象 |
| c4 | 16384/2048/32/1024/4 | 2048 | 0.795 | 0.825 | 短 K，padded 已做 |
| c5 | 8192/3584/64/2560/8 | 1024 | 2.762 | 3.608 | 1.641 | MD 重，特殊 H，不先动 |
| c6 | 8192/3584/64/1024/8 | 1024 | 1.262 | 1.443 | 0.656 | 短 K DN |
| c7 | 16384/4096/96/2048/3 | 512 | 2.187 | 2.474 | 2.250 | E 非二次幂，旧扫描很多 |
| c8 | 16384/4096/96/1024/3 | 512 | 1.248 | 1.237 | 1.125 | 短 K DN |
| c9 | 4096/4096/256/2048/8 | 128 | 2.454 | 1.649 | 6.000 | **权重工作集大，EP2 首选** |
| c10 | 4096/4096/256/1536/8 | 128 | 1.872 | 1.237 | 4.500 | EP2 第二案，通信收益比更苛刻 |
| c11 | 65536/1024/32/1024/2 | 4096 | 0.885 | 0.825 | 0.09375 | **短 K、大 M，DN epilogue 首选** |
| c12 | 65536/1024/32/2048/2 | 4096 | 1.487 | 1.649 | 0.1875 | c11 成功后扩展 |

三个 exact-source 样本的逐案中位数之和是 28.862 ms；这是一个合成描述值，不是某一发真实总时间。c1/c2 约占它的 43.5%，但各案在积分中等权，不能仅按毫秒分配研究预算。

c1/c2 的整链有效吞吐已约 1426/1458 TFLOPS。c9/c10 若全权重每次从 HBM 读一遍，W/tk 对应约 2.63/2.58 TB/s。后两者只是成本估计，L2 重用与重复访问会改变实际流量。**想让 c9 再快几十个百分点，减少每卡必读权重比微调访存提示更有意义。**

### 3.1 评分模型与不确定性

本轮对恢复的 **216 条逐案记录**重算，观察到 `th=0`，均符合：

```text
q_i = floor(100 * tb_i / (tb_i + tk_i))
题目 raw = sum(q_i) / 12
本账号新提交罚后分 = raw - 10
达到下一整数 q：tk <= tb * (100/q - 1)
```

这只覆盖已观察区间。官方确认 100 不是通用硬上限，更快区间有对数计分；不要将上述式子外推到未观察参数或更改后的赛制。

使用同三个 v844 样本的 tb 中位数，固定评分基准后：

| case | 模型 q | 下一档要求 tk 再降 |
|---|---:|---:|
| c1 / c2 | 80 / 78 | 1.20% / 2.72% |
| c3 / c4 | 83 / 86 | 1.50% / 4.65% |
| c5 / c6 | 82 / 85 | 5.86% / 2.04% |
| c7 / c8 | 82 / 85 | 3.65% / 5.55% |
| c9 / c10 | 77 / 78 | 2.82% / 4.36% |
| c11 / c12 | 86 / 84 | 4.37% / 3.11% |

三个 tb 样本也有波动，尤其 c1；这些数字用于安排实验，不能当未来精确积分承诺。详见带 tk/tb 范围的 [阈值 CSV](reports/2026-09-27-v844-thresholds.csv)。

### 3.2 积极突破应该追求什么幅度

| 固定 tb 的条件情景 | 模型罚后分 | 相对当前模型增加 |
|---|---:|---:|
| 当前 v844 描述性模型 | 72.17 | — |
| 仅 c9/c10 各快 20% | 72.75 | +0.58 |
| 仅 c9/c10 各快 30% | 73.08 | +0.92 |
| 仅 c1/c2 各快 15% | 72.67 | +0.50 |
| c9/c10 −30%，c1/c2 −15%，c11/c12 −10% | 73.75 | +1.58 |
| 所有案快 20% | 75.17 | +3.00 |

条件模型中，若所有案同百分比降时，75/77/80 分分别约需 **19.18% / 31.43% / 49.55%**。它们不是实际目标的必要条件，也不是物理不可能证明；它们说明 **0.5% 的微优化和数分的稳定提分完全不是同一工作量**。已有榜分 75.17 与固定 tb 模型 72.17 是不同量，不能直接相减作性能回退。

可复算：[audit_v844.py](reports/audit_v844.py)、[情景 CSV](reports/2026-09-27-v844-scenarios.csv)。

## 4. 方向排序与首轮资源分配

| 优先级 | 方向 | 性质 | 首轮对象 | 主要停止条件 |
|---|---|---|---|---|
| P0 | 正确性路径、缓存合同、测量工具 | 必需基础 | v844 | 不能以 AC 代替任意调用验证 |
| R1 | **两卡一组的部分复制 + 专家分片（EP2）** | 主突破线，高风险 | c9，后 c10 | 计算节省不足支付通信、归并、skew |
| R2 | **静态权重界驱动 DN 输出 FP8 尺度** | 新数值/epilogue 机制 | c11，后 c4/c12 | subnormal 误差过大，或完整 DN+fin 无收益 |
| R3 | k=2 按专家对分组的 DN+确定归并 | 备用研究线，很高风险 | 仅 c11 | gather A、padding、B 重读超过省下的 scratch |
| R4 | offsets、PAD_DELTA、metadata 合并 | 低上限收尾 | c11/c12 | 子链不足数微秒收益，不进入正式队列 |

R1/R2 可以由用户分别派给两个 agent；它们只产候选与证据。另一个 agent 负责 P0、测量与唯一正式提交队列。R3 只有前两线等待或已经关闭时启动。**本轮没有自行创建或联系其他 agent。**

## 5. R1：两卡分组的部分专家复制

### 5.1 为什么允许重开这一种 EP

旧四卡 EP 已有明确负结果：[09-19/20 EP 记录](experiments/2026-09-19/results/ep_build.md)。最好 rank0 诊断读数 c9/c10 为 3.007/2.578 ms，旧 replicated 为 2.483/1.932 ms；即使取低 skew 样本仍慢。旧方案已经有 token-only A、索引合并免二次排序、FP8 dispatch 和 INT8 partial return，不能再把这些当新发现。

**本提案改变两项旧失败前提：** 每个 token 只与一个 partner 交换；每个计算 rank 只汇总两份 token。旧四卡 A 有 64 MiB；新 A 为 32 MiB，原 replicated 是 16 MiB。Hopper 家族 L2 容量约 50 MB，32 MiB 可能有更好的重用，但权重和其它数据仍竞争 L2，**“小于 L2”不是常驻保证**。

本轮定向检索 `experiments/`、`p1/`、历史 handoff 的 two-rank/半复制/双卡/pair/EP2 等记录，未发现同机制完整测量。检索未命中不等于从未有人试过；实现前再次核对最近实验。

### 5.2 精确定义放置与计算归属

固定两个逻辑组 `{0,1}` 与 `{2,3}`。每组持有全体专家，但分散在两个成员上：

| 计算 rank | 服务的源 rank | 持有的逻辑专家 |
|---|---|---|
| 0 | 0、1 | 0…127 |
| 1 | 0、1 | 128…255 |
| 2 | 2、3 | 0…127 |
| 3 | 2、3 | 128…255 |

对源 rank `r`、expert `e`：

```text
pair_base = 2 * (r // 2)
owner = pair_base + (e // 128)
local_expert = e % 128
source_slot = r % 2
branch_code = (source_slot*T + token)*k + slot
```

静态权重原来按四份分布；预热必须完成上述重排。第一版可复用现有低内存全量 FP8 缓存，再取逻辑半专家的只读 view，以减少新预处理错误；其容量仍是全复制，**计算只读半份**。后续才将持久权重压到半份。tiled 权重已扁平化，view 的 offset 必须乘“每 expert 对应的物理 tile 行数”，不能把 expert id 直接当扁平 tensor 第 0 维下标。

每个 source 只路由自己的动态 X。向 partner 发送自己的 token FP8、token scale、flat route weight、稳定 ORDER 与 counts。接收方把两个 source 的相同专家段拼接，expert 内固定 source_slot 0→1，再保持原 branch 顺序。`ORDER//k` 可直接索引 `[2T,H]` 的 token-only A，不物化 `[M,H]` 的 sorted A。

### 5.3 字节与工作量账

c9/c10 中 T=4096、H=4096、k=8：

| 每 rank 指标 | replicated | EP2 | 旧四卡 EP |
|---|---:|---:|---:|
| 计算专家数 | 256 | 128 | 64 |
| c9 权重工作集 | 6 GiB | 3 GiB | 1.5 GiB |
| c10 权重工作集 | 4.5 GiB | 2.25 GiB | 1.125 GiB |
| token FP8 A 工作集 | 16 MiB | 32 MiB | 64 MiB |
| 每专家平均行数 | 128 | 256 | 512 |
| 每 rank 平均分支数 | 32768 | 32768 | 32768 |
| 对外 token payload | 0 | 16 MiB | 48 MiB |
| 对外 BF16 partial payload | 0 | 32 MiB | 96 MiB |
| 对外 INT8 partial payload | 0 | 16 MiB + scales | 48 MiB + scales |

EP2 dispatch 另有约 404 KiB/rank 的 token scale、route weights、ORDER、counts；INT8 return scale 另约 256 KiB/rank。以上是 outbound payload，不应再误乘两次“双向”。网络协议、接收写入、本地 HBM 读写及同步都是额外成本。

旧 rank0 大块推送约 126–154 GB/s，只作为量级敏感性输入。假定单 peer 也能达到该范围，EP2 的两腿纯数据传输约为：BF16 return **0.33–0.40 ms**，INT8 return **0.22–0.27 ms**，未计启动、metadata、partial reduction、最终合并与 skew。**不能直接将旧四卡整链除以二或三。**

### 5.4 先证明不会漏算、越界、读上一轮数据

- 无论路由多偏，每 owner 最多接收 `2*T*k=65536` 条分支，这是严格上界；平均 32768 不能作为容量。旧四卡实现的 65536 对四个 source 只是经验上限，不能原样泛化。
- metadata tile 容量使用 `ceil(Mcap/128)+128`；Down/ACT/scratch 按实际 consumer 的 mask 和容量合同分配。descriptor load 的 padded/越界语义要保留，不能让无效行成为有效 branch。
- 某 owner 没有某 token 分支时，其 partial 必须为零，不能保留前一次内容。
- 所有 rank 以相同 shape、dtype、顺序创建对称 buffer；组内通信不代表可以让部分 rank 跳过全局集体分配。
- 双 buffer 或递增 epoch 都需要证明生命周期。发送完成、可见性、接收读完、下次复用是四个独立条件；只看到 signal 不代表源 buffer 已可覆盖。
- wait kernel 不能占满 SM 后等永远无法调度的 producer。首版按现有 stream 顺序分段执行，先不设计同一 grid 内自旋式融合。
- 任意调用次数、重复输入、新 X、交错 shape 都走相同数学协议；epoch 只协调同步，不能切换“检测/计时”数学。

此次 CPU oracle 已对随机、全部落低半专家、全部落高半专家验证：**9 组、394,944 条 branch**，每条 source/token/slot 恰好映射一次，source 恢复、expert 重编号及最坏容量通过。[脚本](reports/prove_pair_directions.py)、[结果](reports/2026-09-27-pair-directions-proof.json)。这不验证 GPU 协议、浮点或性能。

### 5.5 归并的精度合同

第一版用固定 slot 顺序在 owner 端累加本 owner 的分支，然后返回 BF16 partial；源端以固定 owner 顺序做 FP32 合并。这样可能改变 v844 的加法结合顺序并增加舍入，必须重新测 SQNR 与确定性。之后才测试 INT8 partial return。

精确保留全部原分支 FP8+scale 后回传、按原 slot 求和可作为 oracle，但通信量更大，不能把“partial return 的字节账”和“全分支原顺序的数学”混在同一个收益承诺里。

旧 EP 文档将独立误差功率相加推得只损约 0.05 dB，隐含误差不相关；不构成新协议的保证。最保守应从范数三角不等式预算，并直接对独立 BF16/FP32 参考验收。不同方案日志同为 23.13 dB，不代表输出逐字节相同。

### 5.6 按四道门推进，禁止先写完整复杂通信

**门 A：单卡计算代理。** 使用两个独立 token 源、128 个专家、真实 counts/ORDER 和静态权重 view；总有效分支量与 replicated 相当。比较完整 prepare+MD+DN。保留 32 MiB A、多 expert 权重工作集与路由偏斜，不能用一个小热矩阵循环代替。目标是看是否有足够计算节省支付通信：c9 若连 **0.4 ms** 节省都没有，通常停止；c10 对应初筛约 **0.3 ms**。这些是预算门，不是硬件定理。

单卡 custom 镜像如为 3.6，只支持局部筛选；接近门槛时仍需 P1 同栈证据。返回完整阶段结果、资源和 cold/warm 编译情况，不只报 TFLOPS。

**门 B：一对 peer 的真实 payload 往返。** 因 distributed custom 不可用，必要时用正式 P1 的诊断候选：保持完整正确输出，额外完成来自本次 X 的真实 payload 往返和校验，比较完整 tk 增量。该候选只测通信成本，不能作为优化晋升。协议必须完整计入 fence/wait/归并准备，不测完发送 launch 就声称通信结束。最多做一个 BF16 版本、一个 INT8 版本，不扫几十个 CTA 配置。

**门 C：c9 完整 EP2。** 明确成本式：

```text
预计净节省 = 当前 MD/DN - EP2 MD/DN
           - 新增通信 - 新增索引/partial reduction/最终合并 - rank 等待
```

至少留出约 0.10–0.15 ms 的不确定性空间；模型仅勉强持平时不集成。正式初筛只开 c9；所有 rank 验证完整 P1，不能只读取 rank0 最快调用。优先争取 c9 **≥8%** 的端到端信号，≥5% 可视证据继续，稳定 <3% 则不再加复杂度。其收益不预先承诺。

**门 D：复制到 c10，再研究 overlap。** c9 胜出后才做 c10，因为 c10 的权重更少、同样通信更难摊薄。无 overlap 版明确小幅落后且分析指出可隐藏的成本时，才允许一次本地 token 计算与远端 dispatch 重叠实验；须计入分两批导致的 B 重读、额外 metadata、双 launch。不能假定重叠免费或靠自旋越过当前编译器限制。

建议首轮：3–5 个单卡 custom；门 B–C 共 6–10 次正式评测含控制；达到目标后再申请固定确认批。失败也需交付“计算、通信、aux、skew 中哪项否定假设”，不是“EP 不行”一句话。

## 6. R2：静态权重界替代 DN 的动态 amax

### 6.1 当前还剩的具体成本

v844 DN 的 epilogue 为每 row、每 256 个输出列做：

```text
u = fp32_dot(Aq, Dq) * D_scale
row_max = max(abs(u), columns)
s = max(A_scale * row_max / 448, 1e-12)
q = fp8(u * (A_scale / s))
store q and s
```

当前 ACT 已用融合的幂次行尺度，但 **Down 仍做上述 max 归约与比例计算**。c11 的 K=1024，主循环短，epilogue 相对更重要。这与今天失败的“把 pointer store 改 TMA store”不同；存储布局、Tensor Core 主循环和最终 gather 第一版全部保持。

### 6.2 可推导的静态尺度，不缓存动态输入统计

设实际参与 DN 的反量化权重为 `Dhat[e,h,i]=Dq[e,h,i]*D_scale[e,h]`。静态预处理为每个 expert、256 列 chunk 算：

```text
L[e,p] = max(h in chunk p) sum_i abs(Dhat[e,h,i])
C[e,p] = 不小于 L[e,p] 的 2 的幂，并预留浮点求和的安全余量
V[e,h] = D_scale[e,h] / C[e, h//256]
```

有限 E4M3 的 `|Aq_i|<=448`，因此在实数模型下：

```text
|sum_i Aq_i * Dhat[e,h,i]| <= 448 * L[e,p] <= 448 * C[e,p]
```

于是可尝试：

```text
acc = fp32_dot(Aq, Dq)       # 原循环不变
q = fp8(acc * V[e,h])
s = A_scale[row] * C[e,p]
store q and s               # 原 Down/CSCL 地址与 fin 不变
```

理想上消掉 `tl.max(abs(...), axis=1)` 和动态 reciprocal，并将尺度的列向乘法留在 epilogue。静态 C/V 仅由当次专家权重导出，预热可缓存；**不得用第一次 hidden_states 的输出分布决定后续 scale**。

实现时必须处理：全零 D 行/块选择有限正 C；有限权重与 scale；FP32 dot/reduction 舍入余量；C/V/新 s 的下溢和上溢；无效 padded row 不被读取。可先用 `2*next_power_of_two(L)` 的保守界并验证 `C>=L`；不要依赖近似 log2 在幂边界恰好向上。可加显式有限范围 clamp 作为数值保护，但其成本计入候选。

**界只保护量化范围，不证明 22 dB。** 保守 C 可能让大量 q 落入 FP8 subnormal，精度恶化。幂次缩放在 normal 区间通常保持相对量化分辨率，但不能忽略下溢、额外舍入和最终加法。

### 6.3 为什么值得一个有界实验，为什么也可能很快关闭

收益来自去掉归约依赖和除法，不是减少 GEMM FLOP，也不减少 Down 的主字节量。长 K 的 c1/c2 很可能只获微小收益；先测 c11，再 c4/c12。c9/c10 主要受权重流量影响，最后扩展。

需要报告至少：

1. 对同一真实 MD activation，原 DN 与静态尺度 DN 的 dequant 输出、最终 output，以及独立参考的 SQNR；与 v844 的误差仅作为附加诊断。
2. q 落入 subnormal/零/饱和的比例，**以及这些元素对应的误差能量**；只有计数比例不能解释 SQNR。
3. zero input、尺度极端、全零/极小权重、路由很偏时的有限性与确定性。
4. `C,V` 预处理的内存、缓存失效和总 500 秒影响。不能因预处理不计 tk 就让全评测 TLE。
5. 当前 padded DN+fin 的完整时间；保留 `CSCL[row,chunk]`，首版不删 scale buffer、不引入 fin 二级查表。

建议数值继续门：独立参考所有筛选点至少 **22.5 dB**，并记录相对现役退化；0.5 dB 是项目保护余量，不是赛事新阈值。正式仍须全部≥22。若保守 L1 尺度导致明显下溢误差，允许一次有**动态输入保障**的改进，例如额外计算实际 Aq 行范数，但必须把它的 reduction/存储/launch 计费；不能靠反复调一个经验常数直到公开样本刚过。

建议性能继续门：c11 的 DN+fin ≥5% 改善，且按阶段占比有望贡献全案≥1.5%；完整 P1 目标≥2%。没有可信阶段时间时，以完整 P1 的固定验证批判断，不能由“少了一条 max”宣布已提速。若不到一个实用积分台阶，停止独立扩展，等待与已证实机制组合。

首轮预算 2–4 个 custom、通过后 4–8 次正式评测。R2 是本轮新假设，尚无 GPU 数学/性能证据；代码检索未发现等价 L1-bound DN 输出方案的已归档测量。

## 7. R3：k=2 的专家对 DN 与确定归并，仅备用

目标是让一个 CTA 负责一个 token 的两条 down 分支与最终相加，取消 Down scratch/CSCL/fin 往返，避免浮点 atomic。保持 MD 专家排序不变，在 DN 前按 token 的无序专家对 `(lo,hi)` 分组，同组 tokens 共享两个 D 权重矩阵。

```text
pair_id = lo*(2*E-lo-1)//2 + hi-lo-1
row_lo[t] = INV[2*t + slot_of_lo]
row_hi[t] = INV[2*t + slot_of_hi]
```

DN CTA 对这些 token gather 两份 ACT，分别做两个 dot，在固定顺序下反量化/相加，唯一写入对应 output tile。不重算 gate/up，不引入跨 CTA 浮点归约。若省掉原 DN FP8 舍入，则是数值链变化，要独立验收，不能声称与 v844 exact。

这里必须使用 compact `INV` 定位 ACT；v844 的 `INV_PAD` 只对应 padded Down，不能拿来访问紧凑 ACT。

**先只考虑 c11。** E=32 有 496 对，均匀平均约 132 tokens/pair；DN 全专家权重约 32 MiB，有机会重用。c12 的 DN 权重约 64 MiB，c1/c2 还会显著增加不同 pair 对同专家 B 的重复读取，不先推广。

此次 CPU 合法随机 top-2 路由，T=65536/E=32 的结果：原 expert BM128 共 133120 个 padded branch row；pair BM64 共 83200 个 padded token row，需要两个 dot，即 **166400 branch-row 等价工作量，约增加 25%**。BM128 pair 更差，约增加 55%。这不是免费融合。

可省的 c11 Down FP8 写+读约 256 MiB，另有 scale 与 final launch；新增的是 pair sort、ACT gather、尾部计算、B 重读和第二个 accumulator 的寄存器压力。不要只列被删字节。

首轮最多两个 custom：先用真实 pair counts 和 oracle 测已有 ACT→DN→fin 对候选 ACT→pair DN→output，**包含 pair metadata 成本**。如果理想数据已无≥10% 子链改善，立即关闭，不做正式提交、不扫 BM32/64/128 大表。BM64 是为 pair 几何选择的新算法原型，不能据此重开旧 E256 BM64。

CPU 索引验证 6 组、163874 个 token，通过 pair id 双射、原 slot 与 ACT 行的对应、唯一 output owner；它不证明 dot、舍入或速度。[验证结果](reports/2026-09-27-pair-directions-proof.json)。

## 8. R4：metadata 收尾，最多一轮

v844 的 padded sort 仍执行：hist → colscan → offsets → **pad_delta** → scatter，然后单独 `_prepare_moe_metadata`。直接 INV_PAD 已消掉 fin 的第二级查询，但尚未消掉 pad_delta launch。

融合对象是 offsets 中已可读取的 TOT：每个 expert CTA 同时得到 compact row prefix、tile prefix、PAD_DELTA，独占写自己的 metadata 区间，由一个指定 CTA 写 num_tiles。所有 CTA 读取**前一 launch 已完成的 TOT**，不能在同一 grid 中靠“另一个 CTA 应该写完 counts 了”建立依赖。

必须逐项保持 metadata 的含义：expert、该 expert 的 row 起点、该 expert tile 总数、inclusive tile prefix；这里 row 起点不是本 tile 的 row 起点。极端 skew 用循环覆盖所有 tile，容量以最坏分布推导。只比较实际有效 metadata，不比较未初始化 capacity 尾部。

这最多省少量 launch 与重复前缀操作。两次 custom 内确认，完整链预计不足 0.5% 或只省 1–2 μs 时不进入正式队列。`tl.histogram` 替换仍用 dense one-hot 的 hist 可另作一次小原型，但 packed scatter 已做，别从头重写排序再称新方向。

## 9. c1/c2 为何暂时不给“下一组参数”

c1/c2 是主要计算量，但已经尝试：TMA/指针布局、大 tile、FP16 累加、两 CTA、深流水、GU interleave、ACT TMA store、输出 TMA store、GM/stages/warps/register cap 等。多次降低寄存器或共享内存并未提高实际吞吐。

今天新增直接负证据：

| 改动 | SID / 结果 | 后续处理 |
|---|---|---|
| c1/c2 ACT TMA store | 149609/149612：4.994–5.021 / 8.51–8.523 ms，相对正常约 4.6 / 7.8–7.9 明显慢 | 关闭此 store 改写 |
| 单宽 accumulator、宽 B load | 149630：c1 4.706，c2 7.817 ms | c1 明确不利，c2 小差异未证；不扩扫 |
| gran=1 GU interleave | 149636：4.722 / 8.014 ms | 关闭当前实现 |
| c11/c12 token-only MD | 149614：0.917 / 1.567 ms | 保留 sorted-A/TMA A |
| c4 compact TMA store，无 flatten | 149653：0.849 ms，对约 0.795 | 关闭；149647 TLE 本身不作为性能负证据 |
| final output TMA store | 149659 vs exact v844 149665：正常案总体无清楚收益 | 不晋升；c11/c12 异常低值不算提速 |

第一行时间范围来自 149609/149612 两发，其中 149609 其他案存在异常低计时；c1/c2 自身的明显回退仍可见，优先依赖正常 149612。

若以后要重新立项 c1/c2 主 GEMM，先给出**改变哪个旧失败前提**，再交同栈、同几何、包含 layout/epilogue 的最小基准。纯 dense 微基准赢了仍不够，必须在主链保留收益。没有新前提就把预算留给 R1/R2，不能因其绝对 ms 大而强行制造方向。

## 10. 历史路线的裁决与本轮纠错

| 方向 | 有效结论 | 重启条件 |
|---|---|---|
| 四卡 EP 原协议 | rank0 诊断明显慢；scored 版本未提交，不能称完整 12 案 AC | R1 改了 peer 数、A 工作集和字节；原协议不重跑 |
| packed FP6/INT6/INT4 | 解码、spill、旧 MMA 吞吐抵消压缩 | 原生可用计算路径或包含解码的整链证据改变 |
| FP16 accumulator + 大 tile | 同栈旧纯计算约有 20% lowering 损失；完整实现多次慢 | 新编译器/新 lowering 先实测超过现役 FP32 acc |
| E256 BM64 / BM128+BM64 tail | 已有 B 重读与调度负结果，混合尾也失败 | 先改变实际 B 流量，不重扫 tile |
| MD→DN 同 CTA 融合 | 旧 RS 型量化 dot 与私有 scratch 版明显慢 | 新机制摆脱相同寄存器/重读代价；R3 不重算 MD |
| 动态输入/输出缓存 | 不符合题意；历史 dead 分支不构成授权 | 不立项 |
| 路由分支剪枝、2:4 直接置零 | 路由近均匀、历史精度失败 | 新独立误差证据与确有计算稀疏机制；当前不投入 |
| 常规 launch sweep | 微效应与机器差异交织，已有数百次探索 | 新算法资源结构改变后只做少数必要配置 |

证据入口：[历史审计](reports/2026-09-27-history-audit.md)、[FP16 大 tile 专项](experiments/2026-09-20/results/f16acc_bm256.md)、[今日摘要](experiments/2026-09-27/SUMMARY.md)。

本轮发现的测量问题必须反馈给后续 agent：

- **149639 的 SHA 是 `e4fa1704f87f…`，不是 v844**，且 c7/c8/c11/c12 同时异常降低。今日摘要用它判断 149643 packed warps 变慢，证据不足以作严格 v844 A/B。维持现役配置即可，不据此自动追加 warp sweep。
- `scripts/mnorm.py` 的旧机器签名曾给出假阳性；不能原样作晋升裁判。
- `scripts/sqnr.py` 对 dict 形式 userError 读取 `data`，而平台常返回 `content`，并用枚举位置标案；会漏 SQNR 或串案。使用本轮保存的按 `tc=` 匹配数据，后续解析器需兼容两种形式。
- `score=100` 有时表示 correctness，性能要看 testcase `displayScore` 和 schema 指标；不要混用。
- 凌晨的 `reports/score_audit_20260927.py` 绑定 v842 SHA，面对 v844 会主动停止；新工作使用 `reports/audit_v844.py`。
- `docs/STATE.md`、今日 SUMMARY 含前后冲突的“当前”段落；源码 SHA 和对应 SID 优先于文件页首措辞。

## 11. P0：必须贯穿研究的正确性与缓存工作

### 11.1 数学不能随调用序号变化

v844 仍有 `_CALLN` 全局计数：router 在 call1/2/≥3 不同；`_GA` 只在 call3–5 对部分 shape 成立；call≥6 的部分 MD/输入量化路径又切换。新结构不能再增加一个只覆盖某几个调用的快路径。

先产静态路径表，再为目标 family 建任意调用一致的动态路径，保留一次静态缓存预处理。用同一输入在第 1/2/3/5/6/8 次及 shape A→B→A 中比较；两次独立重复需 byte exact。数值变化对独立参考比较。保留合法未知 shape fallback，不能只通过公开 12 案。

统一路径可能减少真正执行的 JIT 特化，但不是将 `_CALLN` 条件全设真；旧全开有总预算 TLE。先改一个 family。死代码删除并不自动减少 JIT 编译，改行号还可能改变 cache key，因此大规模清理不与提分候选混在一起。

### 11.2 shape-only 权重缓存的合同未闭合

现役多个权重缓存只按 shape。连续两个 same-shape、不同权重可能错误命中；`id()` 也不能处理同对象原位修改和 id 重用。题面只承诺同一测试点静态。新增 R1/R2 缓存必须至少包括设备、dtype、stride、全部标量、输入身份和正确生命周期引用，并对同 shape 更换权重测试。

仍需主办方确认：跨测试点是否可在同 tensor/storage 上原位更改，以及有无允许的 generation/version 信号。若无合同且要完全正确，就应验证内容或重新预处理，其代价需实测。抽样 fingerprint 不能作为完全一致性证明；不能绕过被禁止的 `_version`。

该问题不阻塞 CPU 设计和单卡单次调用筛选，但会影响最终交付的稳健性。不要把未经证实的“每次都 clone”或“永远同权重”写成事实。

## 12. 只用 OJ 的实验协议

### 12.1 三层证据，分别标注

1. **CPU/静态**：索引、容量、数学变换、语法、最后绑定定义、diff/SHA。只能排除逻辑错误。
2. **单卡 custom**：目标算子的正确性、资源和完整被改子链；记录真实镜像/版本差异。不能认证 NVSHMEM 协议或四卡时间。
3. **P1 正式**：全部 rank、sample/fallback、12 案、输入只读、有限性、确定性、完整 tk 与总预算。只有这一层可以晋升。

生成 custom 的 timing wrapper 只使用当时允许的接口。若 Event 被拒，就用平台完整 workload 的 tk 做 A/B；不通过字符串拼接、私有 API 或其他语言绕过。量不到阶段就写“未知”。不为跨版本便利调用正式题面禁止的主 GEMM/官方融合接口。

### 12.2 防止被缓存和计时噪声误导

- 新动态 X、新路由每次都重算。静态预处理按题面允许计费范围区分；新增动态准备必须包含在候选内。
- microbenchmark 的权重/token 工作集要与目标接近，避免反复读一个 L2 小块得到虚高吞吐；合成 counts 同时覆盖均衡、空专家、极偏与 127/128/129 边界。
- 初筛固定 AB/BA，各自存 raw tk/tb；共享一个 anchor 不能当成许多独立样本。
- 确认使用新的一批冻结候选与 anchor，预先确定样本数。对 R1 这种预期≥5% 的大效应，先 2 对筛选，成功后 4 对确认；小效应不靠不断追加直到显著。
- 机器校正只能作辅助；控制集排除全部触碰案，分别检查原始配对与同窗同码分布。全案修改时没有未触碰控制，不强行去噪。
- tk=0、与同码分布悬殊的低值、多个未改案同步断崖下降，保留原始值并标记无效性能证据；剔除规则在看候选收益前确定。低于基线一点不能自动判异常。
- 用冻结 tb 折算结构收益，同时记录真实榜分；不靠重投碰高 tb 或异常低 tk。
- 不能把精度日志两位小数相同叫 bitwise 相同；只在输出直接比较或平台自身 determinism 明确通过时用对应措辞。

### 12.3 一个正式提交者

按 [提交说明](docs/SUBMISSION.md) 使用已验证通道。只读查队列，冻结源文件与 SHA 后才提交；拿到 SID 后只追踪该 SID。Pending、本地 timeout、令牌水位不足不代表需要再发一份。不要取消其他 agent 的未知任务。

此次用户已选择积极突破；后续不为正常可逆的实验设计反复确认。预算用来约束信息价值，不是额外审批流程。外部邮件/论坛询问仍由用户发出，本轮未向其他人发送消息。

### 12.4 每个候选统一交付格式

```text
唯一机制、目标 case、改变了哪个历史失败前提：
baseline path/SHA；candidate path/SHA；精确 diff：
数学路径与 cache 生命周期：
新增/删除的 bytes、launch、JIT 特化、峰值显存：
CPU/静态检查及覆盖边界：
custom CID、版本、oracle、资源、完整子链时间：
正式 SID、anchor SID、所有 rank/案状态：
逐案 tk/tb/th/displayScore/SQNR/determinism 原始数据：
异常标记、固定 tb 积分变化、独立确认结果：
继续/停止：触发哪个预设门；剩余最大可能收益：
集成后累计收益与回退路径：
```

## 13. 可直接交给 coding agent 的任务

### 任务 A：EP2 计算与通信可行性

> 从 `/Users/sakimi/Desktop/xpuoj-p1/p1/kernel_v844_packed_compact_c910.py` 的 SHA `d1f1e0697c3e99a86f3d34bea6509755be6f70470a2373ec0bd355819c671f9e` 出发，执行本指南第 5 节。先交一个两源、128 专家、token-only A 的单卡计算代理与独立 oracle，比较相同有效工作量的 prepare+MD+DN。证明最坏 65536 branch 容量与 expert/source/slot 映射。计算节省通过预算门后才构建一对 peer 的完整往返诊断，初版不做 overlap。只开 c9 的完整 EP2，成功再 c10。保留任意调用顺序正确性；不得只在计时窗口开协议。交源码、diff、SHA、bytes/时间预算、所有结果和停止判断；生产文件不覆盖，平台由唯一执行者提交。

### 任务 B：静态尺度 DN

> 阅读第 6 节，先只抽取 v844 c11 的 padded DN+fin。以实际反量化 D 的 L1 上界构建每 expert/chunk 的幂次 C 和预缩放 V，替换 epilogue 动态 row_max，保持 dot、输出布局、CSCL 和 fin。第一版不要同时改 BM/BN/stages/warps。独立参考测最终 SQNR，报告 subnormal/zero/saturation 的误差能量和有限性。若保守尺度精度或完整子链失败，给出证据后停止，不扫公开样本经验常数。通过后再提仅 c11 的 P1 候选与明确预期收益。

### 任务 C：正确性、证据与有限收尾

> 先按最后绑定函数做 v844 的 case/call 路径与缓存清单，修正分析器按 `tc=` 归案和 `userError.content/data` 解析。验证候选 source SHA，不把 149639 当 exact v844。支持 A/B 的 CPU 容量、语义和独立 oracle；最多两次 custom 检查第 8 节 offsets+PAD_DELTA+metadata 融合。若上限不足，不发正式微优化。整理统一 manifest、原始结果和后续任务卡，不大改历史文件或生产 kernel。

### 唯一平台执行者

> 只运行已冻结、附假设/成本/停止门的候选。先查队列；串行提交，记录 CID/SID 和准确源文件 SHA。R1 的通信验收须四卡 P1，单卡结果不替代。按固定 AB/BA 顺序执行初筛和独立确认。每次反馈正确性、原始时间、固定 tb 积分与异常情况；不为碰榜面重复同码。所有结果落到独立的当日实验目录，避免多个 agent 争写一个 SUMMARY。

## 14. 预算、日程和最终收口

比赛本地规则副本写截止 **2026-10-01 23:59（UTC+8）**，当前公告是否调整尚未重新取得正文，按未延期准备。[比赛材料](比赛信息/xpuoj-d-31-deliverable-20260806/full.md)

建议积极版本的**首批总预算：12–18 个 custom、24–36 次正式评测，全部包含控制和确认**。这不是必须花完的额度；失败门提前命中就停止。只有至少一条主线显示明确收益，才追加到约 50–60 次正式评测；不要先下发 60 个无机制候选。

| 阶段 | 交付与决策 |
|---|---|
| 第一个工作时段 | A 完成单卡计算代理；B 完成尺度数学/精度门；C 完成路径、SHA 与解析器。各给一页可回收收益预算 |
| 第二个工作时段 | A 若有余量才做 peer 往返与 c9；B 若通过才上 c11。失败时先按门收口，R3 仅作最多两次 custom 备用 |
| 后续一天 | 对幸存候选独立确认、扩一案、测试组合累计收益。不能把各自收益简单相加 |
| 截止前至少一个工作时段 | 冻结最终文件、完整正确性/确定性/缓存检查、记录正式 SID、回退 SHA；停止引入新通信协议 |

规则含赛后复测，异常榜分与调用次数依赖尤其需要在收口前处理。性能研究可与基础审计并行，但最终生产晋升必须同时有完整语义和完整 P1 证据。

仍需用户或主办方补充的高价值信息只有：**P1 静态缓存跨测试点的失效合同；合法的逐阶段诊断入口与 PTXAS/总预算日志；若已延期则提供新截止时间。** GPU 型号、拓扑、3.4 版本和初始化接口已经明确，不重复问。上述未决事项不妨碍当前两条主线的本地与单卡筛选。

## 15. 本轮附带工具与证据

- [只读 OJ 刷新工具](reports/refresh_v844_readonly.py)：使用项目已有会话，仅读取榜面/已有 SID，不取令牌或提交；必须从工作区运行，不输出凭据。
- [平台快照](reports/2026-09-27-v844-platform-readonly.json)、[源码 SHA 与逐案详情](reports/2026-09-27-v844-cases.json)。
- [离线评分与工作量重算](reports/audit_v844.py)、[manifest](reports/2026-09-27-v844-audit.json)、[阈值](reports/2026-09-27-v844-thresholds.csv)、[情景](reports/2026-09-27-v844-scenarios.csv)。
- [两卡映射/专家对分组的 CPU oracle](reports/prove_pair_directions.py)、[通过结果](reports/2026-09-27-pair-directions-proof.json)。
- [原 v842 指引](OPTIMIZATION_GUIDE_2026-09-27.md)：仅供历史机制、旧实验依据查阅；其任务 A/B 已由今天的 v843/v844 实现。

本地复核命令：

```bash
cd /Users/sakimi/Desktop/xpuoj-p1
shasum -a 256 p1/kernel.py
python3 reports/audit_v844.py
python3 reports/prove_pair_directions.py
```

两条主线的共同要求是先区分“数学可行”“成本可行”“完整 P1 提分”。本指南提供前两步的具体推导、证据边界与执行门，GPU 结果出来后再更新方向，不预支尚未测得的收益。
