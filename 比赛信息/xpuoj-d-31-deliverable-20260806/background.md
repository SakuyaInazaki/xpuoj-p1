# XPU-OJ 第一届算子优化比赛 — 赛题背景、硬件与评测环境、Baseline 分析与调研综述

> 本文档与 `goal.md`、`workflow.md` 配套使用。**全部赛题要素均直接引自原始比赛说明**（XPU-OJ 讨论帖 `https://xpuoj.com/d/31`，页面标题「XPU-OJ 第一届算子优化比赛」，由账号 `ceerrep` 发布于比赛开始前，抓取时间为 2026-08-06），引文以「题包原文」标注。
>
> 信息分级约定：
> - **[A] 题包事实**：来自比赛说明原文，可放心引用；
> - **[B] 调研结论**：来自 2023 年至今的公开论文/开源仓库/官方文档，附来源 URL；
> - **[C] 待确认**：题包无法确认、需报名登录或咨询主办方后才能确定的信息，一律显式标注。

---

## 1. 赛题背景

### 1.1 比赛概况

**题包原文（节选）**：

> 大模型训练与推理性能高度依赖底层算子。随着模型结构向 MoE、长上下文注意力和新型残差连接演进,单纯组合现有框架算子往往难以充分利用 GPU 的计算、存储与通信能力。本次比赛选取三类具有代表性的前沿算子,鼓励参赛者从内核融合、访存优化、并行划分、通信计算重叠和数值稳定性等方向开展优化。
>
> 比赛入口: XPU OJ 第一届算子优化比赛
> 主办团队: XPU OJ 组织( github.com/XPUOJ )

- 比赛入口：https://xpuoj.com/contest/13 [A]
- 主办团队：XPU OJ 组织，GitHub: https://github.com/XPUOJ [A]
- 比赛页面讨论帖（原始信息来源）：https://xpuoj.com/d/31 [A]
- 比赛全程在线进行，共 **3 道算子优化题**，比赛期间可随时提交 [A]

### 1.2 赛事安排与关键时间点

**题包原文**：

> 比赛于 2026 年 10 月 1 日 23:59 结束后,平台将停止接收新提交并锁定比赛成绩。主办方将依据比赛结束时的排行榜,对获奖候选提交进行统一复测,重点核验计算结果正确性、运行性能、执行确定性、代码原创性及参赛合规性。
> 最终结果拟于 2026 年 10 月 23 日公布。

| 事项 | 时间（UTC+8） | 说明 [A] |
| --- | --- | --- |
| 比赛开始 | 2026-08-06 00:00 | 开放报名、赛题、代码提交及排行榜 |
| 比赛结束 | 2026-10-01 23:59 | 停止接收新提交，锁定成绩 |
| 结果复核 | 结束后至公布 | 复核正确性、性能、执行确定性、代码原创性、合规性 |
| 结果公布 | 2026-10-23 | 公告/邮箱/赛事群同步发布 |

### 1.3 报名与参赛形式

**题包原文（节选）**：

> 比赛采用单账号参赛方式,每个人/团队一个账号。成绩以账号为单位计分。
> 账号需要是正式账号,避免使用其他比赛的专用账号参赛。禁止使用多账号参赛。
> 在整个比赛持续过程中都可以报名参赛,后续比赛的通知会通过邮箱的形式推送,请各位参赛选手及时关注参赛账号邮箱。

### 1.4 计分与排行榜规则

**题包原文（节选）**：

> 比赛采用分数制。每道题保留参赛者在比赛期间取得的最佳成绩,比赛总分为三道题最佳成绩之和。排行榜首先按总分从高到低排序;总分相同时,按最后一次有效的提交时间由早到晚排序。成绩以 XPUOJ 评测机返回结果为准。
>
> 每道题默认允许尝试 100 次;每位参赛选手同一道题的评测完成提交按录入排行榜顺序计为尝试次数。超过允许次数后,每次新提交的得分将按以下公式扣罚:
> 扣分 =(本次尝试次数 - 允许次数)× 0.1 分,单次扣分最多 10 分。默认允许 100 次时,第 200 次达到扣分上限。
>
> 扣分只作用于当次新提交;排序靠前的提交不会被后续提交影响。各次提交独立计算罚分,该题最终成绩取所有提交中罚分后的最高分。

- 总分 = 三题最佳成绩之和 [A]
- 同分时按「最后有效提交时间早→晚」排序（越早提交者排前）[A]
- 每道题默认允许 **100 次评测完成提交**；超限后扣分公式如上，单次扣分上限 10 分 [A]
- 排行榜实时可见他人分数，但**不可见他人提交代码** [A]
- 比赛期间与结束后，主办方可对获奖候选提交进行统一复测、代码审查与合规性检查；以最终公布结果为准 [A]

### 1.5 奖项设置

**题包原文（表格）**：

| 奖项 | 名额 | 评选方式 | 奖励内容 |
| --- | --- | --- | --- |
| 综合一等奖 | 1 | 最终综合排行榜第 1 名 | 奖金 10000 元 |
| 综合二等奖 | 2 | 最终综合排行榜第 2-3 名 | 奖金 5000 元 |
| 综合三等奖 | 4 | 最终综合排行榜第 4-7 名 | 奖金 2500 元 |

### 1.6 参赛规范（直接影响工程做法的条款）

**题包原文（节选）**：

> - 允许使用 AI 编程工具辅助开发,但参赛者须对最终代码的正确性、安全性和合规性负责。
> - 禁止攻击平台、探测评测数据、读取其他提交或以其他方式绕过真实计算。
> - 禁止提交恶意代码、挖矿程序或可能破坏评测机稳定性的内容。
> - 禁止通过高频提交来影响平台负载。
> - 主办方有权对异常提交进行复测或取消成绩,并保留对规则的最终解释权。

**题包引用**：平台官方「XPUOJ 代码评测指南」https://xpuoj.com/d/2 [A]（详见第 3 节）。

---

## 2. 三道赛题详述（含输出约束）

**题包原文（总述）**：

> 本次比赛共分为三个赛题。比赛不限定单一实现路线,除赛题一需使用 Triton-distributed 实现外,其余赛题均支持 CUDA、Triton 与 TileLang 实现。参赛者需完成指定的 run_kernel 函数,并将计算结果写入约定输出张量。

### 2.1 赛题一：Triton-distributed 算子优化 - MegaMoE

**题包原文（题目描述）**：

> 本题面向单机 4 卡 H800 GPU 上的专家并行 MoE 推理。四个 rank 各持有本地 token 和四分之一的专家,router 权重在四卡上相同。你需要使用 Triton-distributed 实现 MegaKernel。输入包括 BF16 的 hidden_states、gate_weight 和三组专家权重,以及整数 topk;计算结果写入 BF16 的 output。
>
> MegaKernel 先在源 rank 上计算本地 token 面向全部专家的路由结果,再将 token 发送到所选专家所在的 rank。专家 rank 完成 gate/up、SwiGLU、路由加权和 down 投影,并将专家分支结果送回源 rank。源 rank 按原 token 合并各分支,写入 output 中对应的行。

**题包原文（输出约束）**：

> - 以独立的 BF16/FP32 混合精度参考实现为准,SQNR 不低于 22 dB
> - 全部元素有限且不得修改输入
> - 评测程序会对同一输入运行两次,逐字节比较输出,需满足逐字节一致

赛题一要素归纳 [A]：

| 要素 | 内容 |
| --- | --- |
| 硬件 | 单机 4 卡 H800，专家并行（EP）MoE 推理 |
| 数据分布 | 每 rank 持有本地 token + 全部专家权重的 1/4；router 权重（gate_weight）四卡相同 |
| 输入 | BF16 hidden_states、BF16 gate_weight、三组 BF16 专家权重（对应 gate/up/down 投影，形状 [C] 待确认）、整数 topk |
| 输出 | BF16 output（按原 token 顺序合并各专家分支结果） |
| 算法流程 | ①源 rank 路由：本地 token 对所有专家算 logits 取 top-k → ②把 token 发送到其选中专家所在 rank → ③专家 rank 做 gate/up 投影、SwiGLU、路由权重加权、down 投影 → ④结果送回源 rank → ⑤源 rank 按原 token 顺序合并各分支写入 output |
| 实现约束 | **必须使用 Triton-distributed** 实现 MegaKernel；实现 run_kernel 并写入输出张量 |
| 精度约束 | 相对独立 BF16/FP32 混合精度参考实现，SQNR ≥ 22 dB |
| 确定性约束 | 同一输入运行两次，输出逐字节一致（严格确定性） |
| 其他 | 全部元素有限；不得修改输入 |
| 预热预处理 | 题包未在赛题一中写明（仅赛题二、三写明）；是否同样适用 [C] |

### 2.2 赛题二：MagiAttention 算子优化

**题包原文（题目描述）**：

> 本题要求在单卡 H800 GPU 上实现并优化 MagiAttention 前向 kernel。Kernel 通过多个 attention slice 描述长序列中的不同 attention 区域,并支持 GQA 与 Attention Sink。Attention Sink 在普通 attention 的 Softmax 分母中加入若干额外 logits;这些 logits 没有对应的 V,因此只改变真实 K 的归一化权重。输入包括 BF16 的 Q/K/V 以及相应的 slice 和 Sink 配置,输出为 BF16 attention 结果。同一测试点内,部分输入保持不变,可在预热阶段预处理。
>
> 对于每个 Q token 和 Q head,kernel 先根据 slice 的 range 和 mask 确定可见的 K token,并按照 GQA 规则选择 KV head。每个 slice 分别计算局部 attention 结果和 log-sum-exp (LSE);同一 Q token 被多个 slice 覆盖时,再通过 LSE 合并这些局部结果。全部 slice 合并后,将 Sink logits 加入一次 Softmax 分母,并按新的归一化项缩放已合并的 attention 输出。

**题包原文（输出约束）**：

> - 以独立的 BF16/FP32 混合精度参考实现为准,SQNR 不低于 26 dB

赛题二要素归纳 [A]：

| 要素 | 内容 |
| --- | --- |
| 硬件 | 单卡 H800 |
| 输入 | BF16 Q/K/V + slice 配置（range、mask）+ Sink 配置；输出 BF16 attention |
| 算法 | 多 slice 分区注意力 + GQA（KV head 按组共享）+ Attention Sink（仅进 Softmax 分母，无对应 V） |
| 合并机制 | 每个 slice 算局部 attention 与 LSE；同一 Q token 被多 slice 覆盖时按 LSE 合并；全部合并后再把 Sink logits 加入一次分母并缩放输出 |
| 实现约束 | CUDA / Triton / TileLang 均可；实现 run_kernel |
| 精度约束 | SQNR ≥ 26 dB（相对 BF16/FP32 混合精度参考实现） |
| 预热预处理 | **明确允许**：同一测试点内部分输入不变，可在预热阶段预处理 |
| 确定性约束 | 题包未在赛题二写明逐字节一致性要求 [C]（赛题一明确要求；建议默认按确定性实现） |

### 2.3 赛题三：mHC 算子优化

**题包原文（题目描述）**：

> 本题要求在单卡 H800 GPU 上实现并优化 mHC (Manifold-Constrained Hyper-Connections) 前向 kernel。mHC 将 Transformer 的单条残差流扩展为 n 条并行残差流,并动态决定子层如何从这些残差流读取输入、再将结果写回。输入包括 n 路 BF16 residual 以及生成混合系数和子层 F 所需的参数,输出为更新后的 n 路 BF16 residual。同一测试点内,部分输入保持不变,可在预热阶段预处理。
>
> 对于每个 token,mHC 根据 n 路 residual 生成读入、写回和残差混合三组动态系数。读入系数将多路 residual 合成为一路,送入由 SiLU 和低秩 MLP 组成的子层 F;残差混合矩阵重组原有 residual,写回系数将 F 的输出分配到各路。两部分相加后得到更新的 residual。

**题包原文（输出约束）**：

> - 以独立的 BF16/FP32 混合精度参考实现为准,SQNR 不低于 28 dB

赛题三要素归纳 [A]：

| 要素 | 内容 |
| --- | --- |
| 硬件 | 单卡 H800 |
| 输入 | n 路 BF16 residual + 生成混合系数所需的参数（φ、α、b 等）+ 子层 F 参数（低秩 MLP 的 Down/Up 权重）；输出更新后的 n 路 BF16 residual |
| 算法 | 每 token 动态生成三组系数：读入系数（多路→一路）、写回系数（F 输出→各路）、残差混合矩阵（重组原有 n 路）；子层 F = SiLU + 低秩 MLP；两部分相加 |
| 实现约束 | CUDA / Triton / TileLang 均可；实现 run_kernel |
| 精度约束 | SQNR ≥ 28 dB（相对 BF16/FP32 混合精度参考实现） |
| 预热预处理 | **明确允许**：同一测试点内部分输入不变，可在预热阶段预处理 |
| 确定性约束 | 题包未写明逐字节一致性要求 [C]（建议默认按确定性实现） |

---

## 3. 硬件与评测环境

### 3.1 H800 GPU 规格（调研口径）

评测机为 **NVIDIA H800 SXM5**。H800 是面向中国市场的 H100 变体，硬件主体与 H100 相同（同一 GH100 die），官方文档明确差异仅两处：NVLink 带宽与 FP64 性能。

| 项目 | H800 SXM5 80GB | 备注 |
| --- | --- | --- |
| SM 数 | 132 | [B] 多源一致（TechPowerUp/CpuTronic） |
| FP32 核心 | 16,896 | [B] |
| BF16/FP16 Tensor Core（稠密） | ≈ 989 TFLOPS（含稀疏 1,979） | [B] |
| FP32 | 67 TFLOPS | [B] |
| FP64 | 1 TFLOPS（H100 为 34，已被阉割） | [B] |
| 显存 | 80 GB HBM3 | [B] |
| HBM3 带宽 | **3.35 TB/s（5120-bit）** | [B] 主流权威口径；个别中文资料称 2.04 TB/s，判定为张冠李戴，仍标注 [C] 待厂商 datasheet 最终确认 |
| NVLink | 400 GB/s 双向（H100 为 900，已被阉割） | [B] Lenovo Press 官方文档白纸黑字；SXM 版数值 [C] |
| L2 | 50 MB | [B] |
| TDP | 700 W | [B] |

来源：
- H100 官方规格页：https://www.nvidia.com/en-us/data-center/h100/
- H800 SXM5（TechPowerUp）：https://www.techpowerup.com/gpu-specs/h800-sxm5.c3975
- H800 SXM5（CpuTronic）：https://cputronic.com/en/gpu/nvidia-h800-sxm5
- Lenovo Press 官方文档（H800 与 H100 差异）：https://lenovopress.lenovo.com/lp1814.pdf

**对赛题的影响 [B]**：赛题二、三为单卡算子，BF16 Tensor Core ≈989 TFLOPS 与 3.35 TB/s HBM 是优化上限的来源（T_h 的估算口径，见 3.3）；赛题一的卡间通信带宽上限为 400 GB/s 双向（单向 200 GB/s/卡）。注意：评测机上可能多卡共享 NVSwitch，4 卡拓扑为 8 卡基板插 4 卡（NVSwitch 全互联）还是直连，**影响 all-to-all 通信上限** [C]。

### 3.2 评测流程（XPUOJ 代码评测指南，题包引用页面 https://xpuoj.com/d/2）

以下信息来自平台官方评测指南（题包明确引用该指南）。指南为 SPA 页面，正文经调研从平台 API 获取；**数值细节（预热次数、测速次数）以平台实际为准** [C]：

- 每测试点流程：生成随机数据（通常 8 组，同测试点种子固定）→ **预热**（通常 100 次连续执行，不计时，消除 JIT/缓存开销）→ **测速**（通常 2000 次连续执行，GPU 端 cupti 计时取平均值作为 T_k）→ **校验**（输出与 PyTorch baseline 输出按题目容差比对）→ 按同流程测 baseline 得 T_b [B/C]
- T_h 口径：按题目计算量与硬件峰值估算，`T_h ≈ max(flops/peak_tflops, bytes/peak_bw)` [B/C]
- 评测机 CUDA 版本：各评测机不同，**均低于 12.6**（平台管理员回复）[B/C]
- 每个测试点有总运行时间上限（防进程卡死）[B/C]
- 计时纪律：禁止在 run_kernel 内调用 cudaDeviceSynchronize 类同步（污染计时）；内存监控只看 CPU 侧 RSS [B/C]

**提交接口与沙箱约束（指南要点，[B/C]，动手前必须向平台确认）**：

| 语言 | 接口形态 | 关键限制 |
| --- | --- | --- |
| CUDA | run_kernel(*args) 编译为动态库自动绑定符号 | Tensor→void* 设备指针（保证连续）；int→int64_t；float→float；bool→bool |
| Triton | 同名 Python 函数（无装饰器），torch 张量直接传入 | Python 沙箱：仅允许 torch/triton/triton.language/math；恰好 1 个 @triton.jit（可带参）+ 至多 1 个 @triton.autotune；kernel 参数除 tl.constexpr 外禁注解；torch.* 白名单制；禁 os/sys/subprocess/socket/pickle、exec/eval/compile 等 |
| TileLang | @tilelang.jit 包装 + @T.prim_func 内层 | 同 Triton 沙箱；允许 torch/tilelang/math |

**重要推论 [B/C]**：
- 若用 Triton 提交，**只能在评测进程里放一个 @triton.jit + 至多一个 @triton.autotune**——预热阶段也受此约束，需要提前规划 kernel 拆分与 autotune 的使用；
- 平台沙箱细节以指南原文与实测为准 [C]；
- 赛题一（Triton-distributed）的评测方式（如何在 4 卡上启动、如何给 NVSHMEM 分配对称内存、是否需要提交 4 进程代码）**题包未说明** [C]，需报名后看题目详情或咨询主办方。

### 3.3 评分公式与性能分数的含义（题包原文，务必逐字理解）

**题包原文**：

> 每个测试点都会执行输出约束校验,不满足约束则该测试点得 0 分。约束校验通过后,再计算性能分数:
>
> 设测试点 baseline 分数锚点为 s(默认为0.5), T_k 为参赛实现的平均运行时间, T_b 为 baseline 参考实现的平均运行时间, T_h 为依据题目计算量、显存访问量及硬件峰值性能估算的理论最短运行时间(不代表绝对物理极限)。在 T_k > T_h 的计分区间内,性能得分按以下公式计算:
>
> Score(T_k)=⌊100·max(0,[1+(1/s−1)·(T_k−T_h)/(T_b−T_h)]⁻¹)⌋,  T_k > T_h
>
> 正常情况下,单测试点最高为 100 分,各测试点按题目配置平均为该题总分。比赛对每名参赛者保留每题历史最佳成绩。

**分数-时间关系表（题包原文表格）**：

| 性能表现 | 分数含义 |
| --- | --- |
| T_k > T_b | 慢于 baseline |
| T_k = T_b | 与 baseline 等速 |
| T_h < T_k < T_b | 快于 baseline、尚未达到硬件估算上界 |
| T_k = T_h | 达到硬件估算上界,对应 100 分 |

**公式推导（[A] 数学推论，可放心使用）**：设 s = 0.5，则 1/s − 1 = 1，公式简化为

```
Score(T_k) = ⌊ 100 / ( 1 + (T_k − T_h)/(T_b − T_h) ) ⌋ ,   T_k > T_h
```

- T_k = T_b → 50 分（baseline 锚点）；
- T_k = T_h → 100 分（公式中取极限，实际要求 T_k 充分接近 T_h，见下）；
- **分数 ≥ X 的充分必要条件**：`T_k ≤ T_h + (100/X − 1)⁻¹·(T_b − T_h)`。几个常用锚点（X=60 → 0.667 区间；X=70 → 0.429；X=80 → 0.25；X=85 → 0.176；X=90 → 0.111；X=95 → 0.0526；X=97 → 0.0309）。即：**要拿 85 分以上，T_k 必须落在 T_h 与 T_b 之间、且距离 T_h 不超过 T_b−T_h 的 17.6%**；分数越接近 100 对 T_h 的逼近要求越苛刻（90 分→11%，95 分→5.3%）。
- T_k < T_h 区间：题包公式未定义该区间（说明「在 T_k > T_h 的计分区间内」）。调研到评测指南显示 T_k < T_h 时可能超过 100 分（显示上限 150，对数压缩、每快一倍 +10 分），**但比赛计分以题包公式为准** [C]（超 100 分的可能性作为信息保留，不作为目标）。
- 得分取整 floor；不满足约束的测试点直接 0 分；各测试点平均为该题总分；每题保留历史最佳 [A]。
- 主办方保留在分数分布异常时修改公式常数的权利（提前公告、全员统一重算）[A]。

**结论性解读 [A+B]**：性能目标本质是「在满足约束校验的前提下，把平均运行时间压向 T_h」。T_b 与 T_h 的绝对数值由平台给出（题目配置），**做题前必须从题目详情中拿到 T_b/T_h 或至少拿到 baseline 运行时间** [C]。

---

## 4. Baseline 分析

- **Baseline 定义 [A]**：baseline 为平台方的「baseline 参考实现」，T_b 为其平均运行时间，由平台在评测机上按同一流程测定。题包未给出 baseline 的具体实现（PyTorch 参考实现推测为朴素多 kernel 版本 [C]）。
- **Baseline 的分数锚点 [A]**：s = 0.5（默认）意味着「跑得与 baseline 一样快」得 50 分，「不满足约束」得 0 分，T_h 处 100 分。**任何正确的提交（哪怕慢于 baseline）都拿 0~50 分**——先保正确性、再谈性能。
- **Baseline 的定位 [B/C]**：baseline 参考实现同时是正确性校验对象（平台将参赛输出与 baseline 输出按题目容差比对）。推测为 torch 朴素实现（逐算子拼接），其访存效率远低于融合 kernel，因此 T_b ≫ T_h 是大概率情形——这也是优化空间（2~5×）的来源。T_b 具体实现与 T_h 的具体参数（peak_tflops/peak_bw 取值）[C]。
- **我们自建的本地参照 [B]**：本地以 PyTorch 的 BF16/FP32 混合精度朴素实现作为「参考实现」，计算 SQNR；若平台允许，把本地朴素的 run_kernel 提交一次作为自己的 T_b 实测锚点（消耗 1 次尝试，收益是拿到真实 T_b/T_h 口径下的校准，建议尽早做，见 workflow.md 第 0/5 步）。

---

## 5. 技术调研综述（2023 年至今，含可行性对比）

> 本节省略推导细节，保留「结论 + 来源 + 可行性 + 预期收益」四要素，供选型。每节末尾是可行性对比表。标 [C] 的均为「题包无法确认、以平台题目配置为准」的项。

### 5.1 赛题一：MegaMoE（Triton-distributed，4×H800）

**关键前提发现（调研结论 [B]，务必先读）**：
- 原 `github.com/openai/triton-distributed` 已不可访问（404，仓库已删除/转私有）；当前活跃的同名项目是 **`github.com/ByteDance-Seed/Triton-distributed`**（MIT，持续维护中），与主办方（管理员邮箱 @bytedance.com、平台 API 位于火山引擎网关）同源。题包要求的「Triton-distributed」按此理解，但**评测环境中的具体版本/分发方式 [C]**。
- 论文：*Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler*，arXiv:2504.19442；配套 *TileLink*（MLSys 2025），arXiv:2503.20313。
- 架构：基于 **NVSHMEM 对称内存**（非 NCCL）。核心抽象：
  - 通信原语：`triton.language.extra.libshmem_device` 暴露 `putmem/getmem`（含非阻塞变体）、`putmem_signal_nbi_block`、`signal_op`、`signal_wait_until`、`fence/quiet/barrier_all` 等；
  - 上下文：`rank()`、`num_ranks()`、`symm_at(ptr, rank)`、`wait()`、`notify()`、`consume_token()`；
  - 高层：`create_all_to_all_context` / `fast_all_to_all` / `all_to_all_post_process`（EP A2A 上下文）、EP A2A Layer / EP A2A Fused Layer / Low-Latency EP A2A Layer / TP MoE Layer 等。
- 安装形态 [B/C]：pip 包 `triton_dist`（v0.0.2，内嵌 Triton 3.4 分支，**安装需 `pip uninstall triton` 替换官方 Triton**），依赖 `nvshmem4py-cu12==0.1.2`、`nvidia-nvshmem-cu12==3.3.9`、`cuda.core==0.2.0`；官方推荐 NGC PyTorch 25.04 容器。**评测机环境若由平台预置则无此负担，本地必须复刻** [C]。
- 文档与示例（已核实可访问）：
  - 仓库：https://github.com/ByteDance-Seed/Triton-distributed
  - 文档站：https://triton-distributed.readthedocs.io/en/latest/
  - 原语清单：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/primitives.md
  - **EP MoE 推理完整示例**：`python/triton_dist/test/nvidia/test_ep_moe_inference.py`（与赛题流程同构：路由→fast_all_to_all→专家 group GEMM→A2A 回传→combine）
  - **MegaKernel 文档与实现**：`docs/getting-started/megakernel/megakernel.md`、`python/triton_dist/mega_triton_kernel/`（Qwen3-32B 8×H800 单步解码 3.33ms vs torch eager 26.08ms）
  - 计算/路由 kernel：`kernels/nvidia/group_gemm.py`、`swiglu.py`、`moe_utils.py`（histogram、scatter/gather index、per-expert 对齐 `ALIGNMENT_BY_EXPERT`、TMA top-k 归约）、`ep_all2all_fused.py`（官方称 megakernel with token optimization）、`all_to_all_vdev_2d_offset(_inter_node).py`（虚拟专家版）

**MoE 优化技术点与可行性对比表**：

| # | 方案 | 原理 | 来源（URL） | 本赛题可行性 | 预期收益 | 注意事项 |
|---|---|---|---|---|---|---|
| 1 | 复用官方 EP MoE 示例 | 官方示例 `test_ep_moe_inference.py` 流程（路由→A2A→group GEMM→combine）与赛题完全同构，直接移植改造 | github.com/ByteDance-Seed/Triton-distributed | **高** | 高（省去全部地基工作） | 需按赛题签名改写 run_kernel；确认评测环境版本 [C] |
| 2 | EP fused megakernel（通信-计算重叠） | `ep_all2all_fused.py`：把 dispatch/combine 的 A2A 藏进 GEMM 执行期，单 kernel 内完成 | 同上 kernels 目录 | 高 | 高（重叠是分布式题最大杠杆） | 工程复杂；先跑通非 fused 版本再上 |
| 3 | 稳定排序做路由索引 | 用 `argsort(stable=True).argsort()` 而非原子计数生成 scatter/gather index，保证**逐字节确定性** | 官方测试文件 + PyTorch 文档（pytorch.org/docs/stable/notes/randomness.html） | 高 | 正确性必需 | 原子计数（`tl.atomic_add`）在两次运行间顺序不保证一致→**会破坏赛题逐字节一致约束，禁用** |
| 4 | 无 padding 的分桶/对齐 group GEMM | MegaBlocks 思想：按专家分桶、per-expert 对齐（BLOCK_M/ALIGNMENT），避免 token drop 与 padding 浪费 | arXiv:2211.15841；官方 `group_gemm.py` | 中-高 | 中 | 直接引其 CUDA kernel 不可行（须用 Triton）；借鉴思想即可 |
| 5 | 虚拟专家（virtual experts）负载均衡 | 把负载不均的专家拆成虚拟专家，配合 vdev A2A kernel | 官方 `all_to_all_vdev_2d_offset.py`；Tutel arXiv:2206.03382 | 中 | 中（token 分布不均时） | 推理期 capacity factor/aux-loss 不适用；虚拟专家改变路由语义，需与参考实现语义一致 |
| 6 | 通信库对比参考 | DeepEP（V1 NVSHMEM / V2 NCCL Gin）的 dispatch/combine 与 EventOverlap | github.com/deepseek-ai/DeepEP；DeepSeek-V3 arXiv:2412.19437 | 中（仅参考） | 了解上限即可 | 赛题要求 Triton-distributed，不得直接引入 DeepEP |
| 7 | 数值约定 | gate/up 两 GEMM 合成一个 [E,2N,K] 输出再切分；`tl.dot` FP32 累加、BF16 存取；gate_weight 计算用 FP32 避免路由误选 | 官方 `group_gemm.py`/`swiglu.py` | 高 | 正确性必需 | SQNR≥22dB 在该模式下余量充足 |
| 8 | 负载不均兜底 | per-expert 对齐 + token 数 histogram 预分配 | 官方 `moe_utils.py` | 高 | 中 | 与确定性方案（3）联用 |

**赛题一风险评估 [B]**：
- Triton-distributed 处于 0.0.x 快速迭代期，API 可能随 commit 变化 → **锁定版本/commit**，本地与评测环境必须同版本 [C]；
- NVSHMEM put/get 对固定 (src,dst) 保序，但必须用 `putmem_signal + signal_wait_until` 显式同步；双缓冲（`call_count % 2`）为官方 in-flight 缓冲模式；
- 逐字节一致性风险点：原子计数路由索引（禁用）、combine 归约顺序（固定顺序串行累加）、浮点归约（固定 tiling 与编译参数后 `tl.dot` 确定）；
- 通信上限：4×H800 NVLink 单向 200 GB/s/卡，A2A 聚合上限约 800 GB/s（单向）[B]，若评测机为 NVSwitch 全互联拓扑则非瓶颈 [C]；
- 专家权重、hidden 维、E、topk、batch 等测试点配置 [C]，直接影响 tiling 与缓冲上限（官方示例 `MAX_M = 128*topk` 需按赛题最大规模调整）。

### 5.2 赛题二：MagiAttention（单卡 H800）

**关键前提发现（调研结论 [B]）**：赛题名有真实出处。**MagiAttention** 是 Sand.AI（视频生成世界模型 MAGI-1 团队）开源的分布式 attention 库，其核心单卡 kernel 为 **FFA（Flex-Flash-Attention）**：用 `AttnSlice = (QRange, KRange, MaskType)` 分解任意不规则 mask（FULL / CAUSAL / INV-CAUSAL / BI-CAUSAL），slice 级并行 + 多 slice 的 (O, LSE) 归约合并；v1.0.5 起支持 learnable attention sink（含与 FA2/FA3 的插件接口）。赛题描述的「slice 的 range+mask→可见 K；GQA 选 KV head；局部 attention+LSE；多 slice LSE 合并；sink logits 进分母重新归一化」与 FFA 前向**逐条对应**。赛题是单卡前向，FFA 的分布式部分（CP、通信、overlap 调度）不需要。

来源：
- 仓库：https://github.com/SandAI-org/MagiAttention
- 技术博客（AttnSlice/FFA 方法论）：https://SandAI-org.github.io/MagiAttention/docs/main/blog/magi_attn.html
- **Sink 数学推导（公式直接可用）**：https://sandai-org.github.io/MagiAttention/blog/ffa_with_sink.html
- MAGI-1 论文（引用 MagiAttention）：arXiv:2505.13211

**核心数学（正确性地基，[B]）**：
- Online softmax（LSE 形式）：逐 K 块 `m_new=max(m, max(qk))`，`l=l·exp(m−m_new)+Σexp(qk−m_new)`，`acc=acc·exp(m−m_new)+P·V`，`lse=m+log(l)`（FlashAttention-2，arXiv:2307.08691；原始 online softmax arXiv:1805.02867）；
- **多 slice 合并（精确公式）**：两结果 (O₁,LSE₁)、(O₂,LSE₂) 合并：`LSE=log(exp(LSE₁)+exp(LSE₂))`，`O=exp(LSE₁−LSE)·O₁+exp(LSE₂−LSE)·O₂`——数学上精确（FP32 下仅 ~1e-7 舍入），任意分组顺序两两合并等价；
- **Attention Sink（精确公式）**：sink 可视作每行 Q 额外拼接的 logits（无对应 V）。前向后处理：`lse_sink=logsumexp_j(sink_j)`（每 head 常数，可预热预计算）→ `lse'=log(exp(lse)+exp(lse_sink))` → `O'=O·exp(lse−lse')`。**所有统计量（lse/acc/scale）必须 FP32**；`lse−lse'≤0` 恒成立，无上溢风险。

**Attention 优化技术点与可行性对比表**：

| # | 方案 | 原理 | 来源（URL） | 本赛题可行性 | 预期收益 | 注意事项 |
|---|---|---|---|---|---|---|
| 1 | FA2 式 tiling（Triton） | Q 行分块、K/V 循环；FP32 累加；单 head 内跨 block 并行 K | arXiv:2307.08691；Triton 教程 `06-fused-attention.py`（triton-lang.org/main/getting-started/tutorials/06-fused-attention.html） | **高** | 高（约 FA2 水准，H100 上 50-70% MFU） | 开发最快、可 autotune；作为 v1 主力 |
| 2 | FA3 式（TMA+WGMMA+warp specialization+ping-pong） | Hopper 特性：异步 TMA 搬运、warpgroup 级矩阵指令、producer/consumer warp 分工、exp 藏进 GEMM 阴影 | arXiv:2407.08608；tridao.me/blog/2024/flash3/；flash-attn 仓库 `hopper/`（github.com/Dao-AILab/flash-attention，BF16 前向已发布） | 中 | 高（相对 FA2 再 +30-60%） | Triton 侧 `warp_specialize=True` 在 sm90 可用但约束多；工程量大 |
| 3 | mask 感知 tiling（静态 solver） | 预热期把 slice 配置编译成「每 (q-block,slice) 的 k-block 起止表」，运行时零分支；全不可见 slice 跳过 | FFA 博客（static attn solver）；对照 FlexAttention arXiv:2412.05496、FlashMask arXiv:2410.01359 | **高** | **高（本题最大性能杠杆）** | 与「预热预处理」规则完美契合；稀疏度越高收益越大 |
| 4 | slice 级并行 + LSE 归约 | 多 slice 各自算 (O,LSE)，atomic/确定性归约合并 | FFA 博客（AttnSlice-level parallelism）；FlashDecoding（flash-attn 2.2 起 flash_attn_with_kvcache） | 高 | 中-高 | 覆盖重叠 slice；确定性模式避免原子顺序问题 |
| 5 | auto_range_merge（slice 合并） | 合并相邻/重叠 slice，减少 kernel 启动与归约开销 | FFA 博客 v1.0.4 起 | 高 | 中 | 与 3 配合 |
| 6 | GQA KV 复用 | grid 按 (q-block, kv-head) 划分、kernel 内循环共享 KV head（KV 只读一次）；或按 `kv = q_head // ratio` 索引靠 L2 复用 | GQA arXiv:2305.13245；flash-attn 的 pack_gqa 参数 | 高 | 中（KV heads 少时明显） | KV head 数与 Q head 数的整除关系 [C] |
| 7 | split-KV（FlashInfer） | KV 序列切块并行 + 二次 LSE 合并 | arXiv:2501.01005；github.com/flashinfer-ai/flashinfer | 低（本题） | 低 | 为 decode 设计；本题 Q 长、slice 维度已提供并行度 |
| 8 | FlashDecoding++（unified max value） | 全局统一 max 免每块 rescale | arXiv:2311.01282 | **不适用** | — | **改变 Softmax 分母精确值，与赛题「精确 LSE+sink 重归一化」语义冲突，禁用** |
| 9 | 预热期常数/结构预计算 | `lse_sink`、mask 迭代表、autotune 结果缓存、CUDA Graph、归约 workspace 预分配 | FFA 博客；平台评测指南（预热阶段） | 高 | 高（零精度风险） | 受沙箱「1 个 jit + 1 个 autotune」限制 [C] |

**赛题二数值/工程风险 [B]**：
- 26 dB ≈ 噪声 rms ≤ 信号 rms 的 5%；BF16 输入量化本身 ~0.2%，只要中间量全 FP32，通常余量 10+ dB——**达标概率高，但任何 BF16 下的 exp/merge/scale 都会击穿**；
- exp 实现：`__expf` 快路径误差 ~2 ulp，一般不影响 26 dB，但与参考的差异需小规模实验确认；
- 确定性：两两 logsumexp 合并与一次性合并的差异 ~1e-7 级，对 BF16 输出无感；但 atomic 归约的浮点加法顺序不确定 → 复现性风险，优先确定性模式；
- 边界：某 Q 行可见 K 为空（lse=−inf）时 O=0 且须显式处理；sink 极大时真实权重被压小（绝对误差受 BF16 表示下限影响）；sink 极小影响趋零；
- 负载不均：slice 面积差异大导致部分 SM 空转 → 预处理期按面积排序/切分缓解；
- 「BGM / Block-Group-Masked」概念在公开文献中检索无果 [C]，按赛题语义自行实现即可，勿依赖未证实的论文。

### 5.3 赛题三：mHC（单卡 H800）

**关键前提发现（调研结论 [B]）**：
- 论文身份确认：**Hyper-Connections (HC)** = arXiv:2409.19606（ByteDance Seed，Defa Zhu 等）；**mHC (Manifold-Constrained Hyper-Connections)** = arXiv:2512.24880（DeepSeek-AI，Zhenda Xie 等，v2 2026-01-05）。与 YOCO/StreamingLLM 无直接关系（那是 KV 缓存路线）。
- mHC 确切公式（逐层、逐 token，论文 Eq.7–9；HTML 版：https://arxiv.org/html/2512.24880v2）：
  - `x_{l+1} = H^res·x_l + (H^post)ᵀ·F(H^pre·x_l)`；
  - 系数生成（逐 token）：先对展平的 nC 维残差做 RMSNorm → `H̃ = α·(x⃗'·φ) + b`；其中 φ^pre∈R^{nC×n}、φ^post∈R^{nC×n}、φ^res∈R^{nC×n²}（三者可合成**一个 GEMM**，输出 n²+2n 列）；
  - `H^pre = σ(H̃^pre)`（sigmoid）、`H^post = 2σ(H̃^post)`、`H^res = Sinkhorn-Knopp(H̃^res)`（exp + 交替行列归一，论文取 **20 次迭代**）；
  - 子层 F：赛题规定为「SiLU + 低秩 MLP」= `F(x)=Up(SiLU(Down(x)))`，即两个小 GEMM + SiLU 融合（论文本身未限定 F 结构，赛题做了具体化）。
- **系数是逐 token 动态生成**（依赖每个 token 自己的 nC 维残差向量），**不可静态预计算**；但无跨 token 依赖 → 系数生成可整批 GEMM，Sinkhorn 等逐 token 段可全并行。这是本算子最重要的结构事实。
- 计算特性（n=4、C=4096 量级估算）：整体**内存受限**——每 token I/O 约 72 KB（读改写 n 路残差），H800 上纯内存时间约 21 ns/token vs 全算力计算约 9 ns/token；I/O 是绝对主导。
- 论文自带工程方案（§4.3.1，可直接照搬）：φ 用 tf32、x 用 bf16、累加与系数全链路 fp32；**RMSNorm 重排**（先 GEMM 再除以 r=‖x⃗‖₂/√(nC)，数学等价）；**权重吸收**（α、RMSNorm 权重折入 φ，b 留 epilogue）；内核从 ~7 个拆成 3 个：①系数生成统一核 ②轻量系数核（σ/2σ/Sinkhorn 融合，避免 launch 开销）③F_post,res 融合核；论文用 TileLang 实现多数 kernel；n=4 时系统开销仅 +6.7% 训练耗时。

**mHC 优化技术点与可行性对比表**：

| # | 方案 | 原理 | 来源（URL） | 本赛题可行性 | 预期收益 | 注意事项 |
|---|---|---|---|---|---|---|
| 1 | 单 GEMM 生成全部三组系数 | φ^pre/post/res 拼成一个 [nC, n²+2n] 的 φ，一次 `tl.dot` 出全部原始系数 | mHC 论文 §4.3.1 | **高** | **高（结构性收益：x 只读一次）** | 输出列数少（n=4 时 24 列），tensor core 利用率依赖 T |
| 2 | RMSNorm 重排 + 权重吸收 | 归一化移到 GEMM 之后除；α 与 norm 权重折入 φ | 同上（Eq.10–16） | 高 | 中（省一个独立 norm 核） | 数学等价，零精度风险 |
| 3 | F_post,res 融合核（读写各一次） | H^res·x（逐 token 小矩阵×x）+ H^postᵀ⊗F 融合，x 读写从 3-4 次降为 1 次 | 同上 | 高 | **高（内存受限题的命门）** | n×n 小矩阵用元素级外积而非 batched GEMM（K=4 太小） |
| 4 | Sinkhorn 片上化 | 20 次迭代是逐 token 4×4 串行段，压进 warp 寄存器、与 σ/加 b/缩放融合，避免独立核 launch | mHC 论文；加速参考 arXiv:2606.07574 | 高 | 中-高（T 小场景显著） | **必须保留 exp+交替行列归一的确切迭代形式**（对拍），只做执行优化，不换 Newton/对偶法（数值不同） |
| 5 | 低秩两段 GEMM + SiLU epilogue | Down/Up 各一个常规张量核 GEMM，SiLU 在 epilogue 融合 | 通用（CUTLASS/Triton/TileLang）；参考 Liger-Kernel（github.com/linkedin/Liger-Kernel） | 高 | 取决于 r（r 大则 F 是 FLOP 主导） | r 值 [C]，直接决定策略 |
| 6 | persistent 单核（读写一次） | 读 x → 系数（φ 驻 L2）→ h0 → F → 混合 → 写 x'，x 只在片内流动 | FlashAttention 访存哲学 arXiv:2205.14135；mHC 论文 | 中-高 | 高（达到内存带宽 70-85% 时的最终形态） | T 小时延迟主导，必须单核消 launch |
| 7 | 预热期静态预处理 | φ 转置/打包/tf32 转换/吸收常数、b 布局、autotune、CUDA Graph | mHC 论文 §4.3.1；平台「预热预处理」规则 | 高 | 中（零精度风险） | 系数本身**不可**预计算（逐 token） |
| 8 | 线性注意力融合范式参考 | FLA（flash-linear-attention）的 RMSNorm+gate 融合模式与 mHC 系数路径同构 | github.com/fla-org/flash-linear-attention | 中（仅参考） | 低-中 | 直接抄思路（FusedRMSNormGated、RMSNormLinear） |

**赛题三风险 [B]**：
- **不要更换数学公式**：mHC-lite（arXiv:2601.05732，BvN 重参数化）、TBP-mHC、go-mHC 等是数学变体，若参考实现按 mHC 原式对拍，替换即判错；Sinkhorn 只做执行层优化；
- 对拍细节（任一不符即 SQNR 失败）：Sinkhorn 迭代次数（论文 20）、归一化顺序、σ 前是否先加 b、RMSNorm 的 eps [C]——严格按论文 Eq.7–9 实现后再优化；
- Sinkhorn 的 exp/归一必须 FP32（BF16 下严重掉精度）；
- 维度参数 n、C、rank r、T [C]：r 决定 F 的权重（r 大→F 走张量核；T 小→persistent 单核），需模板化自适配；
- 预期加速：相比朴素多核拆分（5-7 个核、x 读改写 3-4 次），融合后 2-4×；最终目标为内存带宽 70-85% 利用率。

### 5.4 通用工具链（2023 年至今）

| 工具 | 用途 | 来源 | 备注 |
| --- | --- | --- | --- |
| Triton | 张量核编程 DSL + autotune | github.com/triton-lang/triton；triton-lang.org | 赛题二三可用；沙箱限制 1 jit + 1 autotune [C] |
| TileLang | TVM 之上的 tile DSL，sm90 TMA+wgmma 支持 | github.com/tile-ai/tilelang | 赛题二三可用；mHC 论文官方工具 |
| CUDA（CUTLASS/CuTe） | 手写 FA3 级 kernel | github.com/NVIDIA/cutlass | 赛题二三可用；工程量最大、上限最高 |
| Triton-distributed | 分布式编译 + NVSHMEM 通信 | github.com/ByteDance-Seed/Triton-distributed | 赛题一强制 |
| Nsight Compute / Systems | 剖析 | developer.nvidia.com/nsight-compute | 本地剖析 |
| nccl-tests | 卡间带宽自测 | github.com/NVIDIA/nccl-tests | 赛题一本地验证 A2A 上限 |

---

## 6. 待确认事项清单（动手前必须逐项核实）

| # | 事项 | 影响 | 核实途径 |
| --- | --- | --- | --- |
| 1 | 三题的题目签名（run_kernel 参数顺序、张量形状语义）、测试点数量与维度（hidden/E/topk/seq/n/C/r/T 等） | 一切工程实现 | 报名后看比赛题目详情 [C] |
| 2 | T_b / T_h 的具体数值与 T_h 口径（peak_tflops/peak_bw 取值） | 目标设定 | 题目配置 + 提交一次朴素正确实现做校准 |
| 3 | baseline 参考实现形态（PyTorch 朴素？）与校验容差方式（allclose 参数 / SQNR 的精确计算口径） | 正确性验证 | 平台评测指南 /d/2 与题目详情 |
| 4 | 赛题一评测方式：4 卡如何启动、NVSHMEM 对称内存如何分配、评测环境中的 Triton-distributed 版本 | 赛题一全部工作 | 咨询主办方/微信群 |
| 5 | 赛题一是否同样允许预热预处理（题包只在赛题二、三写明） | 赛题一优化空间 | 咨询主办方 |
| 6 | 赛题二、三是否要求逐字节一致（题包只在赛题一写明） | 确定性实现取舍 | 咨询主办方（建议默认按确定性实现） |
| 7 | Triton/TileLang 沙箱细节（白名单、autotune 限制、预热阶段是否同样受限） | 提交方式 | 评测指南 /d/2、/d/3 |
| 8 | H800 显存带宽（3.35 vs 2.04 TB/s）与评测机 4 卡拓扑（NVSwitch 全互联？） | T_h 估算、通信上限 | 厂商 datasheet、主办方 |
| 9 | 评测机 CUDA 版本（<12.6）、PyTorch/Triton 版本 | 本地环境对齐 | 评测指南、实测 |
| 10 | 评分公式 T_k<T_h 区间的行为（是否可超 100 分、显示上限） | 极端目标 | 主办方（以题包公式为准） |

---

## 附：主要来源索引

**题包（[A]）**
- 比赛说明全文：https://xpuoj.com/d/31（本目录 `full.md` 即其抓取件）
- 比赛入口：https://xpuoj.com/contest/13
- 评测指南（题包引用）：https://xpuoj.com/d/2

**论文（[B]）**
- Triton-distributed: arXiv:2504.19442；TileLink: arXiv:2503.20313
- MegaBlocks: arXiv:2211.15841；GShard: arXiv:2006.16668；Switch: arXiv:2101.03961；Tutel: arXiv:2206.03382；DeepSeek-V3: arXiv:2412.19437
- FlashAttention: arXiv:2205.14135；FA2: arXiv:2307.08691；FA3: arXiv:2407.08608；Online softmax: arXiv:1805.02867；FlashDecoding++: arXiv:2311.01282；GQA: arXiv:2305.13245；StreamingLLM: arXiv:2309.17453；FlashInfer: arXiv:2501.01005；FlexAttention: arXiv:2412.05496；FlashMask: arXiv:2410.01359；MAGI-1: arXiv:2505.13211
- HC: arXiv:2409.19606；mHC: arXiv:2512.24880（HTML: arxiv.org/html/2512.24880v2）；Birkhoff 投影加速: arXiv:2606.07574；mHC-lite: arXiv:2601.05732

**开源仓库（[B]）**
- github.com/ByteDance-Seed/Triton-distributed（文档站 triton-distributed.readthedocs.io）
- github.com/SandAI-org/MagiAttention（博客 SandAI-org.github.io/MagiAttention/docs/main/blog/magi_attn.html、sandai-org.github.io/MagiAttention/blog/ffa_with_sink.html）
- github.com/Dao-AILab/flash-attention；github.com/flashinfer-ai/flashinfer；github.com/databricks/megablocks；github.com/deepseek-ai/DeepEP；github.com/tile-ai/tilelang；github.com/triton-lang/triton；github.com/NVIDIA/cutlass；github.com/NVIDIA/nccl-tests；github.com/fla-org/flash-linear-attention；github.com/linkedin/Liger-Kernel

**硬件（[B/C]）**
- nvidia.com/en-us/data-center/h100/；techpowerup.com/gpu-specs/h800-sxm5.c3975；cputronic.com/en/gpu/nvidia-h800-sxm5；lenovopress.lenovo.com/lp1814.pdf；nvidia.com/en-us/data-center/nvlink/
