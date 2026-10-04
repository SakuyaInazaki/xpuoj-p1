# P1 MegaMoE：接手审计与后续优化指引

> 历史版本：本文绑定 v842。当天后续已完成 v843/v844 及多项新实验；当前方向、评分和任务分派请改读 [v844 优化任务书](OPTIMIZATION_GUIDE_V844_2026-09-27.md)。不要再次直接执行本文第 11 节的旧任务。

日期：2026-09-27（UTC+8）。适用目录：`/Users/sakimi/Desktop/xpuoj-p1`。

**建议先交给一个 agent 实现“c4 padded 输出直接逆索引”，同时由另一个 agent 核实调用路径与编译工作集。第二条性能路线是稳定计数排序的算法改写。停止无目标的 GEMM 参数扫描。** 当前没有证据支持这些小范围改动能稳定把 P1 提到 75；应以可复验的端到端收益决定是否继续。

本轮交付是优化决策与执行说明：读取题面、源码、历史实验和新增官方答复，实时只读核验平台，重建评分模型，并验证候选索引的 CPU 性质。没有进行新的 GPU benchmark、正式提交或生产内核晋升。`p1/kernel.py` 保持不变。

标记约定：**事实**有代码、原始结果或官方说明支持；**推断**是从这些证据作出的判断；**假设**必须由下面的实验验证。历史“关闭”只约束当时机制和环境，不能代替本轮成本分析。

## 1. 先锁定真实起点

| 项目 | 本轮确认的状态 |
|---|---|
| 生产文件 | `p1/kernel.py`，与 `p1/kernel_v842_sort_nw4_c1112.py` 相同 |
| SHA-256 | `5cfd6a80d066d8a729ff1b6741541c1c70df7df75eeffe40c67440d4fe962a3c` |
| 当前 P1 榜分 | **73.50**，best SID **146307**，累计正式提交 **3260** |
| 本账号总排名 | 第 **4**，三题合计约 **241.16**；这不是 P1 单题名次 |
| 最新 P1 提交 | SID **146915**，2026-09-20，Accepted，display 81.67 |
| 可用资源 | 用户确认只有比赛正式评测和单卡 custom；本轮查询 `triton-dist` custom 为 `available=false` |
| P1 软件 | 官方补充答复：Triton-distributed **3.4**、Triton **3.4**；确切 fork commit/镜像 digest 仍未知 |
| P1 硬件 | 官方补充答复：4×H800 80G SXM，默认 700 W，各卡相互连接显示 NV8 |
| 超时 | 官方说明是一次 torchrun 批量执行的 **500 秒总预算**，不是“每个新 kernel 有 500 秒编译时间” |

平台核验时间为 2026-09-27 02:57 左右，结果见 [平台快照](reports/2026-09-27-platform-readonly.json)。仅查询了最近五项，不把该查询包装成所有历史作业的全量队列审计。开工提交前仍由唯一平台执行者检查当时状态。

依据文件：

- [本轮基线 manifest](reports/2026-09-27-baseline-manifest.json)：哈希、当前函数行号、重复定义和 12 案几何。
- [新增官方答复快照](reports/2026-09-27-addinfo-snapshot.md)：来自用户提供的 `/Users/sakimi/Desktop/addinfo/addinfo.md`，保留原有相对日期，不擅自还原发帖绝对日期。
- [最新与最佳提交逐案数据](reports/2026-09-27-latest-cases.json)：本轮只读补取 SID 146307、146910–146915。
- [题面原文](1-full.md)、[提交入口说明](docs/SUBMISSION.md)。公网网页工具没有取得可读题面正文，本轮题意依据本地题面副本及用户补充材料。

### 必须修正的旧认识

1. `docs/STATE.md` 页首的 v836、`CODE_MAP.md` 的旧 SHA/行号不能当当前状态；STATE 后半已追加到 v842。部分 SUMMARY 也在同一文件前后保留不同阶段的“当前”。以文件哈希为准。
2. “samples[0] TLE ⇒ 新 JIT kernel 编译不可行”没有充分证据。总预算可能包含初始化、多案编译、预处理和运行；目前日志不足以精确归因。明确的编译断言与单纯 TLE 必须分开。
3. “只计 MD/DN，route/sort/quant/fin 都不计时”不能继续使用。历史 D2H 探针和后来的辅助阶段改进与这一绝对说法冲突。优化一律核算完整链。
4. “两次日志 SQNR 相同 ⇒ 两份输出逐位相同”不成立。日志通常只显示两位小数；跨候选 bitwise 比较需要单独 oracle。
5. 100 分不是比赛通用硬上限。官方说低于目标上界时间会进入对数分段。本文当前 P1 模型只使用已经观察到的 `th=0` 区间。
6. “全部参数方向永久封闭”过强。部分末轮实验只留了提交号。本轮已恢复最新六次结果，但样本仍不足以判断小效应；这不构成再扫几十发的理由。

## 2. 算子语义与当前实现

### 2.1 正确性契约

每个 rank 输入本地 `X[T,H]`、共享 router `Wgate[E,H]` 和本地四分之一专家的 gate/up/down 权重。路由选 k 个全局专家，专家执行 gate/up → SwiGLU × routing weight → down，最后把 k 条分支按原 token 合并到 BF16 `output[T,H]`。

需要保留题面规定的参考语义：路由 logits 的 BF16 舍入、FP32 softmax/top-k/归一化，专家投影与中间值的规定舍入，以及最终 FP32 分支累加。允许混合精度逼近不等于可以漏算任意分支。验收至少包括 SQNR≥22 dB、输出全部有限且完整写入、输入不变，以及相同输入独立调用的逐字节确定性。

静态权重在**同一测试点内**允许预热预处理；切换测试点可能变化。动态输入、route、activation、输出不能依据调用次数或某次输入缓存后复用。

### 2.2 12 案与实际工作量

以下 case 编号来自本项目提交日志，不等于题面“示例测试用例”的编号。`m_rep=T·k/E` 是现役 replicated 路径在全部 E 个专家上的精确平均本地分支行数；标准 EP 汇总四卡后，全部专家的平均值是 `4m_rep`。把平均值代入某个专家或某个 rank 的负载模型才需要均匀分布假设，真实 counts 每次由路由决定。

| case | T | H | E | I | k | m_rep | 现役主特点 |
|---|---:|---:|---:|---:|---:|---:|---|
| c1 | 16384 | 4096 | 8 | 8192 | 2 | 4096 | 大 GEMM，sorted FP8 A，TMA A/B |
| c2 | 16384 | 4096 | 8 | 14336 | 2 | 4096 | 最高绝对计算量；大 GEMM |
| c3 | 16384 | 2048 | 32 | 2048 | 4 | 2048 | token FP8 + ORDER gather，融合 MD |
| c4 | 16384 | 2048 | 32 | 1024 | 4 | 2048 | 短 K DN；输出 TMA 优先验证对象 |
| c5 | 8192 | 3584 | 64 | 2560 | 8 | 1024 | 不规则 H，k=8，MD 较重 |
| c6 | 8192 | 3584 | 64 | 1024 | 8 | 1024 | DN 已使用 GM8/s3 |
| c7 | 16384 | 4096 | 96 | 2048 | 3 | 512 | E_PAD=128，token gather |
| c8 | 16384 | 4096 | 96 | 1024 | 3 | 512 | MD 已使用 s4/r232 |
| c9 | 4096 | 4096 | 256 | 2048 | 8 | 128 | 低内存、tile 连续权重；padding/权重流量重要 |
| c10 | 4096 | 4096 | 256 | 1536 | 8 | 128 | 同 c9；不能仅因总耗时小就忽视得分敏感度 |
| c11 | 65536 | 1024 | 32 | 1024 | 2 | 4096 | 大 T、短 H；排序/输入/输出带宽占比更高 |
| c12 | 65536 | 1024 | 32 | 2048 | 2 | 4096 | 同 c11；距下一档较近 |

12 案现役均分发到 replicated 家族：预热复制全专家权重后，每卡只处理自己的 token，稳态避免 token 的跨卡派发/回传。**这不意味着在四卡上重复计算同一份全部 token**。典型稳态链为：

```text
静态权重：跨卡收集 → FP8/布局转换 → 缓存
动态输入：route → stable sort / metadata → 输入 FP8
        → MD：gate/up + SwiGLU + 路由权重 + activation FP8
        → DN：FP8 GEMM + 分块输出量化
        → inverse gather + FP32 分支求和 → BF16 output
```

MD 是 gate/up 与激活一侧的项目简称；DN 是 down projection。代码里带 `int` / `intq` 的函数名不保证使用 INT8 Tensor Core，必须看 `tl.dot` 两个操作数的实际 dtype。

当前文件有 170 个顶层函数/类定义，按最后绑定计 160 个，其中 80 个带 JIT 装饰器；10 个名字重复定义。**定义个数不是实际编译特化个数**。当前入口见 `run_kernel:6472`、`_run_replicated:6024`；这些行号只适用于上述 SHA。

### 2.3 正确性与编译的共同前置问题

当前 `_CALLN`、`_GA`、`_FL` 影响 route、排序、量化和主 GEMM 路径。部分优化只在 calls 3–5 启用，其他调用走另一条数学路径。历史 AC 并不直接证明任意调用顺序下的所有路径都被同一个 oracle 验证。

后续基线应满足：**动态计算由输入的公开 shape/dtype/stride 和已验证的配置决定；不同调用可因静态缓存命中省去预处理，但计算数学不应依赖“这是第几次”。** 第 1/2/3/5/6/8 次与 interleave shape 后的调用，都要有独立正确性覆盖。不要把扩展一个调用窗口当算法改进。

多个权重缓存仅使用 shape 作为 key。同 shape 换权重对象就可能错误命中；只补 `id()` 也不能处理同对象原位改写或释放后的 id 复用。改造时记录设备、dtype、stride、标量和输入对象身份，并持有正确生命周期引用；对原位改写仍需要官方边界/版本约定或能够证明正确的失效方案。小 fingerprint 不是全量内容一致的证明，绕过被禁止的 `_version` 也不是方案。

这一前置工作保护可复现性，也可能减少 warmup、正确性与稳态分别触发的 JIT 特化。它本身不计入稳态提分承诺。

## 3. 用积分门槛分配时间

### 3.1 已验证评分区间

本轮在原始/恢复数据中核对：已检视的 P1 计时记录 `th_time_ms=0`，其性能积分与下式一致：

\[
q_i=\left\lfloor\frac{100\,t_{b,i}}{t_{b,i}+t_{k,i}}\right\rfloor,
\qquad S_{raw}=\frac{\sum_{i=1}^{12}q_i}{12},
\qquad S_{display}=\operatorname{round}(S_{raw},2).
\]

本账号已达每次新提交扣 10 分的上限，历史最高分保留。因此新提交想达到榜分 75，必须有 `Σq_i≥1020`；77 需要 1044；80 需要 1080。旧免罚次数材料有 100/200 两种口径，但对已经 3260 发的本账号，这一差异不改变当前扣分上限。

单案加 1 个整数分只增加整题 **1/12≈0.0833 分**。不能把“c4 快 1%”写成“P1 加 1 分”。对固定 tb，达到单案 q 分的门槛为：

\[
t_{k,i}\le t_{b,i}\left(100/q-1\right).
\]

取整前，每省 1 ms 的整题边际价值是 `100*tb/(tb+tk)^2/12`。这解释为什么 c2 的绝对时间最高，却不一定比 c4/c11 更值得为一个小幅优化投入时间。

### 3.2 v842 的代表性阈值

使用 10 份精确 v842 记录的逐案 tk 均值，以及邻近 20 份记录的逐案 tb 中位数建立条件模型。总 tk 均值 **29.6851 ms**。这不是今日新跑出的性能，也没有把跨机器差异校正成一个虚构精确值。

| case | tk均值 ms | tb中位数 ms | 模型当前分 | 本轮观察目标 | 所需再降 ms | 所需再降比例 |
|---|---:|---:|---:|---:|---:|---:|
| c1 | 4.6701 | 18.0045 | 79 | 80 | 0.1690 | 3.62% |
| c2 | 7.9478 | 28.8465 | 78 | 79 | 0.2797 | 3.52% |
| c3 | 1.4074 | 7.0345 | 83 | 84 | 0.0675 | 4.80% |
| c4 | 0.8352 | 5.0165 | 85 | 86 | 0.0186 | 2.22% |
| c5 | 2.8495 | 12.6780 | 81 | 82 | 0.0665 | 2.33% |
| c6 | 1.3283 | 7.6030 | 85 | 86 | 0.0906 | 6.82% |
| c7 | 2.3057 | 10.2660 | 81 | 82 | 0.0522 | 2.26% |
| c8 | 1.3489 | 7.1585 | 84 | 稳住84 | 已在中位门槛内 | 真实10发仅6发达标 |
| c9 | 2.5100 | 8.3935 | 76 | 77 | 0.0029 | 0.11%，小于噪声 |
| c10 | 1.9461 | 6.6830 | 77 | 78 | 0.0612 | 3.14% |
| c11 | 0.9653 | 5.6475 | 85 | 86 | 0.0459 | 4.76% |
| c12 | 1.5708 | 8.1530 | 83 | 84 | 0.0178 | 1.14% |

c8 真正的下一档 85 在该条件模型中需要再降约 6.35%，并非零成本。表中 c8/c9 是稳定性机会，不是“捡分已成功”。每次提交的 tb 会变，表中门槛不能作为未来精确承诺。详细样本、分位情景和原始越线比例见 [评分审计](reports/2026-09-27-score-audit.md)、[阈值 CSV](reports/2026-09-27-score-thresholds.csv)。

在固定中位 tb 下，当前模型整数分和是 **977**（display 81.42，罚后 71.42）；真实历史最佳 73.50 对应的提交有较高 tb。两者不是同一指标。固定 tb 且假设所有 case 同比例降时，首次达到榜分 75 需约 **21.95%** 降时，77 约 **34.99%**；这只是一个可复算情景，不是必要条件，更不是物理不可能证明。

因此目标分成两层：近期争取能稳定复现、能跨一个或几个台阶的结构收益；想拿回数个榜分，需要多个案的大幅改进或新的技术条件。历史六项 v837–v842 微调合计约 0.065 ms，不足以支持“大量微调可以追回几分”的计划。

## 4. 优先级与依赖

| 优先级 | 任务 | 性质 | 首轮范围 | 决策目的 |
|---|---|---|---|---|
| P0，并行进行 | 统一数学路径、缓存边界、实际编译工作集审计 | 基础工作 | v842，不改数学 | 让后续结果可解释，减少不必要特化 |
| P1 | padded DN 输出 + 直接 `INV_PAD` | 有历史正信号的结构假设 | **仅 c4** | 消掉原 padded 方案的准备与索引成本 |
| P2 | 稳定排序局部排名算法改写 | 尚待验证的算法假设 | E=32/96/256 | 消掉 dense one-hot scan 的成本 |
| P3 | offsets 与 MoE metadata 合并 | 上限较小的辅助方向 | 先 c11/c12 | 同一 counts 前缀只计算一次；支持 P1 的布局 |
| 条件重启 | 少量仅因总 TLE 未取数的候选 | 有明确机制才准入 | 一个几何、一个候选 | 纠正“超时=慢”的旧误判 |

P1 可以从冻结 v842 提取最小单卡子链，与 P0 并行。正式集成和晋升必须同时满足路径、正确性和端到端验收。P2/P3 先独立对照，之后再合并；不能把三项一起写完再猜哪项起效。

## 5. 执行卡 A：c4 的直接 padded inverse 与 TMA 输出

### A1. 为什么优先做

**事实：** [09-13 SUMMARY](experiments/2026-09-13/SUMMARY.md) 的 padded TMA 实验通过了 FP8、scale、BF16 final、数值零和 PAD_ROW 精确检查。初测 DN-only 快 2.80–3.18%；[六组确认原始结果](experiments/2026-09-13/results/public24_f5c14b6e-30cf-4e52-8c94-cf7bdb1a5595_parsed.json) 六组均快，范围 2.52–5.52%，ratio of means 对应降时 3.98%。但是独立 PAD_ROW 构建 + DN + mapped gather 总体 ratio of means 为 **1.000875**，没有完整收益。

**新假设：** 原实现最终 gather 先查 `INV`，再查 `PAD_ROW` 才读 Down；如果已有阶段直接生成 `INV_PAD[原分支]=padded行号`，Down 和它的 scale 都按 padded 行存储，就可省掉独立构建 kernel 和依赖式二级索引。此假设改变了成本来源，不是重跑原失败结构。

**先确认版本差异：** 旧正信号的 DN 使用 stages=4，当前 c4 已 stages=3。第一步先在当前 s3 下核实 TMA 本体信号是否还存在；不能把旧 +3% 原样加入现底盘。

### A2. 数学映射

对专家 e，设实际分支数为 `n[e]`，紧凑 row 起点为 `r[e]=Σ(j<e)n[j]`，BM=128：

```text
tiles[e] = ceil(n[e] / 128)
p[e]     = 128 * sum(j < e, tiles[j])
delta[e] = p[e] - r[e]

ORDER[pos] = 原始分支号 b
INV[b]     = pos
INV_PAD[b] = pos + delta[e]     # e 是该分支的专家
```

保留 `ORDER` 和原 `INV`，供现有输入量化/MD 使用。新增 `INV_PAD` 只供最终归并。

DN 保持从紧凑 ACT/ACT_SCALE 读取。其输出改为：

```text
Down[p[e] + local_row, h] = 原 Down[r[e] + local_row, h]
DSCL[p[e] + local_row, chunk] = 原 DSCL[r[e] + local_row, chunk]
```

final 按原分支 j=0..k-1 顺序，读取 `INV_PAD[t*k+j]` 对应的 Down/DSCL 并累加。归约顺序保持不变，不引入浮点 atomic。

### A3. 两种产生 INV_PAD 的实现，先选一个

**最小原型：在现有 MD epilogue 中附带写逆索引。**

当前 metadata 已给出每个专家的 tile 数与累积 tile 数。swizzle 后的实际 `local_m`、列 tile `pid_n` 和行 lane 确定 padded 行：

```text
padded_row = (t_cum - t_num + local_m) * 128 + lane
if pid_n == 0:
    INV_PAD[ORDER[offs_m]] = padded_row    # 只写真实 row
```

必须使用 **swizzle 后** 的列 tile 0，保证每条有效分支只有一个 writer。它新增约 M 个整数写入；若编译器把条件/寄存器压力扩散到整个 MD 导致回退，停止这一载体，不以提高寄存器帽掩盖问题。

**不能遗漏出表路径：** c4 的 calls 3–5 使用 `_fgs_t1i_mdq_kernel_g`，call≥6 切到 `_fgs_t1i_mdq_tma_kernel`，call1/2 还有旧路径。仅修改 `_g` 却对全部 c4 调用切换 padded DN/fin，会读取未初始化的 INV_PAD。原型的生产者、DN、fin 必须共享一个明确的 `use_padded` 条件；完整实现应在统一数学路径后全部覆盖，或由所有调用都经过的 sort 路径统一产生表。不能只在猜测的计时窗口出表。

**后续集成载体：在 sort 的 offsets/scatter 中生成。**

`_csort_offsets_kernel:733` 已加载 TOT，能同时生成 `delta[e]`。`_sort_scatter_kernel:694` 已知道 id、稳定 pos、原分支号，顺手写 `INV_PAD[offs]=pos+delta[id]`。这样 MD 不变，但会增加排序阶段前缀/读写，必须计费。两种载体都要与原完整链比较，不能只报告 DN。

### A4. 必须处理的接口与容量问题

- `_gather_branch_sum_f8:4715` 当前从 `down.shape[0]//k` 推导 T。Down padded 后必须从 `output.shape[0]` 或显式原 T 获得 T，否则会多写/越界。GPU fin 算法可复用，host 不能照抄。
- CSCL 也要 padded，且 DSCL 地址仍是 `padded_row*NCHUNK+chunk`。只 pad Down、不 pad scale 会迫使 fin 保留两套索引。
- 容量可用 `128*(ceil(M/128)+E)` 的安全上界，避免每次 `.item()` 同步读取 GPU 的 num_tiles。精确需求为 `128*Σceil(n[e]/128)`；安全上界不要求均匀路由。
- 空专家不产生 tile；尾部无效 row 只能写入其独占 padding，不得覆盖相邻专家真实行。
- 跨 SM 的 TMA store、barrier/wait 使用必须由当前 Triton 3.4 能正确生成的实现保证。保留已有成功 padded 候选的 flatten/存储结构，先做最小差异。
- c4 下额外 padded Down 容量量级至多约 8 MiB，加少量 scales/逆索引；仍要测峰值存储，不能把 arena 分配错误算成 GEMM 慢。

### A5. 验证、预算和停止条件

参考实现：[padded custom 确认版](experiments/2026-09-13/candidates/fp8_c4_dn_tma_padded_custom_bench_confirm.py)。本轮提供 [CPU 映射性质检查](reports/2026-09-27-padded-map-proof.py) 及 [结果](reports/2026-09-27-padded-map-proof-results.json)。CPU 检查只证明索引关系，不能证明 GPU 并发、TMA 同步、舍入或性能。

本轮重跑 **97 组、939,997 行全部通过**，包括全部 12 案的行数、空专家、边界和偏斜分配，并捕获 swizzle 前错误设置单写 guard 的反例。随机部分允许同 token 重复专家，属于对真实 router 的更宽索引输入测试，不是执行了真实 GPU router。

最小实验顺序：

1. CPU 检查空专家、1/127/128/129、尾 chunk、偏斜 counts、随机合法 top-k，验证唯一写入、容量、逆映射和分支顺序。
2. 单卡 custom：当前 s3 baseline 对旧 padded s3，确认旧机制还有收益；结果明显负则停。
3. 单卡 custom：baseline 对直接 INV_PAD 方案，**计入新增索引生成、MD、DN、fin**。输入、路由和 quant 如两边相同也应在完整 P1 验收时包含。
4. 对所有**真实 row**，按紧凑→padded 映射全量比较 FP8 q 和 scale；最终 BF16 全量比较。不同 shape 的整块 scratch 不能直接比较，未使用 padding holes 只检查永不被读取，不要求额外初始化。要求本轮纯布局改动 exact，通过零行/finite/确定性检查，再测随机动态输入和偏斜路由。
5. 完整 P1 只让 c4 使用候选，其他 case 固定作控制；同时运行 sample/fallback。成功后才单独验证 c6、c11、c12，不能直接照搬比例。

建议首轮预算为 2–3 个有明确区分目的的 custom；通过后最多 3 组紧邻候选/锚初筛（6 次正式评测）。达到实用幅度才追加固定确认批，避免为 ±0.1% 继续投几十对。

**继续条件：** 完整被触碰子链稳定快，且端到端预估收益大于当时测量噪声；争取 c4 全案至少 0.5–1%，再按收益与成本决定确认。即使 DN 快 3%，如果 DN 只占全案比例 f，上限也只有约 `3%*f`，还要扣新增流量；不要承诺跨过 c4 所需的 2.22%。

**停止条件：** s3 下本体信号消失；原型 exact/确定性不能解释地失败；映射加入后全链中性/变慢；或只能靠增设独立 PAD_ROW 才正确。输出“机制关闭、哪项税抵消收益”，不要继续扫 stages/store 配置。

## 6. 执行卡 B：稳定排序的局部排名改写

### B1. 当前成本来自哪里

现役 `_sort_hist_kernel` 和 `_sort_scatter_kernel` 构造 `[BLOCK,E_PAD]` 的专家相等矩阵，scatter 再沿 token 轴 `cumsum`，得到稳定 local rank。E=256 时每个 chunk 是 64×256；E=96 使用 E_PAD=128。参数调优已经很充分，但这不证明 dense one-hot 局部排名是最优算法。

**首个假设：** 对 BLOCK64 的一维 `(expert_id, local_index)` packed key 做局部排序，再计算段内排名，可替换 scatter 的二维等值矩阵。保留已有 hist/colscan/offsets，通过确定的 prefix 计算全局位置，不采用 CTA 抢号式 atomic append。首先只动 scatter，别把 hist 改写一起混入。

### B2. 需要维持的语义

对 chunk c 中 lane/元素 l 的 id=e：

```text
pos = expert_prefix[e]
    + sum(hist[earlier_chunk, e])
    + count(same_chunk positions < l with id == e)
```

建议先选 c9/c10、BLOCK64 的最小实现：

```text
key[l] = (id[l] << 6) | l     # 有效元素；无效尾元素设为大 sentinel
s = ascending_sort(key)
e = s >> 6; original_lane = s & 63
segment_start = 当前 sorted_position，当它是该 expert 的首项；否则 0
last_start = inclusive_max_scan(segment_start)
local_rank = sorted_position - last_start
pos = BASE[e, chunk] + local_rank
ORDER[pos] = chunk*64 + original_lane
INV[chunk*64 + original_lane] = pos
```

低位原位置使同专家 key 按输入次序排列；每个 chunk 的 BASE 保留跨 chunk 稳定性。无效 sentinel 对应的 BASE load 和两项 store 都必须 mask，不能用无效 expert id 读 BASE。BLOCK 改成 128/256 时相应改 shift 位数，不沿用常数 6。

官方 [Triton 3.4 standard.py](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/language/standard.py) 有 sort 与基于 associative scan 的实现；仍需对当前 fork 实测编译、layout 和速度。bitonic 网络的 shuffle 可能吃掉全部节省，指令阶数减少不保证更快。

先不手写 `match.any`/ballot：物理 lane 与 Triton tensor layout、跨 warp 及每线程多元素的顺序还要另证。若 packed-key 因可解释的 shuffle/layout 成本受阻，而上界仍足够，才允许一次这类不同机制的可行性实验。hist 的 `tl.histogram` 改写也留到下一独立步骤，不并入首版。

### B3. 最小实验和停止条件

- 第一个 custom 先测现役 **hist+colscan+offsets+scatter** 整段的成本，建立可省时间的上限；不能拿 scatter-only 快 2 倍直接预测整题快 2 倍。
- oracle 是全量 ORDER、INV、COUNTS 与稳定排序基准一致，且 `ORDER[INV[b]]=b`。覆盖 E=8/32/64/96/256、无效尾元素、所有 id 相同、空专家、严重偏斜、随机合法 top-k。
- 分别报告 E=32/96/256 的资源、总排序时间、变化的 scratch 字节和 launch 数。先挑一个几何进入正式 P1。
- 只有排序整段至少出现清晰改进，并足以贡献所选 case 约 0.5% 以上端到端收益，才进入正式筛选；这是预算门，不是物理门。
- 若收益上界只有 1–2 μs 且当前 P1 噪声更大，或者新的跨 warp scan 抵消节省，停止独立提分线。它可作为 A 的索引载体，但必须说明其工程作用。

历史检索未找到同一局部排名机制的完整负结果；这仅意味着“在审计范围内未发现”，不宣称从未尝试。开始写之前对历史候选再搜 `match.any`、`ballot`、`radix`、`local rank`，相同机制有可靠负证据时回到裁决。

## 7. 执行卡 C：融合 offsets 与 metadata

这条路线的预期上限比 A/B 小，但能复用同一 counts 前缀，并为 INV_PAD 提供布局信息。

当前 `_csort_offsets_kernel` 从 TOT 得到 expert row prefix，另一个 `build_block_row_idx_info_kernel` 又从 counts 求 row/tile prefix 并写 GEMM metadata。最终生效的 host 在 `_prepare_moe_metadata:4118`。

可测试的融合方案：offsets 的每个 expert CTA 读取完整 TOT，算 `row_start`、`tile_start`、`tile_count`；该 CTA 独占写其 expert 区间内的所有 tile metadata，并由指定 CTA 写 `num_tiles_total`。不要让同一 launch 的一个 CTA 写 counts、另一个 CTA 未同步就读取；都直接读取前一 launch 已完成的 TOT。

必须保持字段语义：每个 tile 的 expert id、该 expert 的紧凑 row 起点、该 expert tile 总数、inclusive tile prefix；不能把 `row_offset` 误写成 tile 自身 row 起点。极端 skew 时用有界循环写 tile 列表，不能假设每 expert 平均 512/2048 行。容量按 worst case 分配。

最小验证先逐整数比较原 metadata 六项返回值：per-expert 项比较 `[:E]`，四张 tile 表只比较 `[:num_tiles_total]`，总数标量直接比较；capacity 尾部未初始化，不能要求全 buffer exact。再对完整 sort+metadata、最后全案做测试。只减少一个 launch 不足以晋升；要包含增加的 per-expert prefix 运算和资源影响。建议 1–2 次 custom 的预算上限，无实测正信号就降为 A 的支撑改动。不得把同时融合 INV_PAD 与 metadata 的收益分开重复计入总收益。

## 8. 执行卡 D：统一路径与减少实际 JIT 特化

### D1. 目标和边界

目的：得到任意调用序列均正确的基线，以及可容纳一次新结构实验的总运行预算。**不要把“大文件删成小文件”当作已知加速机制。** JIT 延迟编译，未调用定义可能没有编译成本；而删代码改变有效定义和起始行，可能使缓存失效。

官方 [Triton v3.4.0 jit.py](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/runtime/jit.py) 的 cache key 包含源码依赖信息及起始行号；运行特化还与参数类型、constexpr、对齐和选项有关。线上 fork 细节仍须区别，不能从本地文件直接断言平台缓存行为。

### D2. 推荐拆成两步

1. **语义路径清单。** 从最后绑定定义出发，列 12 案和 sample/fallback 在各次调用时实际调用的 kernel、dtype、布局、constexpr 与 launch 参数；辨认哪些只为旧 warmup 路径触发。保留静态预处理缓存，但让同几何的动态数学路径统一。这个改动先在单卡抽取算子上直接对参考验证，不是一把把 `_CALLN` 条件改真。
2. **编译工作集控制。** 统计实际会执行的 `(kernel源码/依赖, dtype, constexpr, options)`，裁掉确实不再可达的旧数学路径。尽量一次只改变一个 kernel 家族；过渡稿可保留其余有效 JIT 的原文和行位置以减少缓存扰动，但最终交付不能依赖评测器恰好保留旧缓存才完成。

保留一条正确的未知 shape fallback；不要把 `_KNOWN12` 当可忽略其它输入的理由。`torch` 高层矩阵接口/官方 MoE 禁止调用的限制仍适用，fallback 必须遵守现有允许 API。

### D3. 验证

- 对相同输入执行 ≥8 次，并交错不同 shape；逐次比较独立参考及同输入 byte determinism。
- 对同 shape 的新权重对象、新动态 X、新 topk 组合做缓存失效测试。原位变更的契约仍未获明确答复，需单列，不以 identity key 宣称完全解决。
- 验证动态 route、activation 和输出没有跨调用复用；无跳过真实计算。
- 在兼容 Linux/CUDA 编译环境可用时，可按官方答复以 `triton.compile` 做无对应 GPU 的编译诊断；本机 macOS 没有可直接替代 H800 执行的环境。当前资源只有平台，不为实现此建议擅自升级依赖或租赁设备。
- 单卡 custom 如版本不同，只能支持数学/局部筛选。明确记录版本差，不能用 3.6 编译成功承诺 P1 3.4 可用。
- 首次完整 P1 仍因 500 秒超时且没有新诊断时，停止同码重投。记录原始失败位置，向主办方索取分阶段/PTXAS 日志；不能把排队耗时当编译耗时。

P0/D 的产出应是独立候选、路径表、缓存生命周期说明、特化清单和 correctness 证据。稳态 tk 与 v842 分开验证；允许“更正确、更易复现但暂未提分”的结果，不能擅自覆盖生产基线。

## 9. 当前不要重开的路线

详见 [历史审计](reports/2026-09-27-history-audit.md)。这里列下一轮最容易浪费预算的方向。

| 路线 | 关键负证据 | 只有什么变化才值得重启 |
|---|---|---|
| c9/c10 改 EP/TP | 最新 EP rank0 阶段读数即使 zero skew 也约 3.007/2.578 ms，对 replicated 2.483/1.932；TP 多方案也慢 | 通信量、重叠或存储机制实质变化，完整模型能追回差距；仅确认 NV8 不够 |
| FP16 accumulator + BM256/BN512 | 历史隔离微基准中 FP16 累加版吞吐约为 FP32 累加版的 0.80；P1 大 tile 候选慢约 23–50%，另有不同环境的单卡 BM256 负结果。lowering 开销是历史机制解释，不是只凭整链比值证明的唯一原因 | 同栈微基准证明该开销已消除，再谈大 tile |
| packed FP6/INT6/INT4 解码 | 字节减少被 decode、spill 和旧 MMA 吞吐抵消 | 新原生低位指令/新布局已在含全部成本的微基准胜 FP8 |
| c9/c10 BM64 或 BM128 主体+BM64 尾 | 不只全 BM64 失败，混合尾方案也有约 +19.5% 的历史负结果 | 明确改变 B 重读/调度结构，不能只减 padding |
| 再扫 stages/warps/GM/maxnreg | 已有数百次参数实验，末六次晋升累计收益很小 | 新主 kernel/新布局导致资源约束改变 |
| 普通“融合 GU、SwiGLU、量化” | 当前 MD 已融合这些工作，往往是重复提案 | 指出当前仍存在的实际中间写读，并证明去掉它 |
| 路由小权重剪枝 | 路由接近均匀，已有 c5 SQNR 21.37 dB 失败且筛选开销大 | 允许输入下有可验证的新误差/稀疏机制 |
| 高层 GEMM/官方接口绕过 | 题面限制和源码验证明确拒绝 | 主办方明确修改允许 API，不是拼接符号名 |
| BF16 atomic 直接归并 | k≥3 的求和顺序与确定性；k=2 原型也慢 | 新的确定性归并结构，且总同步/流量更少 |

EP 读数版、单卡 kernel probe、完整 P1 AC 的证据层级不同。报告可以用明显负收益停止当前 EP 家族，但不能把尚未提交的 scored EP 版本描述成“12 案正式验证全部通过”。

Gluon 显式 Hopper WGMMA 的 3.4 官方入口只有 barrier/TMA 等能力，当前不能套用新版本教程。参考 [固定 3.4 源码](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py) 与 [本地版本审计](notes/gluon_version_capability_0913.md)。官方未来升级才重新立项，当前无需再花一次正式提交探同一版本。

## 10. 统一实验协议

### 10.1 三层验收

1. **CPU/静态层：** 数学索引、容量、AST/语法、有效定义、diff 与触碰 case。可淘汰错误设计，不产生 GPU 性能结论。
2. **单卡 custom 层：** 抽取所有被改阶段，oracle 全量对照，动态输入多样化，测包含新增代价的子链。一个正结果只是正式 P1 的入场券。
3. **完整 P1 层：** 四 rank、全部 case、sample/fallback、输入只读、有限性、确定性、真实端到端 tk、总运行预算。只有这一层可晋升生产。

对于纯布局/索引优化，优先要求与 v842 数学输出 exact。对于数值算法变化，必须对独立题意参考算 SQNR，不能只对另一个近似 kernel。已有日志最差 SQNR 约 23.11 dB，距离 22 dB 门槛有限；新误差与旧误差可能相关，不能把 1.11 dB 直接兑换成任意额外压缩率。

### 10.2 测量设计

- 冻结候选 SHA、锚 SHA、触碰 case、唯一机制、主要指标和停止条件，再提交。不要边看结果边扩展“成功”的定义。
- 平台只有一个操作者，串行提交；其他 agent 产代码、oracle 和解释，不能同时取 token、提交或清理他人 Pending。
- 初筛交替候选/锚，确认批预先固定 AB/BA 顺序与数量。异常点保留原数据和剔除理由，不只保留最小 tk。
- 分别保存 tk、tb、逐案性能分、display、状态、SQNR、determinism。`score=100` 在部分 API 表示 correctness，与性能 `displayScore` 不能混用。
- 优化结论以 tk 为主，同时用冻结 tb 模型折算积分。实际榜分用于记录，不用随机增大的 tb 给候选背书。
- 机器差异逐案不同。旧 `scripts/mnorm.py` 的签名已经被后续数据否定，不能原样当裁判。参考 `strat*.py` 的方法，但更新同代码路径的 anchor 池，并从机器控制集合排除所有触碰 case。
- 全案都改变时没有未触碰控制，使用交替配对与独立复测，不强行“去机器噪声”。不同平台窗口需要重新校准。
- pooled mean、配对差、校正差应方向一致；不一致时结果是“未证实”。共享 anchor 生成的许多 pair 不是相同数量的独立运行。
- 筛选与确认分开。先筛出赢家再持续采样直到 z 过线存在选择偏差；确认使用冻结实现和新的一批样本。不要在 20 个 case/候选组合里只报告最有利的一个。

简单预算估算：若配对效应标准差为 σ、想检出的降幅为 δ，检测样本量随 `(σ/δ)^2` 增长。历史 c9/c10 的 0.3% 级信号可能需要近百对，收益不足时应直接止损。

### 10.3 阶段计时的当前限制

新增材料里有参赛者报告 `torch.cuda.Event` 被拒，历史本项目则有 event 读数。两者都不能单独代表今天所有通道的权限。优先使用当时允许的 custom 完整 workload tk；若阶段 event 接口确实允许，记录其范围和开销。**不绕过代理、符号扫描或诊断限制。**

以两个 payload 比子链时，wrapper、重复次数、输入生成/预热和计费范围必须一致；至少保证每次重新执行动态计算，不能重复读已缓存输出。使用不同矩阵或重复机制避免把 L2 热命中误认成 P1 全量流量收益。没有可信计时入口就把阶段性能标为未知，不能由资源数字推算“已提速”。

### 10.4 本地工具使用

```bash
cd /Users/sakimi/Desktop/xpuoj-p1
shasum -a 256 p1/kernel.py
python3 reports/score_audit_20260927.py
python3 reports/2026-09-27-padded-map-proof.py
```

正式提交前使用 [SUBMISSION](docs/SUBMISSION.md) 的已验证入口；`scripts/best_score.py` 是旧认证入口，不用于当前榜面核验。`scripts/nscore.py` 还依赖旧临时目录，不应直接执行来分析本轮。本轮评分重算器仅用仓库内保存的数据，不访问网络或凭据。

## 11. 可直接派给 coding agent 的任务

### Agent A：c4 padded 输出主线

> 工作目录为 `/Users/sakimi/Desktop/xpuoj-p1`。先读本指南第 1、2、5、10 节及 09-13 padded custom。冻结 v842 SHA 为 `5cfd6a80d066d8a729ff1b6741541c1c70df7df75eeffe40c67440d4fe962a3c`。实现仅 c4 的 padded Down+DSCL 和直接 INV_PAD，保持原 INV、紧凑 ACT 和分支求和顺序。先选 MD-epilogue 或 sort-scatter 一个索引载体。修正 fin host 的 T 推导。做全量索引/数值 oracle，并把所有新增准备成本纳入子链。不得改生产文件或自行并发提交。交付候选、diff、SHA、oracle、成本预算和“继续/停止”的证据；不要扫 launch 参数。

### Agent B：排序算法初筛

> 先读本指南第 6、10 节。测现役完整 sort 的成本，再实现一个确定性局部 rank 算法；目标减少 dense eq+cumsum，不改变 ORDER 的稳定性。覆盖 E=32/96/256 和极端分布，ORDER/INV/COUNTS 逐元素精确比较。报告整个 sort 链和端到端收益上限。只有收益明显时申请一个 case 的正式候选；若上限不足 0.5% 全案，停止独立路线。不得覆盖 A 的修改或提交平台。

### Agent C：正确性、路径与编译工作集

> 先读本指南第 2.3、8 节，按最后绑定定义审计 12 案及 sample 的路径。输出任意调用序列的统一数学路径候选与实际 JIT 特化清单，处理 same-shape 新权重对象的 cache 失效并明确原位改写未决契约。不要把删除函数数目当编译收益，不要依赖先跑错误/低精度路径再只在计时调用启用快路径。保留未知 shape 的合法 fallback。提交前给出至少 8 次重复、跨 shape、动态 X 与权重变化的测试计划/结果。

### 唯一平台执行者

> 只执行已经冻结并附带 oracle 与成本说明的候选。读 SUBMISSION 并检查当时队列；按第 10 节保存逐案 raw tk/tb/SQNR/状态、SHA、触碰集、配对关系。先 custom 筛选后 P1，保持单提交者。不得以等待超时为由重复发同一候选，不为碰 tb 异常窗口重投。提交结果同时反馈“性能是否成立”和“距离积分门槛多远”，不要仅说 Accepted。

Agent 可以先各自分析/实现；只有平台执行者串行。若资源只能同时跑一个 custom，也按 A→B→C 的具体 ready 状态调度，不让等待成为盲扫理由。

## 12. 时间与预算安排

本地比赛材料记载截止为 10 月 1 日 23:59（UTC+8）；本轮未取得当前公告正文确认是否变更，因此实际安排以平台最新显示为准。按未延期准备：

- **首个工作日：** A 的 CPU/当前 s3 单卡验证；B 的纯排序成本；C 的路径表和 cache 风险清单。各线交一页“还有多少可能收益、下一发能区分什么”。
- **第二个工作日：** 只让通过单卡门的 1–2 个候选进入 P1；先单独作用一个 case。总预算建议 6–12 次正式评测用于初筛，不同时开十个微候选。
- **第三个工作日：** 对幸存候选进行独立固定确认，验证累计而非相加收益；有收益再考虑扩到第二个 case。
- **最后一个工作日：** 冻结、正确性/确定性/缓存复查、回退材料、最终精确源文件与 SID。避免最后时刻引入新通信或新精度链。

这些是实验预算建议，不是平台额度限制。即使正式提交当前没有额外经济成本，也要控制队列时间、证据质量与比赛禁止高频扰动的约束。

## 13. 仍需要主办方澄清的具体信息

用户已经补齐 GPU 资源、版本、拓扑与超时的关键说明，不需要再重复询问相同内容。仍未解决的是：

1. P1 相同 shape 跨测试点是否换 tensor/storage，是否可能在同对象上原位更新权重，有无允许使用的缓存失效信号？新增材料中的提问针对 P2，且未见答复，不能自动外推 P1。
2. 500 秒覆盖的具体阶段、sample 与正式十二案的组织方式。官方文字“十个”与当前十二案日志差异应澄清；最好提供可见阶段时间/PTXAS stderr。
3. 当前合法的阶段性能诊断方式，以及单卡 custom 与 P1 镜像的准确版本差异。不要让 agent 自行假设 Event/graph/底层 profiling 能绕过拦截。
4. 路径统一后的预热预处理与静态缓存能否按正式契约复用；官方题面的一般许可不等于 shape-only cache 永久有效。

不必等这些答复才能做 A/B 的 CPU 与单卡工作，但它们限制完整缓存合同和超时因果结论。需要补问主办方时，由用户决定对外沟通，本轮没有发送任何外部消息。

## 14. 每个后续候选的交付模板

```text
候选名称 / 日期 / 负责人：
基线文件 + SHA：
候选文件 + SHA：
唯一机制 / 触碰 case：
历史最近邻及本次改变的前提：
数学、数据布局、缓存与并发不变量：
新增/减少的字节、launch、特化、峰值显存：
CPU 验证：
单卡版本、oracle、完整被触碰链时间：
P1 SID、锚 SID、状态、逐案 tk/tb/SQNR/determinism：
原始/配对/机器校正效应与不确定性：
固定 tb 积分变化 / 当次真实榜分：
继续或停止，触发哪个预设门：
合并后累计收益 / 回退文件：
```

要求把结果写回当日 `experiments/YYYY-MM-DD/`，原始数据与摘要都保留。生产晋升要更新 SHA 和有效代码导航，不再用页首旧状态加很多互相覆盖的“当前”段落。

## 15. 本轮证据附件

- [内核路径与新机制审计](reports/2026-09-27-kernel-audit.md)
- [历史负结果、证据层级与重启条件](reports/2026-09-27-history-audit.md)
- [评分审计及样本边界](reports/2026-09-27-score-audit.md)
- [评分重算脚本](reports/score_audit_20260927.py)、[逐案阈值](reports/2026-09-27-score-thresholds.csv)、[情景表](reports/2026-09-27-score-scenarios.csv)
- [padded 映射 CPU 检查](reports/2026-09-27-padded-map-proof.py)、[检查结果](reports/2026-09-27-padded-map-proof-results.json)
- [实时只读平台结果](reports/2026-09-27-platform-readonly.json)、[补取逐案结果](reports/2026-09-27-latest-cases.json)
- [冻结基线 manifest](reports/2026-09-27-baseline-manifest.json)、[新增官方答复快照](reports/2026-09-27-addinfo-snapshot.md)

这组方向的价值在于：保留一个已有正机制但尚未兑现整链收益的机会，验证一个尚未被同机制负结果覆盖的算法缺口，并修复妨碍判断的路径与测量问题。任何进一步提分，都必须由后续 GPU 和完整 P1 证据确认。
