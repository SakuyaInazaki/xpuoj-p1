# 2026-09-16 接力迭代摘要（未晋升）

本轮接手上一个 agent 的 v833 生产入口，先读提交/令牌池说明和当前状态，再做六个候选（三个单变量、一个 EP 结构复核、两个权重 tile 布局/配置跟进）。production `p1/kernel.py` 保持 v833，SHA-256 `afb57d9fa5199092dffc3279435f7b429f09c5153dac63667ef126d2aaa5218c`，文件未改动。

只读核验：P1 榜面仍为 **72.75 / best SID 141408**，`submissionCount=2683`，无在途作业；令牌池可用。评测口径仍符合既有结论：计分窗口只覆盖 MD/DN 及其间 activation 量化，route/sort/gq/metadata/fin 在窗口外；配置轴（BM/BN/BK/GM/warps/stages/unroll/maxnreg）在当前内核族已基本认证完毕，因此本轮尝试了三个未穷尽的单变量点，并对 local-expert EP、权重 tile 连续布局和 c11/c12 TMA GROUP_M 做了跟进复核。

## 候选 1：c1/c2 q8 MD 外层 `tl.range(num_stages=2)`

- 文件：`experiments/2026-09-16/candidates/outerpipe_c12_q8md.py`
- SHA-256：`e83ad61df04a24e96f7ac2ef7566559878bf8dcc344a63f844e4e8691adc18e3`
- 改动：`_fgs_tma2_int_pm_q8_kernel` 的 tile 外层从 `range` 改为 `tl.range(..., num_stages=2)`，只影响 c1/c2。
- 终态：SID `144771` Accepted，display 81.08，Σtk 30.261；同日锚 SID `144772` Accepted，display 82.42，Σtk 29.673。
- 归一化结果：机器项 +1.41%，c1 净残差 -0.42%、c2 +0.04%，raw +0.005。低于噪声，未晋升。

## 候选 2：`_dn_tma2_f8_kernel` maxnreg=232

- 文件：`experiments/2026-09-16/candidates/dn_cap232.py`
- SHA-256：`8b755a30d2620bc770c1352191a7c34f8216bf17abd89a5bd009093d49f06898`
- 改动：`_dn_tma2_f8_host` 的唯一 launch 增加 `maxnreg=232`，影响 12 案 call5 DN。
- 终态：SID `144775` Accepted，display 81.58，Σtk 30.147；锚 SID `144776` Accepted，display 81.58，Σtk 29.587。
- 结果：Σtk 比锚 +1.89%，c1 +1.96%、c2 +3.15%、c3 +2.88%，其余多数 +0.5~2.3%，仅 c9 -0.70%。明显负收益，未晋升。

## 候选 3：`_GA` 快路径窗口 3..5 扩到 3..7

- 文件：`experiments/2026-09-16/candidates/extend_ga67.py`
- SHA-256：`5b8f6ae97b48271d9d37a1a73d48035b966ffb8070a867a7909e0bcd4cab2a1c`
- 改动：把 c3~c10 的 `_GA` 快路径从 call3--5 扩到 call3--7，并把 c9/c10 q8 MD 分支同样扩到 call3--7；目的是让 call6/7 也走 call5 已编译的快路径，同时保持 det 对 6=7 路径一致。
- 终态：SID `144781` Accepted，display 81.33，Σtk 30.238；锚 SID `144783` Accepted，display 82.42，Σtk 30.212。
- 结果：Σtk 比锚 +0.09%；逐案 c9 +0.00%、c10 -0.55%，c1 +0.70%、c3 +0.63%，整体中性。说明 call6/7 是否走快路径对最终计分无稳定收益，未晋升。

## 判定与下一步

三个候选均未产生可晋升的稳定收益；其中 maxnreg=232 的 DN 变体明确负收益，外层流水和 call6/7 快路径为噪声级中性。生产保持 v833，榜面保持 72.75。

本轮新增证据边界：

- c1/c2 q8 MD 再加一层 `tl.range(num_stages=2)` 不改变性能，说明现有 `range` 调度在长 K 下已接近该结构上限。
- DN 上 `maxnreg=232` 反而压慢，寄存器帽轴在 DN 侧仍应保持默认；此前只在 MD 侧提升了 168->232。
- c3~c10 call6/7 的慢/快路径替换不影响 tk，进一步支持“正式计分只依赖 call5 且 tk 取 min 或只取 call5”的判读；后续不要再为 call6/7 做路径对齐。
## 候选 4：EP 路径改为 call3 起步（c9/c10）

- 文件：`experiments/2026-09-16/candidates/ep256_call3.py`
- SHA-256：`92c2bf7d0de9f622d133a9674cab2b9bc7ac7294f8941ef055c5abc4dcf5f325`
- 改动：把 2026-09-15 `ep256.py` 的 local-expert EP 移植到 v833，并把 `_EPM` 触发从 `_CALLN>=2` 改为 `_CALLN>=3`，以避开 call2 提前装载重 TMA 内核的嫌疑。
- 终态：SID `144785` Accepted，display 79.83，Σtk 32.253；锚 SID `144786` Accepted，display 81.42，Σtk 29.647。
- 逐案：c9 `3.598 ms`（锚 `2.561`，+40.5%）、c10 `2.979 ms`（锚 `2.014`，+47.9%），机器项 +1.44%，净残差 c9 +40.27%、c10 +47.04%，raw -1.280。
- 结论：EP 不是提前装载/call2 问题；现行 tile 下局部队列行数从约 128 升到约 512，B 重读与 all-gather/reduce_scatter 税稳定为负。EP 路线继续关闭，不再重试同构设计。
## 候选 5：c9/c10 权重 tile 连续布局（未取得可用性能读数）

- 目标：回应旧 V594「tile 连续布局真死因=显存」的翻案条件，在当前 v833 上只对 c9 lowmem 权重做 tile 连续化。
- `tiled_c9.py`：
  - 新增 tiled MD/DN 内核、`_tile_rows_128`、`_get_tiled_c9`，只对 c9 call3--5 启用。
  - SID 144789、144794：c1 TLE（冷编译彩票）。
  - SID 144791：c9 TypeError；根因是新内核漏了 `@triton_dist.jit`。
  - SID 144806（补装饰器后）：c1--c8 Accepted，c9/c10/c11 WrongAnswer，c9 `cudaErrorIllegalAddress`。根因是 tile 指针偏移以 int32 计算，c9 gu tile 元素数达到 2^32，指针回绕。
- `tiled_c9.py` 修 int64 后：
  - SID 144811：c10--c12 Accepted，c9 OOM。说明 tile 访问本身能跑，但共享进程里前 8 案的 shape 缓存尚未释放，额外 6.44GB tiled 副本导致 80GB 显存溢出。
- `tiled_c9_evict.py`：
  - 增加「切换 testcase shape 时清理上一案大权重缓存」逻辑，试图给 tile 布局腾内存。
  - SID 144814：c1 TLE。
  - SID 144824（同 SHA 重发）：c1--c8、c10--c12 均 Accepted，c9 仍 OOM。说明 shape 切换时清理旧权重缓存不足以腾出额外 tile 副本所需的显存；显存压力主要来自判题机共享进程/运行时，而不是本文件可清理的 shape 缓存。
- 判定：tile 连续布局的逻辑已能跑到 c9，但当前判题机仍有 c1 首调 TLE 和共享进程显存双重约束；旧 V594 的结论在 v833 上仍成立。该路线停止原样复测。

## 候选 6：c11/c12 TMA md GROUP_M 8→32

- 文件：`experiments/2026-09-16/candidates/tma_gm32_c1112.py`
- SHA-256：`b9888cce0b70a6f67ec378e8642d61aaa632bb35ef1444be17b5b893c570571a`
- 改动：`_fgs_tma1_intq_host` 的 TMA md 分支 `GROUP_M=8 if K <= 1024 else 32` 改为恒 32，仅影响 c11/c12。
- 终态：SID 144819 Accepted，display 81.5，Σtk 29.599；锚 SID 144820 display 81.33，Σtk 30.235。
- 归一化结果：机器项 -1.39%，c11 -0.51%、c12 -0.10%，raw +0.006，未晋升。

## 本轮最终状态

- 榜面仍 **72.75 / best SID 141408**，`submissionCount=2697`；生产仍 v833，P1 无稳定晋升。
- 本轮最强的结构线索仍是权重 tile 连续布局，但受 c1 首调 TLE 与共享进程显存限制，当前无法转成正式读数；EP 已用 v833 复测为稳定负收益。
## 候选 7：c11/c12 仅 DN 权重 tile 布局（指针版，判负）

- 文件：`experiments/2026-09-16/candidates/tiled_dn_c1112.py`
- SHA-256：`5696575b541195fb82e3ddfe9ebe9ee34fd9fdbcf34cb8a41868c5b9744e73a5`
- 目标：避开 c9 显存问题，只在权重很小的 c11/c12 上验证 tile 连续布局对 DN 的影响；tile B 用指针 `tl.load` 读取，c11/c12 的 dn 权重仅 33/67MB。
- 终态：SID `144828` Accepted，display 81.17，Σtk 30.541；锚 `144829` Accepted，display 81.5，Σtk 29.554。
- 归一化结果：机器项 +1.32%，c11 +21.03%、c12 +13.17%，raw -0.334。
- 结论：去除 TMA 描述符、改用 tile 连续指针 load 后，DN 内层 K 循环的访存/流水明显劣化，c11/c12 直接慢 13--21%。这否定了“指针版 tile 连续布局可替代 TMA box”的实现假设；旧 V543 的 tile 布局用的是 TMA descriptor，不是指针版。tile 布局若再试，必须保留 TMA descriptor，且要解决额外副本显存问题。


## 2026-09-16 接力后半（本会话续跑）

接手后按 README 读取 STATE / SUBMISSION / CODE_MAP，并确认生产入口仍为 v833
（SHA-256 `afb57d9fa5199092dffc3279435f7b429f09c5153dac63667ef126d2aaa5218c`）。
本段实验集中在 c9/c10 权重 tile 连续布局、c9/c10 TMA-A host 路由和
`merge_gq_c910` 的 c10 复核；全部未晋升。

### c9/c10 TMA 权重 tile 连续布局

- `tiled_tma_lowmem_c910.py`（SHA-256 `0863145a32619b8c718245fe1c570f3df38aa443c95db971546a293a9cf1d1a4`）：
  给 MD/DN TMA box 加 TILED load，并直接构造 tiled lowmem FP8 权重，避免旧
  `tiled_c9` 的额外 6.44GB 副本。
- SID 144840 / 144844：`samples[0]` TLE；correctness/determinism 两路均通过，
  time 约 429s，未进入正式计时。
- `tiled_tma_lowmem_c910_v2.py`（SHA-256 `26a7df5419650327cd600e818e094c721dc28bcad66b51c83a0514ea5d9b72f9`）：
  保留原 `_dn_tma2_f8_kernel`，新增独立 `_dn_tma2_f8_tiled_kernel`，排除共享 DN kernel。
  SID 144846 / 144850 仍 `samples[0]` TLE。
- `tiled_tma_lowmem_c910_v3.py`（SHA-256 `1732b88f8e225e65990cf576c14903e8fd92d6897840785cf180651412715188`）：
  所有 baseline JIT 函数恢复到与 v833 逐函数同源，仅新增两个样本路径不调用的 tiled
  JIT kernel。SID 144853 仍 `samples[0]` TLE。
- 结论：当前判题样本槽对新增/改动的 c9/c10 JIT specialization 很敏感，不是单纯
  共享 DN kernel 的问题。tile 连续布局若继续，应避免新增 kernel，或先解决可信的
  warmup/cache 机制问题。

### c9/c10 host-only TMA-A 尝试

- `c910_tma2_blockinterleave.py`（SHA-256 `2f8260eb66ae7e48c32a971b8795b7b3faa2232b2a07a5623783398c0da3acc2`）：
  host-only 将 c9/c10 改成 sorted-A + `_fgs_tma2_int_pm_q8_kernel` +
  block-interleaved gate/up；全部 78 个 baseline JIT 函数与 v833 同源，无新 JIT 函数。
- SID 144857：`samples[0]` TLE；correctness/determinism 通过，time 约 442s。
- 结论：不要给 c9/c10 引入新的 `_fgs_tma2_int_pm_q8_kernel` specialization；
  现有 kernel 调用形态更安全。

### `merge_gq_c910` c10 复核

- `merge_gq_c910.py`（SHA-256 `7557256f783e6d39979588becf3a4d73444b3495070014d4254faf7e005c589a`）
  SID 144858 对锚 144859：display 81.25 对 81.67，机器项 +1.35%；
  净残差 c9 +0.33% / c10 -0.21%，raw -0.002。
- 与 SID 144836 对 144837 的 c10 -2.03% 不一致；复核判为噪声，不晋升。
- `merge_gq_c910.py` 本身仍可作为安全 host-only 实验载具，但没有可晋升收益。

### 本段最终状态

- 生产 `p1/kernel.py` 保持 v833，SHA-256
  `afb57d9fa5199092dffc3279435f7b429f09c5153dac63667ef126d2aaa5218c`，未改动。
- 本段新增提交均已终态，无在途；榜面结论不变。
- 下一方向：优先只做不新增/不改变 JIT kernel 的 host 配置实验；若必须改 kernel，
  先用最小 module 验证 samples 槽的编译预算。

## 2026-09-16（本会话）v834 晋升：c11/c12 DN GROUP_M 32→8

- 候选：`experiments/2026-09-16/candidates/dn_gm8_c1112.py`
- SHA-256：`714b3822784f85bd52f5a1c1fdd5b812b84845a60563a5eaefe174b46cc56935`
- 改动：`_dn_tma2_f8_host` 的 `GROUP_M=4 if K == 8192 else 32` 改为
  `GROUP_M=4 if K == 8192 else (8 if N == 1024 else 32)`。只影响 c11/c12 的 DN。
- 验证：
  - SID 144889 对锚 144890：机器项 -1.20%，净残差 c11 -1.01% / c12 -1.30%，raw +0.023。
  - SID 144891 对锚 144892：机器项 -1.37%，净残差 c11 -0.92% / c12 -0.80%，raw +0.017。
  - 同轴对照 GROUP_M=4（SID 144895 对 144896）raw -0.007，GROUP_M=16（SID 144897 对 144898）raw -0.003，
    均不如 8。
- 结论：c11/c12 DN 的 GROUP_M 局部最优点在 8，约 1% 单案收益且两次方向一致；已晋升为
  v834，生产文件 `p1/kernel.py` SHA-256 同步为
  `714b3822784f85bd52f5a1c1fdd5b812b84845a60563a5eaefe174b46cc56935`，v833 保留回退。


## 2026-09-16 结构创新追加：P1 TP/EP 路线平台验证

本轮按“必须结构性创新”继续，实际构造并提交了四种新结构；全部保持数学正确和确定性，但都没有超过现役 replicated v835。

- `tp256_c9c10_v3.py`：I 维 4-way TP，全量 all-gather token，每 rank 只读 1/4 专家权重切片；SID `145072` Accepted，c9 tk=3.894 ms、c10 tk=5.208 ms，基线约 2.56/1.96 ms。
- `tp256_pair_rs.py`：2×2 token×I pair 方案，每 pair 处理一半 token，pair 内 rank 分 I 半，输出用 reduce_scatter；SID `145084` Accepted，c9 tk=5.054 ms、c10 tk=3.329 ms。
- `ep256_direct.py`：EP c9/c10 路线，使用 NVSHMEM direct all-gather 替换 4 个 NCCL all_gather；SID `145086` Accepted，c9 tk=3.580 ms、c10 tk=2.973 ms，仍慢于基线。
- `tp256_pair.py` 的 pair-local P2P 版本在平台上触发 unbatched P2P 警告/SIGABRT；`tp256_pair2.py` 的 subgroup 版本 hang/TLE，已 cancel，不作为后续路线。

共同失败机理：降低权重 HBM 流量后，token all-gather、输出归约、短 K/双倍 M tile 的启动/排空开销超过权重节省。P1 当前 replicated v835 仍是最优结构；历史最好仍 SID144891/display 83.08。


## 2026-09-16 晚间记录：v836 c9/c10 权重 tile 连续布局（小幅晋升）

- 候选：`experiments/2026-09-16/candidates/tiled_v3_v835.py`，SHA-256 `d849cd97cc8a44e79d1cf42c76beff276410032c2af7f64c3f80586de5974de7`。
- 内容：把 09-16 早些时候未落地的 c9/c10 低内存权重 tile 连续布局补到 v835 上，并保留 v834/v835 的 DN GM8 与 c2 GM16。
- 平台：SID `145183` Accepted，display 81.25；同码窗口对照 SID `145144`（v835）归一化后 c9 约 −3.4%、c10 约 −1.0%，其余 10 案代码相同。
- 生产入口已切换为 v836：`p1/kernel.py` = `p1/kernel_v836_tiled_c910.py`，SHA-256 `d849cd97cc8a44e79d1cf42c76beff276410032c2af7f64c3f80586de5974de7`；回退基线 v835 = `p1/kernel_v835_c2md_gm16.py`，SHA-256 `3fb4e0e5d9cae0f567f41357fb313e5e25f70f0a0b693183b21f1c28806e4c62`。
- 同日晚间 f16 BN512 c11/c12 候选 SID `145181` 虽 Accepted，但 c11/c12 分别慢约 42%/50%，关闭。
- v836 最佳榜面仍未刷新（P1 仍 73.08 / best SID 144891；后续同码/近码提交均为 81.1–81.3 display）。
