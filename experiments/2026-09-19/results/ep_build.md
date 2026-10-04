# c9/c10 融合式 EP 第一阶段构建日志（2026-09-20）

基线：`p1/kernel.py` = v837，SHA-256 `81d994bea8e0a22e8cd95d7b5dfe7a519a95968d822be54bcddaa7d0423f7f91`（不改动）。

## 设计要点（与 09-19 三个探针的差别）

- **不物化 A**：4 个 rank 的 x_fp8 收进一个连续 `[world*T, H]` 对称张量；
  md 内核本来就用 `rows = ORDER[m] // KTOP` 取 A 行 ⇒ 令
  `ORDER[m] = (s*T + t)*k + j` 即可直接寻址；`W`/`A_SCALE` 同理落在
  `[world*T*k]` / `[world*T]` 连续视图上。排序阶段只搬 int64 索引（256 KB），
  不搬 134 MB 的 A（这是 09-15/16 两次 EP 尝试的主要开销之一）。
- **免二次排序**：每个 rank 本来就做 E=256 计数排序（replicated 同一特化），
  专家 ∈ [64d, 64d+64) 在 `order` 里是连续区间；全量 all-gather
  `order`(256 KB)+`counts`(1 KB) 后，收端只用一个 `[4,64]` 前缀和
  （`_ep_off_kernel`，单 program）+ 一个段拷贝（`_ep_ord_kernel`，256 program）
  即可拼出全局有序 ORDER 与反向索引 INVEP。
- **GEMM 零新增特化**：md/dn 直接复用 `_fgs_tma1_kernel_gq_tiled` /
  `_dn_tma2_f8_tiled_kernel`，全部 constexpr（K/N/BLOCK_*/GROUP_M/KTOP/EPP/KT/FLAT）
  与 launch 参数逐字与 replicated 相同；权重用 tile-contiguous FP8 的
  `narrow(0, 64*rank, 64)` 视图（expert-major ⇒ 零拷贝）。
- **行数上限 `_EP_MCAP = 65536`**（均值 32768 的 2 倍），全部中间 buffer 按上限
  预分配并缓存 ⇒ 不需要把 device 侧的 Mtot 同步回 host，也保证各 rank 分配
  shape 完全一致。
- 通信沿用 bw sweep（SID 146245）的 132-CTA 大块 `putmem_nbi_block`；每个推送
  CTA 自己 `fence()` + `putmem_signal_nbi_block(..., SIGNAL_ADD, 1)`，收端
  `signal_wait_until(CMP_GE, epoch*NCH)`（epoch 递增、无 reset）。
- 对称 buffer 在 call 3 一次性创建 + 一次 `nvshmem_barrier_all_on_stream`。

## 提交记录

| SID | 候选 | 说明 | 结果 |
|---|---|---|---|
| 146452 | `ep_v1_readout.py` (sha256 `dcb94dec0f5ca0051cd7a2ce52936eed7ea6b5d09f3d877693eb26f9fdcf22be`) | 第一阶段读数档 | **跑通、无死锁**；c9 tot 3.702/3.989 ms，c10 3.591/3.781 ms（call4/call5）；其余 10 案 Accepted |

## SID 146452 逐段读数（rank0 CUDA event，单位 ms）

| 段 | c9 call4 | c9 call5 | c10 call4 | c10 call5 |
|---|---:|---:|---:|---:|
| rq 路由+量化+E256排序+槽拷贝+fill | 0.715 | 0.662 | 0.904 | 0.572 |
| dpush dispatch 推送内核 | 0.546 | 0.531 | 0.415 | 0.384 |
| dwait dispatch 等信号 | 0.100 | 0.470 | 0.006 | 0.849 |
| sort ORDER 合并+metadata | 0.018 | 0.019 | 0.128 | 0.020 |
| md | 0.905 | 0.906 | 0.909 | 0.670 |
| dn | 0.449 | 0.452 | 0.379 | 0.347 |
| rred 按目的 rank 归并 | 0.143 | 0.145 | 0.153 | 0.143 |
| rpush return 推送内核 | 0.644 | 0.653 | 0.632 | 0.659 |
| rwait return 等信号 | 0.118 | 0.087 | 0.006 | 0.074 |
| comb 4 路求和 | 0.064 | 0.063 | 0.058 | 0.063 |
| **tot** | **3.702** | **3.989** | **3.591** | **3.781** |

- 现役 replicated：c9 ≈ 2.50 ms、c10 ≈ 1.95 ms ⇒ EP 第一阶段 **慢 1.2~1.8 ms**，远未达 ≤2.30 闸门。
- GEMM 侧收益属实：md+dn = 1.36 ms（c9）/ 1.02 ms（c10），对比 replicated 约 2.3 / 1.9。
- 亏在通信：dispatch(dpush+dwait) 1.00 ms / 1.23 ms + return(rpush+rwait) 0.74 ms ⇒ 通信 1.74~1.97 ms。
  - return 腿 100.7 MB 出带宽 0.74 ms ⇒ 136 GB/s，已贴 09-19 bw sweep 的 157 GB/s 平台。
  - dispatch 腿只有 50.3 MB 却要 1.00~1.23 ms ⇒ 等效 40~50 GB/s，差额是 rank 间 skew
    （dwait 在 c9 call5 = 0.47、c10 call5 = 0.85，且与 dpush 呈此消彼长）。
- aux 侧 rq 0.57~0.90 ms 也比 replicated 的同名段贵（含 11 个 cuda Event 构造/record 的读数开销）。

## 基线对照（同底盘 v837，SID 146398）

`tc9 tk=2.483 sqnr=23.13`、`tc10 tk=1.932 sqnr=23.11`（Σtk 29.393，display 81.5）。

## 修补轮（stage 2(i)：int8 回传 + 读数开销移出计时路径）

| SID | 候选 | 改动 | 结果 |
|---|---|---|---|
| 146463 | `ep_v2_int8_readout.py` (sha256 `0b3970e4dbb54dec114a2d97e515c810b981035945aedbaabc5ec21cfe532b9f`) | ① return 腿由 bf16 [T,H] 改 int8 [T,H] + 每 (行,256 列块) 一个 fp32 尺度（出流量 100.7 MB → 51.5 MB）；② 11 个 cuda Event 改为缓存复用，不进计时路径；③ call 5 数值自检 | **c9 最好 3.007 / c10 最好 2.578 ms**；数值 s2n=18430（c9）/18437（c10）⇒ EP 相对 replicated 仅 −42.7 dB；其余 10 案 Accepted |
| — | `ep_v1_scored.py` (sha256 `e2cbd9672c45b8ffb501cc576df0d0d1b1912a4fb4ced65294b72b43fd681b76`) | 第一阶段计分档（不抛异常） | 未提交：api_token 额度耗尽（0/10，下次回血 17:04Z） |
| — | `ep_v2_int8_scored.py` (sha256 `635b5cb84a9f9c961107e69b9212735eba35a9e4a3581fc3e54045f0ab0c489c`) | 修补轮计分档 | 未提交，同上 |

## SID 146463 逐段读数（int8 回传，单位 ms）

| 段 | c9 call4 | c9 call5 | c10 call4 | c10 call5 |
|---|---:|---:|---:|---:|
| rq | 0.706 | 0.672 | 0.545 | 0.917 |
| dpush | 0.417 | 0.383 | 0.387 | 0.415 |
| dwait | 0.113 | **0.006** | **0.006** | 0.634 |
| sort | 0.020 | 0.019 | 0.020 | 0.020 |
| md | 0.883 | 0.969 | 0.736 | 0.654 |
| dn | 0.437 | 0.439 | 0.365 | 0.336 |
| rred | 0.135 | 0.138 | 0.138 | 0.138 |
| rpush | 0.331 | 0.336 | 0.336 | 0.332 |
| rwait | 0.182 | **0.006** | **0.006** | 0.432 |
| comb | 0.039 | 0.039 | 0.039 | 0.040 |
| **tot** | 3.262 | **3.007** | **2.578** | 3.918 |

- int8 回传按预算兑现：`rpush` 0.653 → 0.334（−0.32 ms），`comb` 0.063 → 0.039。
- **数值正确**：`s2n`（EP 输出 vs 同调用 `_run_replicated` 输出的功率比）= 18430 ⇒ −42.66 dB。
  与 replicated 自身 23.13 dB 叠加后 SQNR ≈ **23.08 dB**（阈值 22），即 EP+int8 只吃掉 0.05 dB。
- 两次提交的其余 10 案全部 Accepted，**无样例槽编译预算 TLE** ⇒ 新增的 7 个小内核
  （push_d / push_r / wait / off / ord / ret / comb）+ E=64 metadata 特化在预算内。

## 结论：c9/c10 融合式 EP 不可用（路线关闭）

取两案各自 **零 skew 的那一次调用**（即最好情况）：

| | EP (int8 回传) | replicated v837 (SID 146398) | 差 |
|---|---:|---:|---:|
| c9 | 3.007 | 2.483 | **+0.524** |
| c10 | 2.578 | 1.932 | **+0.646** |

第一阶段闸门 ≤2.30 ms 未达成，修补一轮后仍 ≥2.60 ⇒ 按约定停。

c9 零 skew 调用的收支（对比 replicated 约 1.45 md + 0.48 dn + 0.55 aux）：

- **GEMM 侧只省了约 0.52 ms**，不是账上的 0.95。原因：EP 的 md 是 0.969 而不是
  09-19 GEMM 探针的 0.80 —— 权重流量降 4×，但 A 的不同行数从 4096（16.8 MB，L2 常驻）
  涨到 16384（67 MB，非常驻，平均被读 2 次 ⇒ 134 MB 实流量），这一项把权重侧的收益吃掉大半。
- **通信底价 0.719 ms**（dpush 0.383 + rpush 0.336），已经是 157 GB/s 平台上的两腿
  50.3 MB + 51.5 MB 的物理下限，没有进一步压缩空间（x 已是 fp8，回传已是 int8）。
- **EP 专属 aux +0.31 ms**：按目的 rank 归并 0.138 + 4 路合并 0.039 + 槽拷贝/fill 约 0.12。
- ⇒ 0.52 收益 vs 1.03 支出。**即使 skew 完全消除、通信完美重叠掉一半，也追不平 2.483。**

附带记录（可复用的事实）：
1. 「不物化 A」的 EP 形态是可行的：`ORDER[m] = (s*T+t)*k+j` + `[world*T,H]` 连续对称
   buffer，md/dn 零新增特化，`sort`（ORDER 合并 + metadata）只要 0.019~0.020 ms。
2. 132-CTA `putmem_nbi_block` 的 return 腿（51.5 MB 出）0.334 ms ⇒ 154 GB/s，
   与 09-19 bw sweep 的 157 GB/s 平台一致；dispatch 腿（50.3 MB）0.383~0.417 ms ⇒ 126 GB/s。
3. **rank skew 是本题 EP 的隐性大项**：同一份代码，`dwait`+`rwait` 在不同调用里
   从 0.012 跳到 1.066 ms（c10 call5）。没有 barrier 的两腿结构会把上一轮的
   计时抖动全部搬到本轮的等信号上。
4. 每 (行,256 列块) int8 + fp32 尺度的回传量化只损 −42.7 dB，对 23.13 dB 的总 SQNR
   影响 0.05 dB —— 这条量化配方本身是安全的，可留作其他场景的备用手段。
