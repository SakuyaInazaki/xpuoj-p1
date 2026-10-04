# 2026-09-29 EP2 / 静态 DN 迭代收口

> 2026-09-30 补齐：v930 BM256/BN64 pair MD（151332）已 AC，但 c9/c10 为
> 2.950/2.737 ms，仍明显负；v914、v915–v926、v931 终态也已按 SHA 恢复。
> v912 最低日志 SQNR 实际为 c9 22.56 / c10 22.69。
> 本文下方保留当时快照；当前方向与更完整证据见
> [9 月 30 日指引](../../OPTIMIZATION_GUIDE_2026-09-30.md)。

生产保持 v890：`p1/kernel.py` SHA-256
`436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc`。

## 已确认

1. 旧 EP2 metadata 修复成立。把 per-expert row start 改传 per-tile row start 后，
   pair MD/DN 可执行。最终只用 `torch.cumsum` 被 determinism proxy 拒绝；
   换成小 scatter kernel 后通过。

2. 单 peer NVSHMEM direct put 版 EP2 可 Accepted。
   - 文件：`experiments/2026-09-29/candidates/v906_pair_ep2_direct.py`
   - SID 150965，Accepted，display 81.50。
   - c9 tk=2.705 ms，c10 tk=2.310 ms。
   - 同源 v890 描述性中位 c9=2.472 ms，c10=1.881 ms。
   - 结论：direct peer 比 world all-gather 版快 0.14 ms 左右，但仍慢于
     replicated v890。半专家权重没有降低每个 expert 的 B 读放大；EP2 不再扩大投入。

3. BM=256 的原位尝试全部失败，原因未拿到 ptxas stderr。
   - v907/v908（MD+DN BM256）、v909（仅 DN BM256）均 ptxas exit 255。
   - v910（MD BM256/BN64 + DN BM256/BN64 自定义）TLE。
   - 不再盲调 BM256。

4. c9/c10 compact static DN 数值通过但速度收益不足。
   - 文件：`v912_static_dn_c910_fix.py`
   - SID 151051，Accepted，display 81.83。
   - c9 tk=2.470 ms，c10 tk=1.856 ms；SQNR c9 22.62 / c10 22.75。
   - c9 无收益，c10 约 1.3%，未跨 q 档，且 c9 SQNR 余量下降。
   - 不晋升。

5. c1/c2 static L1 DN 界不可用。
   - 文件：`v913_static_dn_c1c2.py`
   - SID 151056，WrongAnswer；c1 SQNR 20.76。
   - 原因推断：K=8192/14336 时 sum_abs 界远大于 max_abs，q 被压入 FP8 次正规区。
   - 关闭。

## 关键 SID

| SID | 候选 | 结果 |
|---|---|---|
| 150961 | v905 旧版，metadata 已修但仍有 torch.cumsum | WrongAnswer |
| 150964 | v905 scatter 修复 | Accepted 81.33 |
| 150965 | v906 direct peer | Accepted 81.50 |
| 150966/150967/150969 | BM256 变体 | WrongAnswer, ptxas |
| 150973/150976 | v910 自定义 BM256/BN64 | TypeError 后 TLE |
| 151051 | v912 c9/c10 static DN | Accepted 81.83 |
| 151056 | v913 c1/c2 static DN | WrongAnswer, SQNR 20.76 |

## 判断

- EP2 的计算收益不足以支付通信、过滤和并行 overhead；在现有 BM=128 内核下，
  expert parallel 与 replicated 的权重 B 读放大相同，这是结构性原因。
- BM=256 是唯一能让半权重真正减少 B HBM 流量的方向，但当前 Triton/ptxas
  资源下未找到可用特化，需要新的 B 复用内核而不是继续改 launch 参数。
- 生产不晋升本轮候选。
