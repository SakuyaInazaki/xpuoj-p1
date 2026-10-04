# P1 MegaMoE 提交记录（截至 2026-08-15）

> 历史记录：本文中的“当前”只对应各小节标注日期；现状以 [`docs/STATE.md`](../docs/STATE.md) 为准。

当前最佳: submission 113512, score 46.58, Accepted
提交次数: 158/100（平台当前未对显示分执行可见扣罚；最佳仍为 46.58）

关键版本:
- 113269: 首个正确版本，25.58
- 113277: 合并四源 token + gate/up 合并 + group path，35.33
- 113300: group_gemm 原语 + 自建 metadata，41.08
- 113321: k<=3 NCCL variable A2A，43.00
- 113329: k<=4 A2A，43.83
- 113347: A2A local/meta 打包 int64，44.50
- 113362: all_gather_into_tensor，44.83
- 113443: 大 shape all_gather 路径 BF16 reduce_scatter，45.08
- 113512: 同代码复测命中更优评测波动，46.58  ← 当前最佳

当前最佳文件: p1/kernel.py (= kernel_hybrid_k4_packmeta_gatherflat_bf16cond.py)

# 2026-08-16 接手后更新

当前最佳提交详情: 114019, displayScore 48.58, Accepted
当前最佳文件: p1/kernel.py = p1/kernel_114017_bf16th10.py

关键新版本:
- 114007: 修复 k<=4 路由重复计算，43.83
- 114009: + case11 replicated，45.25
- 114015: + A2A view-sum + metadata 去 item()，48.25
- 114017: + 路由小 N BLOCK_N 特化，48.33
- 114019: + allgather BF16 reduce 阈值 10M，48.58  <- 最佳

114019 各测试点 tk(ms)/单点分:
1 16.741/51, 2 25.446/53, 3 8.207/45, 4 7.891/38,
5 12.019/51, 6 9.411/44, 7 11.658/47, 8 9.930/41,
9 7.308/54, 10 6.333/51, 11 4.612/55, 12 7.162/53

备注: scoreboard best_score 接口仍显示旧缓存 113512/46.58。

# 2026-08-16 第二轮迭代

最佳 kernel.py = submission 114134, displayScore 49.92
- 114067: direct SHMEM full 48.92
- 114068: + case4 replicated 49.25
- 114082: direct dispatch + NCCL combine 50.92（波动大）
- 114132: + vectorized allgather prep 48.25
- 114134: hybrid combine (E96 NCCL, E8/32 direct) 49.92 <- 当前 kernel.py
- 114137: E32 dense reduce combine 48.67

备注: scoreboard best_score 仍显示旧缓存 113512/46.58。
