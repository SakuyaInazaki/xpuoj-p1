# R1 两卡分组 EP2 探针记录（2026-09-27 下午）

## 结论

- 本轮把 R1 推进到“通信 + 路由过滤 + 半专家计算”原型，未得到可计分结果。
- 已单独证明：**pair 通信本身可运行**，**过滤后的 counting sort + metadata 可运行**；
  失败卡在 **pair 路由/半专家 MD GEMM 启动**。
- 没有改动生产文件。`p1/kernel.py` 仍为 v844，SHA-256
  `d1f1e0697c3e99a86f3d34bea6509755be6f70470a2373ec0bd355819c671f9e`。

## 已工作的部分

| 探针 | SID | 结果 | 证明 |
|---|---|---|---|
| 只做 pair 通信 + 常规 replicated 计算 | 149740 | Accepted，display 81.0 | 世界 all-gather 路径不挂 |
| 只做过滤 sort + metadata + 常规 replicated | 149745 | Accepted，display 83.0 | filter/sort/metadata 正确可跑 |
| 半专家 sort + metadata + 常规 replicated | 149752 | Accepted，display 81.83 | E=128 局部计数排序/metadata 可跑 |

## 失败的候选

| 候选 | 思路 | SID | 结果 |
|---|---|---|---|
| v858 | P2P `isend/irecv` 交换本地 FP8/route/scale | 149707 | WrongAnswer 0，NCCL P2P 异常 |
| v859 | `dist.new_group` + pair all-gather | 149710 | WrongAnswer 0，rank2 exit 255 |
| v860 | world `all_to_all_single` 零分片 | 149712 | Pending 后由本 agent 取消，疑似死锁 |
| v861 | world all-gather 输入，但 all-gather 输出形状写错 | 149724 | Pending，取消；根因是 shape 不满足 `world*N` |
| v862 | 修正 `all_gather_into_tensor` 输出形状 | - | 未单独提交 |
| v866-v874 | 半专家权重/局部 metadata/MD host/direct、M pad 对齐等诊断 | 149747 等 | WrongAnswer 0；rank2 exit 255，无 stderr |
| v868 | 半专家 sort+metadata 诊断（不含 MD） | 149752 | Accepted，证明失败点在 MD |

## 当前最可疑点

- pair 通信成功后，把两源 branch 过滤为本地 128 专家，再用现有
  `_fgs_tma1_kernel_gq_tiled` 做 MD 时，rank2 以 exit 255 结束且平台不回传 stderr。
- 常规 replicated c9/c10 使用同一 kernel 特化与同一 batch 配置，唯一差异是
  `A=[2T,H] token-only`、`ORDER` 值域 `[0,2T*k)`、局部 `counts` 长度 128。
- 已排除：collective 形状、world all-gather dtype、clone/视图偏移、`M` 是否 16 对齐。
- 下一手需要平台日志/单卡最小复现，不能继续盲发正式提交。

## 建议

- 暂停 R1 盲提交；先拿到 rank2 的真实异常或做单卡 pair-MD 最小复现。
- v855（c11 静态 DN 尺度）仍是当前唯一已验证 Accepted 的结构增量
  （SID 149680，c11 约 −1%，SQNR 23.10 dB，未跨 q=87）。

## 退出码诊断（2026-09-27 晚）

- 平台日志 5120 字节截断让 print 诊断只能看到 EP2_COMM。
- v896 改为在 metadata 后 `raise SystemExit(100 + num_tiles//100)`，所有 rank 返回
  exitcode 103 ⇒ metadata 的 counts 和 `num_tiles_total` 正常，且输入通信/排序/metadata
  阶段可完成。
- v895 在 pair-MD 后 `torch.cuda.synchronize(); raise SystemExit(102)`，提交后长期
  Pending，由本 agent 取消 ⇒ **pair-MD kernel 执行阶段确实会挂死，不是 metadata 计数错误**。
- 目前 R1 需要单卡最小复现或平台侧 rank2 stderr 才能继续；不再盲发 pair-MD 正式提交。
