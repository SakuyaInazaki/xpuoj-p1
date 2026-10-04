# XPUOJ P1 MegaMoE 工作环境

## 当前成绩（继续迭代后）

- 平台最佳（scoreboard）：submission **116792**，raw **76.00**，扣罚后 **66.00**。
  - 116792 的 raw 高来自 case5 tb 异常，不代表实际性能。
- 实际性能 base：submission **116882**（v159），timeUsed **43131**（另一次 43198）。
  - 当前 `p1/kernel.py` = v159 = `p1/kernel_116882_backup.py`。
  - 同窗口 v155 对照 43777/43714，v159 稳定快约 0.52-0.65ms。
- 当前优化栈（v159）：
  1. fused gateup orderW + GM16 + s4 + persistent grid132；
  2. case2 SwiGLU amax/quant 也 orderW；
  3. **case2 token gather 改为 custom `order // k` gather，省掉 token_idx[order] + PyTorch gather**；
  4. 所有 activation amax atomic `sem="relaxed"`；
  5. topk_ids 保持 int64；
  6. **E<=16 的 routing GEMM 改 BM64/BN16/BK128/w4**，case1/2 路由更快。
- 历史 base：
  - `p1/kernel_116858_backup.py` = v155（route E8 BM64/BK64）
  - `p1/kernel_116808_backup.py` = v147（case2 custom gather）
  - `p1/kernel_116773_backup.py` = v143（GM16）
  - `p1/kernel_116767_backup.py` = v142
  - `p1/kernel_116754_backup.py` = v140
  - `p1/kernel_116735_backup.py` = v136
- 查询：`python scripts/best_score.py`

## 提交

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

## 实现要点

- 全部 12 个 case 全量权重 replicated，所有专家 GEMM 均为 FP8。
- 非 case2 fused gateup：persistent grid=132，BM128/BN128/BK128/GM16/w8/s4，
  kernel 内 `w = W[ORDER[offs_m]]`。
- case2 token gather：custom `_gather_tokens_from_order`，`row = src // k`，不再生成 token_idx。
- case2 SwiGLU amax/quant：BM128/BN256/w8，同样 `W[ORDER[offs_m]]`。
- 所有 activation amax 的 `tl.atomic_max` 使用 `sem="relaxed"`。
- topk_ids 保持 int64，只 reshape 一次。
- 非 case2 token gather：`row = order // k` + per-row amax。
- FP8 down：persistent TMA-B，grid132，BM128/BN256/BK128/GM8/w8/s3。
- 所有 custom GEMM 无 K/N remainder mask；E=256 expert 基址 int64。

## 已验证的负结果（勿重试）

- fused s5 OutOfResources；fused w16、grid128/160 慢。
- fused orderW GM4 较慢；GM24 结果不稳定，不采用。
- case2 full-fused 仍 TLE；case2 row SwiGLU/down 慢。
- case2 gather+amax（atomic / partial 两种）无稳定收益。
- fused persistent M-only / contiguous-M batched amax：SQNR 错或明显慢。
- metadata num_sms=132：慢。
- packed sort：tensor.sort 和 functional torch.sort 均被 sandbox 拦截。
- 中间张量全局 buffer cache：TLE/WA。
- TMA descriptor cache、case2 binned amax、in-place weight norm：无稳定收益。


最新完整交接：`handoff-session-20260818-final4/`（submission 116594-116897 共 101 次）。
