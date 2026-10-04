# 06 候选文件与下一步

## 当前 canonical

- `p1/kernel.py` = `p1/kernel_case9_int8_both_clean.py` = **114970 / 69.33**。
- 不要用任何低分候选覆盖它。

## 本会话最重要的候选文件

| 文件 | 相关提交 | 结果 | 建议 |
|---|---:|---|---|
| `p1/kernel_case9_int8_both_clean.py` | 114970 | **69.33** | 当前 canonical |
| `p1/kernel_case9_int8_both.py` | 114966 | 68.58 | raw tk 总和更低（57859），display 被 tb 噪声压住；值得在 tb 正常轮次重试 |
| `p1/kernel_best_case9_fp8both.py` | 114971 | 68.75 | case9 单点 4.936ms 略好；可与 case10 FP8-down 再组合 |
| `p1/kernel_best_case10_fp8down.py` | 114972 | 67.33 | case10 FP8 down（3.88ms）比 INT8 down 略好，但总分受 case9 噪声拖累 |
| `p1/kernel_hybrid_int8_case3_case12fp8.py` | 114955 | 67.50 | case3 2.90ms、case12 3.73ms；这两个 FP8 扩展是真实收益 |
| `p1/kernel_hybrid_int8_gateonly_case10down.py` | 114951 | 68.17 | case1/2/10 都很好；case9 4.88ms；离 canonical 很近 |
| `p1/kernel_int8_tiled.py` / `_clamp.py` | 114969/114974 | WA | per-tile INT8 非有限值；保留反例 |
| `p1/kernel_case9_int8down.py` | 114965 | 68.67 | case9 BF16 gateup + INT8 down |
| `p1/kernel_fp8_swiglu_quant.py` | 114913 | 68.33 | 融合 SwiGLU+FP8 quant 的原始版本 |
| `p1/kernel_case10_int8down.py` | 114962 | 68.50 | case10 INT8 down 版本 |

## 已知“不要重复提交/不要继续”的路线

- E256 full replicated：114864 TLE。
- E256 full FP8/INT8 权重量化：OOM。
- per-tile INT8：非有限值。
- case9 路由打包 NCCL：114925 case9 39ms。
- case9 route direct flat AG：114959 case9 8.9ms，不如 NCCL。
- E256 hidden AG NCCL：114885 case9 12.5ms，不如 direct c32。
- E256 hidden AG c16/c24/c64、w4：均不如 c32/w8。
- k>4 单次 token gather：连续更差。
- tensor.index_select / 自写 Triton 行 gather：被禁或未胜出。

## 达到 70 分的剩余差距（基于 114970）

| case | 当前 tk | 70 分所需约 tk | 差距 | 主要瓶颈 |
|---:|---:|---:|---:|---|
| 2 | 15.322 | 12.34 | -2.98ms | INT8 gateup / FP8 down |
| 9 | 5.022 | 3.61 | -1.41ms | route/gather/sort/reduce |
| 10 | 3.963 | 2.89 | -1.07ms | BF16 gateup |
| 12 | 3.662 | 3.52 | -0.14ms | 最终 prep/gather |
| 3 | 2.877 | 2.99 | 已达标 | — |

## 建议下一步（按性价比）

1. **稳定复现 114970/114966 架构在 case9 正常轮次的表现**。
   - 优先重试 `kernel_case9_int8_both.py`（raw tk 总和最低）或把 114970 与 114971 的 case9 FP8-both 组合。
   - 每次提交前确认 case9 近期噪声水平。
2. **case9 reduce/route 专项**：
   - 用 `_diag_mark` 对 114970 的 case9 再做一次 phase 诊断。
   - 重点看 reduce_scatter 和 sort；尝试 BF16 partial 的直接 NVSHMEM reduce 或更优的 all-to-all reduce。
3. **case10 gateup**：
   - 当前 BF16 gateup 是 3.96ms 的主要部分；需要 local H800 profile。
   - 可试 full down FP8/INT8 与不同 BN/BK；不要 full gateup 低精度（OOM）。
4. **case2 gateup**：
   - 当前 INT8 per-row + BN256/BK128 是最佳已知；BK256 OOR。
   - 值得在有本地环境后试 `BLOCK_M=64+BK256`、双 expert 分组、或更优 INT8/FP8 kernel。
5. **清理 canonical 的重复历史代码**：
   - `p1/kernel.py` 第 1488 行附近有重复的旧 helpers；Python 后定义覆盖先定义，提交正确。
   - 只有在准备重新提交并验证 hash/逐点后，才做“仅清理、不改逻辑”的候选。
6. **不要再做**：
   - 无本地 GPU 的 tile 盲扫；
   - 重复提交同一文件碰波动；
   - 任何会破坏 SQNR/determinism 的激进量化。

## 交接检查

```bash
cd /home/sakimi26/xpuoj-p1
sha256sum p1/kernel.py p1/kernel_case9_int8_both_clean.py
python -m py_compile p1/kernel.py
python scripts/best_score.py
# 应显示 totalScore=59.33, submissionId=114970
```
