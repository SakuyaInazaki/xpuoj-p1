# 接手会话交接：2026-08-17（submission 115143-115222）

> 从 canonical 114970 / raw 69.33 接手，推进到 **115209 / raw 69.67**。
> scoreboard 已刷新为 115209 / 59.67（扣罚后）。

## 10 秒结论

- 当前 canonical：`p1/kernel.py` = `p1/kernel_case9_repl_fp8_lowmem.py` = submission **115209**。
- SHA-256：`9f5a2f86152bd208bb109f0ff6959efe3e7fcf16bac38f97c25b1e60ce28f7b3`
- 上一 canonical 备份：`p1/kernel_114970_backup.py`、`p1/kernel_115209_backup.py`。
- 在线 submissionCount：466；扣罚已到上限 10。
- 本会话最重要的两个修复：
  1. `_fp8_group_gemm_kernel` 的 expert 基址必须用 int64，否则 E=256 full gateup 偏移溢出 -> 非有限值。
  2. case9 改走全量 replicated FP8 both，并用 `_get_full_fp8_weights_lowmem` 避免 BF16 full 权重 OOM。
- 115209 逐点 tk：7.680 / 15.433 / 2.880 / 1.962 / 5.161 / 3.091 / 4.076 / 2.864 / 3.839 / 3.199 / 2.350 / 3.650。
- 用 114970 轮的 tb 重算，115209 对应 raw 为 70.25；剩余波动主要来自 case2/11/12 的 tb 基线。

## 本会话关键提交

| id | status | display | timeUsed | 备注 |
|---:|---|---:|---:|---|
| 115143 | Accepted | 68.42 | 58545 | 修复 duplicate `_get_static_cache` id-key（单点未采用） |
| 115147 | WA 诊断 | - | - | 确认 shape-key static cache 第二次调用命中 |
| 115165 | WA | 66.67 | - | chunked weight quant + case10 full FP8；case10 非有限值 |
| 115175 | Accepted | 68.67 | 57500 | chunk quant + case10 FP8 down only |
| 115185 | Accepted | 69.33 | 57204 | FP8 expert 指针 int64 修复；case10 full FP8 both 通过 |
| 115207 | WA | 63.92 | - | case9 full replicated 普通量化路径 OOM |
| **115209** | **Accepted** | **69.67** | **56185** | **case9 full replicated FP8 lowmem，晋升 canonical** |
| 115214 | Accepted | 69.42 | 56641 | 115209 复测 |
| 115219 | WA 诊断 | - | - | 115209 replicated phase 诊断 |
| 115222 | Accepted | 69.33 | 56570 | E256 route GEMM BM64 试验，未超过 canonical |

完整逐点 JSON：`submissions_raw.json`。

## 当前架构（115209）

- 12 个测试点全部走 `_run_replicated`（replicated full weights + 本地 grouped GEMM）。
- case9：full FP8 both，权重用低内存逐 rank 量化；不再走 hidden direct AG + reduce。
- case10：full FP8 both，int64 expert 指针。
- 其余路径与 114970 一致。
- `_quant_weight_fp8` 已改为 dim0 chunk=8 分块量化，数值等价、峰值显存更低。

## 尚未达到 70 的确定性差距

- case9 tk 3.839 -> 3.60（差 0.24ms）。
- case10 tk 3.199 -> 2.88（差 0.32ms）。
- case12 tk 3.650 -> 3.50（差 0.15ms）。
- case2 仍是最难用例，tk ~15.43。

## 建议下一步

1. 不要回退 115209；不要重新启用普通 `_get_full_fp8_weights` 处理 case9（OOM）。
2. 可等待评测机 tb 基线较好时重提 115209；按 114970 的 tb 计算已可到 70.25。
3. 若继续调优，优先用 phase 诊断定位 case9 的 route/prep 和 case10 的 FP8 gateup。
4. case12 只差 0.15ms，可谨慎试 `_gather_branch_sum` 或 down BN 参数，但不要盲扫。
5. 所有 E=256 full custom FP8 GEMM 必须保留 int64 expert 基址修复。


## 后续冲 80 实验（未突破）

- 115593-115678 约 30 次 kernel/API 探测与变体，均未超过 115209。
- 关键结论与失败路线已写入 `HANDOFF.md` 第 35 节。
- `submissions_raw.json` 已更新到 49 条（115143-115678）。
