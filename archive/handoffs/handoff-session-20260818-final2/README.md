# XPUOJ P1 接手后续（submission 116594-116643）

## 10 秒结论

- **晋升新 base：116627 / `p1/kernel_v115_persist_fused_order_token.py`，当前 `p1/kernel.py` 已是该文件。**
- 116627：Accepted，raw 75.00，timeUsed **42753**，scoreboard total **65.00**（扣罚 10 已到上限）。
  - 复测 116633：raw 74.33，timeUsed 43220。
  - 同窗口旧 base 116310 复测：116620=44630、116637=44625，v115 稳定快约 1.4-1.9ms。
- 两个有效改动：
  1. 非 case2 fused gateup 改为 persistent grid=132（每 tile 仍 immediate atomic row-amax）。
  2. 非 case2 token gather 直接用 `order // k` 推导源 token row，
     省掉 `token_idx = token_idx[order]` 一次索引 kernel。
- 注意 116627 的 case8 tb=11.282 是异常；真实性能看 timeUsed/tk。

## 关键提交表

| id | 文件 | 说明 | raw | timeUsed |
|---:|---|---|---:|---:|
| 116594 | kernel_v109_order_gather.py | order-derived token row + weights 融合进 gather（非 persistent） | 74.42 | 43018 |
| 116600 | kernel_v110_order_token_only.py | 仅 order-derived token row（非 persistent） | 74.50 | 42952 |
| 116605 | kernel_diag_v98_phases.py | 当前 v98 逐 phase CUDA-event 诊断（WA，读 userError） | 0 | 160409958 |
| 116607 | kernel_v111_fused_persist_tiles.py | fused gateup persistent tile-loop 全量 | 74.17 | 43610 |
| 116612 | kernel_v112_gather_monly.py | token gather M-only（每个 program 循环全部 H block） | 73.42 | 44539 |
| 116616 | kernel_v114_persist_case1.py | 仅 case1 persistent fused | 73.00 | 44566 |
| 116620 | kernel.py（旧 116310） | **同窗口 base 对照** | 73.25 | 44630 |
| 116622 | kernel_v111_fused_persist_tiles.py | persistent fused 复测 | 73.92 | 44086 |
| 116626 | kernel_v110_order_token_only.py | order token 复测 | 74.00 | 43462 |
| 116627 | kernel_v115_persist_fused_order_token.py | **persistent fused + order token，新 base** | **75.00** | **42753** |
| 116629 | kernel_v116_persist_fused_order_weight.py | v115 + weights 融合进 gather | 74.17 | 43180 |
| 116633 | kernel_v115_persist_fused_order_token.py | **新 base 复测** | 74.33 | 43220 |
| 116637 | kernel.py（旧 116310） | **同窗口 base 对照 2** | 73.08 | 44625 |
| 116639 | kernel_v117_case2_order_gather.py | v115 + case2 custom order gather | 73.83 | 43725 |
| 116643 | kernel_v118_case2_order_div.py | v115 + case2 `order // k` | 73.50 | 44377 |

## 有效改动实现位置（以 `p1/kernel.py` 为准）

- `_fused_gateup_swiglu_kernel_rowA_persistent_tiles`：
  grid=132，`for tile_id in range(pid, total_tiles * num_block_n, num_pid)`，
  其余 swizzle/dot/row-amax 与原 non-persistent 完全一致。
- `_gather_tokens_row_amax_order_kernel` / `_gather_tokens_row_amax_order`：
  2D grid 不变（BM128/BH128），kernel 内 `token_row = src // K_BRANCH`；
  host 不再创建/permute `token_idx`，但 `flat_weights = flat_weights[order]` 保留独立执行。

## 负结果（不要重试）

- fused gateup persistent M-only / contiguous-M batched row-amax：仍 SQNR 错。
- token gather M-only：几乎无收益（116612 vs base 对照约 -0.09，属噪声）。
- weights 融合进 gather kernel（pid_h==0 store）：慢（116629）。
- case2 custom order gather / `order // k`：无收益或更慢（116639/116643）。
- v109/v110 首次提交与 v111/v115 跨窗口比较不可靠；本会话所有结论以同窗口对照为准。

## 建议下一步

1. 继续在 v115 上做同窗口 A/B，任何新实验旁边都先/后投 `p1/kernel.py` 对照。
2. 可试 persistent fused gateup 的 grid=96/160、num_stages=2/4。
3. case2 仍是最大单点（v115 两次 10.11/10.34），但本次两个 case2 微改均不成立。
4. 不要用单次 raw/tb 判断；116627 case8 tb 异常就是例子。

## 补充（v119）

- 116646 `kernel_v119_case2_fused_rowA.py`：在 v115 上尝试把 case2 也走
  persistent fused gateup（global token-A scale + per-row activation amax），
  再次 **TLE**（timeUsed 473140376）。case2 full-fused 方向确认关闭，
  即使 K constexpr + persistent + per-row amax 也不行。
