# R2 静态权重界 DN 输出尺度：c11 初筛记录（2026-09-27）

## 结论

- R2 的静态 L1 界在 c11 上数值可行，但完整 P1 只测到约 **1.0%** 的 c11 改善，低于任务书“完整案 >=2%”继续门，未晋升生产。
- 生产文件保持 v844：`p1/kernel.py`，SHA-256
  `d1f1e0697c3e99a86f3d34bea6509755be6f70470a2373ec0bd355819c671f9e`。
- 最佳可用候选为 `v855_c11_static_dn_scale_tight.py`，SHA-256
  `63117da967202f46e6694e9c07fbe74e673aeeec620bcc34d70367022a613d34`，
  SID 149680，Accepted，c11 tk 0.876 ms、SQNR 23.10 dB。
- 候选命中了 c11 的静态尺度路径，但未把 c11 推过 q=87 的稳定门槛（约 0.846 ms）。

## 机制

在 c11 的 padded DN host 前增加静态 cache：

```text
L[e,p] = max_{n in chunk p} sum_i abs(Dq[e,n,i] * D_scale[e,n])
C[e,p] = 1.1 * L[e,p]
B_SCALE_folded[e,n] = D_scale[e,n] / C[e,p(n)]
```

静态 kernel 直接执行 `q = fp8(dot(Aq, Dq) * B_SCALE_folded)`，并写
`CSCL = A_scale * C`。它省掉原 epilogue 的逐行 `tl.max(abs(acc))`、比例除法
和一次行向量乘；DN 主循环、padded TMA store、CSCL 形状和 final gather 未改。
只对 c11 key `(65536,1024,32,1024,2)` 启用；未知 shape fallback 保持原路径。

## 精度

- c11 SQNR 从 23.31 dB 降到 23.10 dB，仍高于 22.5 dB 内部保护门；确定性通过。
- 失败版本 v854 用 `C=4*L` 时 c11 SQNR 19.92 dB，说明保守尺度会把 q 压进
  FP8 次正规区；降到 `1.1*L` 后恢复。
- TorchProxy determinism sandbox 拒绝 `torch.clamp_min`。v855 helper 只用
  `torch.maximum`、`torch.full_like`、切片/归约等基础算子。
- v856 进一步给 c3 打开静态路径，c3 SQNR 降到 22.60 dB，且 c3 稳定 tk 没有
  跨过 q=84；不建议保留 c3 扩展。

## 关键 SID

| SID | 候选 | 状态 | 说明 |
|---|---|---|---|
| 149674 | v853 | WrongAnswer | `torch.clamp_min` 被 determinism TorchProxy 拒绝 |
| 149676 | v854 | WrongAnswer | `C=4L` 下 c11 SQNR 19.92 dB，次正规下溢 |
| 149680 | v855 | Accepted | c11 static，c11 0.876 ms / 23.10 dB |
| 149684 | v844 anchor | Accepted | 同窗 anchor，c11 0.885 ms / 23.31 dB |
| 149687 | v856 | Accepted | c3+c11 static，c11 异常 0.772 ms，未复现 |
| 149690 | v844 anchor | Accepted | 第二 anchor，c11 0.884 ms / 23.31 dB |
| 149695 | v856 repeat | Accepted | c11 0.876 ms，确认 0.772 为快窗 |

## 停止判断

- c11 完整改善约 1.0%，低于 R2 完整案 >=2% 门。
- c3 扩展没有稳定跨档，并显著降低 SQNR 余量，停止。
- 不把 v855/v856 晋升生产；v855 可作为后续“已通过全部 P1 正确性”的 c11
  增量候选，与更强机制组合时再评估。

## 更新：静态 DN 扩展到 padded 八案并晋升 v882

- `v877_allpadded_static_dn.py` 首次把静态 DN 输出尺度扩展到所有 direct INV_PAD
  padded 案（c3/c4/c5/c6/c7/c8/c11/c12），SID 149768 Accepted。
  - 同窗 anchor SID 149772 配对：c4 −2.7%、c6 −1.7%、c11 −2.4%、
    c8 −1.4%、c7 −0.8%、c12 −1.1%、c5 −0.9%、c3 −0.4%。
  - c5 SQNR 22.48 dB、c7 22.57 dB，低于 22.5 dB 保护余量；不采用全开版本。
- `v878_static_dn_safe_cases.py` 关闭 c5/c7 的静态 DN，保留其余六个 padded 案。
  SID 149773 Accepted，所有 SQNR ≥22.70 dB。
- `v879/v880/v881/v882` 做了 c6/c4 的短 K host 微调。
  - v882 = v878 + c4 `num_stages=3` + c6 原配置。
  - SID 149784 Accepted，display 82.25；同窗 anchor SID 149772：
    c4 −1.6%、c6 −1.7%、c11 −1.8%、c8 −0.5%、c3 −0.6%，
    c6 在 tb≥7.62 ms 的窗口多次显示 q=86。
- 已晋升：
  - `p1/kernel.py` = `p1/kernel_v882_padded_static_dn.py`
  - SHA-256 `f54ff731cb5ae8193e992c9cddf42f0390b14bc6add1d18ced300db9d956af69`
  - 旧生产 v844 保留为 `p1/kernel_v844_packed_compact_c910.py`。
- 未晋升 v877：c5/c7 SQNR 余量不足；需要更大机制才能把 c6 稳定推过 q=86，
  并把 c4/c11/c12 推向下一档。

## c6 微调补充（负结果）

- `v884_static_dn_c6_s2.py` 把 c6 静态 DN 的 `num_stages` 从 4 改成 2。
- SID 149911 Accepted，但 c6 `tk=1.425 ms`，明显慢于 v882 的约 `1.241–1.247 ms`。
- 结论：c6 静态 DN 保持 `num_stages=4`，不采用 s2。c6 的稳定 q=86 仍差约 1.2%。

## R4 小项：融合 PAD_DELTA（已晋升 v890）

- `v889` 首次把 `_pad_delta_kernel` 并入 `_csort_offsets_kernel`；但 delta 公式写成了
  单 expert 对齐，导致所有 direct INV_PAD 案 SQNR 崩溃（SID 149925 WrongAnswer 26.08）。
- `v890` 修正为全局前缀差：
  `tile_start = sum_{e<pid} ceil(tot[e]/128)`，
  `delta = tile_start*128 - sum_{e<pid} tot[e]`。
- SID 149930 Accepted，display 82.42；同窗 v882 anchor SID 149932：
  c3 `1.335 vs 1.364`、c4 `0.779 vs 0.793`、c6 `1.241 vs 1.246`、
  c11 `0.865 vs 0.872`、c12 `1.459 vs 1.468`。
  SID 149935 repeat 为 `c3 1.347 / c4 0.789 / c6 1.246 / c11 0.868 / c12 1.471`，
  c3/c4 增益部分收敛。
- 晋升：
  - `p1/kernel.py` = `p1/kernel_v890_fused_delta.py`
  - SHA-256 `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc`
  - v882 保留为回退。

## c3/c4 final gather BLOCK_T 试验（负结果）

- `v892_c3c4_fin_bt64.py` 把 c3/c4 的 final gather `BLOCK_T` 从 32 改成 64。
- SID 149943 Accepted，但 c3 `1.363 ms`、c4 `0.793 ms`，没有改善，且 c3 明显慢于 v890 的 1.335–1.347 ms。
- 结论：维持 final gather `BLOCK_T=32`。

## c6/c3 MD 微调负结果

- v893 改 c3 TMA-md `GROUP_M 32→16`，实际 c3 走 `_g` md 分支，改动未命中；SID 149947
  c3 1.356 ms，无收益。
- v892 c3/c4 final gather `BLOCK_T 32→64`，SID 149943，c3 1.363 ms，退化。
- 结论：c3/c6 剩余 ~0.4–0.7% 不能靠当前已试的 launch 参数补齐，需要再消一个 launch
  量级的结构变化。
