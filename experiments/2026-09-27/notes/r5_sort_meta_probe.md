# R5 排序 / metadata 融合探针（2026-09-28 凌晨）

生产保持 v890：`p1/kernel.py` = `p1/kernel_v890_fused_delta.py`，
SHA-256 `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc`。
本轮未晋升任何候选。

## v901 fused scan-base（历史候选复核）

- 文件：`experiments/2026-09-27/candidates/v901_scan_offsets_sort.py`
- 历史单发 SID 150008 Accepted display 82.75，但 c11/c12 处于已知异常低窗。
- 同窗 AB 复核：cand SID 150920 vs anchor 150917，display 81.67 vs 82.67。
  - c3 +7.3%、c4 +8.7%、c5 +3.8%，c11/c12 无正收益。
- 结论：`_csort_scan_base_kernel` 用 O(E*N) 扫描换 launch，完整案负收益，关闭。

## v902 E<=64 全 HIST 融合 offsets/delta

- 文件：`experiments/2026-09-27/candidates/v902_fused_offsets_delta.py`
- 第一对：cand SID 150923 display 82.33 vs anchor SID 150926 81.75；
  但 cand 的 c11 0.767 ms、c12 1.374 ms 与历史分布不符，属于低窗；
  第二对 AB：anchor SID 150934 81.83 vs cand SID 150936 81.75。
  - 第二对 c3 +0.5%、c4 +1.0%、c11 −0.3%、c12 −0.3%，其余在噪声内。
- 结论：只省一个 metadata launch 的微效应，无法稳健超过 0.3% 噪声，不晋升。

## v903 hist_tot + 全 E 融合（TLE）

- 文件：`experiments/2026-09-27/candidates/v903_hist_tot_fused_offsets.py`
- SID 150928 `TimeLimitExceeded`，第一个 testcase 的 oracle/determinism 已通过，
  推测新增 atomics/fused 特化把含 JIT 编译的 500 s 总预算推爆；不重发。
- 结论：关闭。

## v904 c11/c12 metadata 融合

- 文件：`experiments/2026-09-27/candidates/v904_fused_meta_c1112.py`
- 把 build_block_row_idx_info 的 tile metadata 写进 `_csort_offsets_delta_kernel`
  的 c11/c12 专用变体，只去掉一个 metadata launch。
- 第一对：cand SID 150941 display 84.50 vs anchor SID 150942 81.92；
  但 c11 tk=0.096 ms、c12 tk=0.000 ms，是平台计时异常，不可作为收益。
- 第二对 AB：anchor SID 150943 81.83 vs cand SID 150945 81.75；
  c11 −0.34%、c12 +0.27%，仍未超过噪声。
- 结论：机制正确且 Accepted，但效应太小；不晋升。

## 当前判断

- `p1/kernel.py` 保持不变。
- 排序/metadata 子链的 launch 数已接近收益下限；继续微调只会落入 ±0.3% 噪声。
- 下一步高价值方向仍是 c9/c10 的 pair-EP2。已有两卡通信、过滤排序和半专家 metadata
  可跑通，卡在 `_fgs_tma1_kernel_gq_tiled` 对拼接 A / 过滤 order 的执行阶段。
  需要单卡最小复现或 rank2 stderr，不能继续盲发完整 EP 候选。
