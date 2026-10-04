# 方向 B：c9/c10 宽 GU 单累加器试验

候选：`experiments/2026-09-30/candidates/p1_dirB_c9c10_v14_meas.py`
基线：生产 v12 SHA `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`

## 实现

- 新增 `_get_full_fp8_weights_lowmem_tiled_wide`：GU 改为 `[E,R,KT,128,2,128]`，scale 改为 `[E,R,128,2]`。
- 新增 `_fgs_tma1_kernel_gq_tiled_wide`：单个 `[128,256]` accumulator，`[256,128]` B_DESC load。
- 新增 `_fgs_tma1_host_tiled_wide`，并在 c9/c10 分支替换旧 tiled 路径。
- DN 与最终 output 路径不变。

## 结果

- SID 152485：首次 SHA 提交 TLE。
- SID 152488：同码重复完整 AC。
  - c9 `2.448 ms`，c10 `1.869 ms`，与基线 c9 ~2.45 / c10 ~1.85 无可分辨收益。
  - 其他案未回退，SQNR 保持正常。
- 结论：该宽 GU 单累加器版本不晋升，生产仍保留 v12；后续若继续 B，需要改 pack layout 之外的 epilogue/TMA store 或更深 pipeline，而不是仅合并 B load/accumulator。
