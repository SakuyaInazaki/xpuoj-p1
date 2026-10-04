# 会话最终交接：XPUOJ P1 MegaMoE（2026-08-17 会话，submission 115143-115800）

> 本文件夹覆盖本 agent 会话从接手（114970 / 69.33）到当前（115705 / 71.67）的全部尝试。
> 上一个 agent 的交接在 `handoff-final/`；本会话之前的另一段交接在 `handoff-takeover-20260817/`（旧，已被本文件夹取代）。

## 10 秒结论

- 比赛：XPUOJ contestId=13, problemOrder=1, language `triton-dist`；账号 `dpsk-test`。
- 平台当前最佳 submission：**115705，raw displayScore 71.67**；scoreboard 扣罚后 **61.67**；submissionCount 523。
- 平台最佳代码：`p1/kernel.py` = `p1/kernel_gm2.py` = submission 115705。
- **实际最快/推荐继续迭代的代码**：`p1/kernel_fused_gateup_swiglu.py` = submission **115738**（raw 71.08，但 timeUsed 51044，比 115705 快约 2.6ms）。
  - 115705 的 raw 更高主要来自 case6 tb=45.559 的异常基线；其真实 tk 反而更慢。
- 本会话最重要的三个技术突破：
  1. 自研 grouped GEMM 的 `expert` 基址必须 int64，否则 E=256 full gateup int32 溢出。
  2. case9/10 全量 FP8 replicated + low-memory 权重量化。
  3. custom FP8/INT8 grouped GEMM 增加 `tl.swizzle2d`（GROUP_M=8），并融合 FP8 gateup epilogue 到 SwiGLU+amax。
- 离 raw 80 仍差约 8.33 分；当前最快代码 12 点 tk 总和约 51.0ms，80 分粗估需约 28ms。

## 阅读顺序

1. `01-current-state.md`
2. `02-competition-and-platform.md`
3. `03-architecture.md`
4. `04-session-timeline.md`
5. `05-pitfalls-and-sandbox.md`
6. `06-candidates-and-next-steps.md`
7. `submissions_raw.json`（76 次提交完整 meta + 逐点 tk/tb/error tail）

## 接手检查清单

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
# 应输出 115705 / 61.67

sha256sum p1/kernel.py p1/kernel_gm2.py p1/kernel_fused_gateup_swiglu.py p1/kernel_115738_backup.py
python -m py_compile p1/kernel.py
```

关键 SHA-256：

```text
d6567ac008b47e22e69a47ae3b8e883618f23542600b17084cc85490061a6ac3  p1/kernel.py (=115705)
d6567ac008b47e22e69a47ae3b8e883618f23542600b17084cc85490061a6ac3  p1/kernel_gm2.py (=115705)
a0541ce5b8a477f4580dada0c4e3bbf6efed0c475552bc1cf2721c19058278b9  p1/kernel_fused_gateup_swiglu.py (=115738)
a0541ce5b8a477f4580dada0c4e3bbf6efed0c475552bc1cf2721c19058278b9  p1/kernel_115738_backup.py
c2f2471912b9adf606a809fe163e3aa92fd36d66f7972b7f0b323fb42b8058e4  p1/kernel_115696_backup.py (=115696, INT8+FP8 swizzle)
08a9c46ae74861dcf887e2dc6567eb0dacd44b7f8e21727c48800efd0c4d2d45  p1/kernel_115691_backup.py (=115691, FP8 swizzle)
9f5a2f86152bd208bb109f0ff6959efe3e7fcf16bac38f97c25b1e60ce28f7b3  p1/kernel_115209_backup.py (=115209, 本会话早段 canonical)
```

提交新实验：

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

注意：本会话 submissionCount 从 447 增至 523，扣罚早已到上限 10；低分/WA/TLE 不会降低历史最佳。

## 附加材料

- `probes/official_moe_grouped_gemm_kernel_src.txt`：官方 `moe_grouped_gemm_kernel_nk_const` 源码。
- `probes/build_block_row_idx_info_kernel_src.txt`：官方 metadata 构建 kernel 源码。
- `probes/kernel_probe_*.py`：本会话使用的合法探测文件模板。
