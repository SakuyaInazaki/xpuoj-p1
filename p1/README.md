# 当前提交源码

- `kernel.py`：现役 v12，SHA `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`。SID 152238 异常计时完整 AC / raw 89.08，152241 同源码正常 AC / raw 82.00；后续性能实验首选此 SHA 作锚。冻结同内容副本在 [v12 候选](../experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py)。
- `references/kernel_v926_measured.py`：已测原文，SHA `2633cc995eb2563e22c1a212df1256141bdc1b520fe841d4dd4c15c0d382e7bf`，正式 SID 151806/151807；用于回溯 A 晋升前的性能。
- `references/kernel_v890_measured.py`：上一实测锚，SHA `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc`。

reference 不原地编辑。新候选放 `experiments/YYYY-MM-DD/candidates/`，验证后才晋升。接手先读 [最新异常证据与优化任务书](../docs/OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)，再重新核对生产 SHA。本次指导工作未改 kernel；新生成的未测布局对照单独放在 `experiments/2026-10-01/guide-candidates/`。

原来混放的 1552 个历史文件完整迁到 [archive/p1](../archive/p1/README.md)；按旧文件名或 SHA 使用 `python3 scripts/find_candidate.py <query>` 从项目根查找。迁移清单可核对原文与回退位置。
