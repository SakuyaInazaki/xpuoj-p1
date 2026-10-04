# 2026-09-08 实验

- `candidates/`：本轮候选源码；文件名应表达唯一改动。
- `results/`：平台原始结果或可复算结构化数据。
- `notes/`：短分析和未提升为 `docs/STATE.md` 的暂定结论。

每个候选至少记录来源 SHA-256、相对 `p1/kernel.py` 的 diff、触达 case、预期信息增益、提交 SID 和终态。当前权威结论只写入 [`docs/STATE.md`](../../docs/STATE.md)，逐 SID 文件入口见 [`notes/results_index.md`](notes/results_index.md)。
