# 最终交接文件夹：XPUOJ P1 MegaMoE（本会话完整记录）

> 生成时间：本会话结束时。当前最佳 submission **114970 / raw 69.33**。
> 阅读顺序：README → 01 → 02 → 03 → 04 → 05 → 06。

## 10 秒结论

- 账号：`dpsk-test`，比赛 ID `13`，problemOrder `1`，语言 `triton-dist`。
- 当前 canonical：`p1/kernel.py` = `p1/kernel_case9_int8_both_clean.py` = submission **114970**。
- SHA-256：
  `e8b70ee81ef48e40b491a7729321d6619afff0b81d2195c08fcbbce4d4976270`
- submission detail raw：**69.33**；scoreboard（扣罚后）：**59.33**。
- 在线 `submissionCount`：447。扣罚已到上限 10。
- 当前距目标 70 分只差 **0.67 raw**。
- 最危险的历史教训：**所有静态权重缓存必须按 shape，绝不能按 `id(tensor)`**。
- 本会话共有 78 次提交：65 Accepted、12 WrongAnswer、1 TimeLimitExceeded。
- 完整逐点原始数据：`handoff-final/submissions_raw.json`。

## 检查清单

```bash
cd /home/sakimi26/xpuoj-p1
sha256sum p1/kernel.py p1/kernel_case9_int8_both_clean.py
python -m py_compile p1/kernel.py
python scripts/best_score.py

# 提交新实验
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

## 文件导航

| 文件 | 内容 |
|---|---|
| `01-current-state.md` | 当前 canonical、114970 逐点、备份文件 |
| `02-competition-and-platform.md` | 题目约束、12 个 shape、评分、API、评测 harness |
| `03-architecture.md` | 当前代码架构、路径选择、低精度 GEMM、缓存设计 |
| `04-all-submissions.md` | 本会话 78 次提交总表 |
| `05-pitfalls-and-sandbox.md` | 所有沙箱限制、踩坑、失败路线 |
| `06-candidates-and-next-steps.md` | 候选文件清单、下一步优先级 |
| `submissions_raw.json` | 78 次提交完整 meta + 逐点 tk/tb/score/error |
