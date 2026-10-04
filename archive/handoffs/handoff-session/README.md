# 会话交接文件夹：XPUOJ P1 MegaMoE

本文件夹是上一 agent 会话（从接手到 submission 114758）的完整交接材料；
当前 canonical 已由新会话推进到 **114970 / raw 69.33**，`01-current-state.md` 已更新。
阅读顺序（最新完整交接请改看 `../handoff-final/README.md`）：

1. `01-current-state.md`：立即必读。当前成绩、canonical 文件、检查清单。
2. `HANDOFF.md` 第 32 节：冲刺 70 分的最新完整记录。
3. `02-session-timeline.md`：上一会话所有提交和实验的时间线、结论。
4. `03-architecture-and-code.md`：上一会话 canonical（114706）的架构与关键代码位置。
5. `04-pitfalls-and-sandbox.md`：沙箱限制、已踩坑、必须避免的写法。
6. `05-candidates-and-next-steps.md`：候选文件清单、哪些值得重试、下一步优先级。
7. `06-submission-summary.md`：上一会话 37 次提交的逐点 tk/单点分摘要表。
8. `session_submissions_raw.json`：上一会话 37 次提交的原始 detail JSON。

## 10 秒结论

- 最佳提交：**114970，displayScore(raw) 69.33，Accepted**；scoreboard 扣罚后 59.33。
- 当前 `p1/kernel.py` 与 114970 完全一致。
- SHA-256：
  `e8b70ee81ef48e40b491a7729321d6619afff0b81d2195c08fcbbce4d4976270`
- Scoreboard 当前返回 114970 / 59.33。
- 在线 submissionCount 当前 447；扣罚已到上限，新提交不会降低历史最佳。
- 远程无 GPU，所有 Triton/NVSHMEM 行为只能靠平台评测验证；评测机噪声极大，
  同架构分数可在 67~69 之间波动。
