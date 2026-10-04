# 历史交接：XPUOJ P1 MegaMoE（截至 submission 116324）

> **更新：本会话完整最终交接已整理到 `../handoff-session-20260818-final/`，请优先阅读该文件夹。**

> 本文件夹汇总从 agent 接手（submission 115143 之前的会话）到当前（116324）的全部工作；
> 覆盖多轮 agent 提交；最近两轮摘要见 `07-takeover-20260818-2.md`、`08-takeover-20260818-3.md`。
> 上一个交接见 `handoff-session-20260817-final/`，本 agent 三个阶段的临时交接见
> `handoff-session-20260817-final2/`、`handoff-session-20260817-final3/`。

## 10 秒结论

- 比赛：XPUOJ contestId=13, problemOrder=1, language `triton-dist`；账号 `dpsk-test`。
- scoreboard 最佳：**116142，raw 74.75，扣罚后 64.75**（仍未变）。
- **实际性能 base（当前 kernel.py）**：**116310，timeUsed 42881，复测 43524**。
  主要结构：FP8 down B 走 host TensorDescriptor TMA + token/activation per-row FP8 scale。
- 旧稳定 base 仍保留：115907（timeUsed 43538）。
- submissionCount：633；扣罚 `min((attempt-100)*0.1,10)` 早已到上限 10。
- 从接手时 115705（raw 71.67 / 61.67）到稳定版 115907（74.00 / 64.00），
  主要收益来自：
  1. FP8 down persistent kernel launch bug 修复；
  2. custom GEMM 去掉 K/N remainder mask；
  3. FP8 activation amax 自定义 Triton kernel；
  4. case2 gateup INT8 -> plain FP8；
  5. case4/6/11 也切 FP8；
  6. 排序 gather 与 activation amax 融合。

## 阅读顺序

1. `01-current-state.md`
2. `02-competition-and-platform.md`
3. `03-architecture.md`
4. `04-session-timeline.md`
5. `05-pitfalls-and-sandbox.md`
6. `06-candidates-and-next-steps.md`
7. `07-takeover-20260818-2.md`（116130-116195 第二轮接手会话）
8. `08-takeover-20260818-3.md`（116215-116324 第三轮，TMA / per-row FP8）
9. `submissions_raw_all.json`（115818-115966 共 61 次提交 meta + 逐点 tk/tb + userError tail）

## 接手检查清单

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
# scoreboard 应输出 totalScore 64.75, submissionId 116142

sha256sum p1/kernel.py p1/kernel_116310_backup.py p1/kernel_116142_backup.py p1/kernel_115907_backup.py
python -m py_compile p1/kernel.py p1/kernel_115907_backup.py
```

关键 SHA-256：

```text
2c8204026ecea36158b62d2a04c566918a4c7e4c0f8d36b8cac42f8cb98f71f7  p1/kernel.py (=116310, 实际性能 base)
2c8204026ecea36158b62d2a04c566918a4c7e4c0f8d36b8cac42f8cb98f71f7  p1/kernel_116310_backup.py
31321bb0116b509203536914e8dc57a03a6b68ff45f235baedea2dc49397d2f4  p1/kernel_116142_backup.py (=116142, scoreboard best)
52a8533badbddceb1ce6ef4cc9548443d4f3e5662e37a23b67f3545d9306032a  p1/kernel_115907_backup.py (=115907, 稳定性能 base)
```

提交新实验：

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

解析 submission detail 逐点 tk/tb 的方法见 `02-competition-and-platform.md`。
