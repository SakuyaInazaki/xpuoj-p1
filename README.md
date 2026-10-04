# XPUOJ P1 工作区

先读 **[本轮最新状态](experiments/2026-10-01/session-2156/STATUS.md)**，再读 **[详细优化任务书：异常证据、受控探针与结构优化](docs/OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)** 并核对实际源码 SHA。目标是在 **2026-10-01 23:59（Asia/Shanghai）前争取 raw 90 / net 80**，目前尚未达到。旧“CUPTI 缓冲必然耗尽、增加辅助 launch 确保破 90”的推断已撤销，不再作为执行依据。

**最新实测（2026-10-01 17:11:44，Asia/Shanghai）**：P1榜分 **79.75 / SID153151 / raw89.75**，12案完整AC；q总和1077，距raw90差3。c10–c12精确零、c8–c12低值，c1 tb=40.455ms异常上浮；源码只是c67等价副本，不能称结构提速。生产已为153151对应SHA `f9ca009609bc…`；旧正常v12锚`08dd08eb51b1…`仍冻结保留。**先看[本轮状态与未测候选](experiments/2026-10-01/session-2156/STATUS.md)**；J0/J1/J2/J12均已正常完整AC复测，纯布局路线关闭。历史锚152238/raw89.08、同源码152241/raw82.00及152976/raw89.00保留供对照。

## 当前源码和实测锚

| 文件 | SHA-256 | 状态 |
|---|---|---|
| [p1/kernel.py](p1/kernel.py) | `f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a` | 当前生产；153151完整AC/raw89.75；c67等价副本，异常高分不能当正常结构收益 |
| [旧正常v12锚](experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py) | `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9` | 冻结保留；152238/raw89.08，正常同码152241/raw82.00；S1/S2候选基底 |
| [v926 实测快照](p1/references/kernel_v926_measured.py) | `2633cc995eb2563e22c1a212df1256141bdc1b520fe841d4dd4c15c0d382e7bf` | SID 151806 / 151807 完整 AC；用于回溯 A 之前的性能 |
| [v890 回退快照](p1/references/kernel_v890_measured.py) | `436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc` | c10 static DN 晋升前的已测版本 |

旧方向 **A 已晋升**：GQ 生成 AH/WI/ACT_SCALE，MD 消费，c11/c12 两对约 1.6%–2.4% 收益，随后扩至 c3–c8，并清理 c2 无用分配。c3–c8 的 `_GA` 调用支路仍会绕过 A。旧方向 **B 已试且无可辨收益**（152488）；v13 的 c1/c2 pre、v15 的 c9 pre 也没有正常结果支持晋升。详见 [A/c2 记录](experiments/2026-09-30/notes/dirA_c2_promoted_20260930.md)、[B 负结果](experiments/2026-09-30/notes/dirB_wide_negative_20260930.md)。

当前状态：**S2b153681完整AC/raw81.08，无zero/low，c4=1.456ms较父S2退化约85%，关闭flatten推广。** 第9df5c修正版已SID153725成功提交，实际9份、暂停自动新增；C356U153682已完整AC/raw82.00、无zero/low且无≥2%正常信号；当前153686/153707/153725三份在途，既有组合不取消，先前8发自定上限撤销。最佳仍153151 raw89.75/net79.75，生产f9ca不变。唯一执行者platform_sol；采用仍须完整12AC/双SQNR/确定性。精确SHA及独立review见[本轮STATUS](experiments/2026-10-01/session-2156/STATUS.md)。

异常研究更新为 **283 发连续记录**：首次 SHA 完整 AC 130 发，其中低值 17 发、精确零值 9 发；重复 SHA 的 77 发 AC 均无低值。21 个 SID 的日志出现 `torch.profiler` cycle 警告，但警告本身不证明计时 bug。回溯证明线上会改写 host 源码，不能把提交行号直接当 JIT 实际缓存身份。已有 44 个中性探针无命中，不再独立随机改注释/名称。当前没有稳定 tk=0 触发方法。

## 接手顺序

1. [AGENTS.md](AGENTS.md)：用户禁令与目标；禁止使用 `deepseek-brainstorm`。
2. [最新任务书](docs/OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)：题目与新信息、283 发证据、积分账、J/S1/S2/S0、验证和停止条件、可直接分派的任务卡。
3. [源码导航](docs/CODE_MAP.md)与[状态摘要](docs/STATE.md)：当前 v12 及实测锚，行号绑定 SHA。
4. [提交说明](docs/SUBMISSION.md)：仅 OJ 正式/允许的 custom；统一一个平台执行者。用户自行分派 coding agent。
5. [平台快照](reports/2026-10-01-platform-readonly.json)、[连续日志](reports/2026-10-01-cohort-readonly.json)、[逐发台账](reports/2026-10-01-ledger.csv)：按 SHA 和逐案结果核对，不沿用旧交接的“当前”。
6. [未测布局候选](experiments/2026-10-01/guide-candidates/README.md)：由本轮生成，尚未提交；每个候选的变化与预算见任务书第 5 节。

旧 [9 月 30 日任务书](docs/OPTIMIZATION_GUIDE_FINAL_SPRINT_2026-09-30.md)、[旧 10 月 1 日指引](docs/OPTIMIZATION_SPRINT_AND_ANOMALY_GUIDE_2026-10-01.md)、[v926 指引](docs/OPTIMIZATION_GUIDE_V926_2026-09-30.md)与 [187 发异常指南](docs/PLATFORM_TIMING_ANOMALIES_2026-09-30.md)保留作历史，**不再定义待办**。题目明确要求切换测试点后更新静态缓存；现代码的调用序号、缓存键和 static scale 边界仍需独立审计。最新指导工作没有修改生产 kernel，也没有新增正式/custom 提交。

## 目录职责

| 路径 | 用途 |
|---|---|
| `p1/` | 当前 kernel、两个不可改写的已测快照和说明 |
| `experiments/YYYY-MM-DD/` | 各轮独立候选、结果、notes |
| `docs/` | 当前指引、状态、源码导航、提交约定 |
| `scripts/` | 提交/查询工具、源码清理、候选查找 |
| `reports/` | 审计快照、积分模型、CPU 证明；保留已有报告原位 |
| `archive/p1/` | 从 p1 迁出的 1552 个历史文件，内容未改 |
| `archive/guides/` | 四份旧指引，供追溯，不定义当前优先级 |
| `archive/historical-docs/` | STATE / CODE_MAP 原文快照 |
| `logs/`、`benchmarks/`、`sandbox/` | 原始记录与既有工具 |
| `比赛信息/`、`1-full.md` | 题目原始资料；最新补充原文副本在 reports |
| `work/repo/` | 本地依赖，保持原位 |
| `.secrets/` | 本地凭据，勿输出、移动或提交 |

目录不是 Git 仓库。历史文件未删除；迁移索引与回退说明在 [archive/README.md](archive/README.md)，逐文件 SHA 见 [1556 项迁移清单](archive/WORKSPACE_MOVES_2026-09-30.csv)。旧文相对链接可能仍指迁移前位置，可按文件名或 SHA 查找：

```bash
python3 scripts/find_candidate.py kernel_v800
python3 scripts/find_candidate.py 436f0a227678
```
