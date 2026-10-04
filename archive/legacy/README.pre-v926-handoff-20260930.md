# XPUOJ P1 工作区

这是本仓库的唯一开始入口。不要从根目录旧交接、会话转储或历史报告推断当前状态。

**2026-09-30 续优化更新：** A 方向（c11 slot 分组 DN 融合）两轮 Accepted 实测均慢于 v890（0.891/1.014 ms 对 0.876/0.871 ms），且 tile-offset 修正版出现 TLE，按门槛停止，不晋升。B 方向 v926（c10 static DN，SHA `2633cc995eb2…`）完成四发 AB/BA 同窗配对，两对 c10 分别 −1.65% / −2.06%，其他 11 案在噪声内，完整提交 Accepted；已将 `p1/kernel.py` 晋升为 v926，旧 v890 保留为 `p1/kernel_v890_fused_delta.py`（SHA `436f0a227678…`）。证据见 [本轮 note](experiments/2026-09-30/notes/slot_dn_A_negative_b_promoted_20260930.md)。官方最佳仍是 SID 149493 raw 85.17；本轮未使榜面 raw 90 达成，也未把 platform `tk=0` 异常写成提速。

**2026-09-30 接手审计更新：** 先读 [最新优化指引与 agent 任务卡](OPTIMIZATION_GUIDE_2026-09-30.md)。用户目标仍为 **10 月 1 日 23:59 前 raw 90 / net 80**；03:46 只读榜分 **75.17**（最佳 SID 149493，raw 85.17），接手时生产为 `p1/kernel.py` = `p1/kernel_v890_fused_delta.py`（SHA `436f0a227678…`；现已按本轮 B 结果晋升为 v926）。本轮补齐 **71 个已有 SID**：EP2 metadata 修复、single-peer、后续 BM256 都已有结果，现有 EP2 路线不再投入；v926 的 c10 static DN 有 17 次 AC，但收益仍需同窗确认。新的主实验是 **c11 按 slot 分组、DN 最后一分支融合最终归并**，附索引/容量/字节模型及 35 场景 CPU 验证；**未实现或验证 GPU 收益，生产未改，本轮正式/custom 均 0 次**。调用序号与静态缓存合同列为独立正确性任务。用户明确禁止 `deepseek-brainstorm`，见 [AGENTS.md](AGENTS.md)。

## 接手顺序

1. [当前优化任务书](OPTIMIZATION_GUIDE_2026-09-30.md)：v890 基线、最新终态、积分模型、slot DN 融合方案、预算与 agent 任务。9 月 29 日指引为历史记录，不沿用其 EP2 优先级。
2. [历史状态](docs/STATE.md)：历轮事实与实验结论；旧页首状态不作为当前依据。
3. [提交说明](docs/SUBMISSION.md)：已验证的提交入口、令牌池限制和提交纪律；软件版本以新任务书为准。
4. [v890 源码导航](OPTIMIZATION_GUIDE_2026-09-30.md#4-v890-的有效计算路径)：当前 SHA 的有效函数入口；旧 [CODE_MAP](docs/CODE_MAP.md) 仅供历史参考。
5. `p1/kernel.py`：当前运行基线，路径保持稳定。
6. [最新候选终态台账](reports/2026-09-30-candidate-ledger.csv)与 [`experiments/2026-09-29/candidates/`](experiments/2026-09-29/candidates/)：按 SHA 对照已有 OJ 证据，不把 Canceled 写成性能失败。[新方向 CPU 证明](reports/2026-09-30-slot-last-proof.json)不替代 GPU 测试。
7. [未决合同](OPTIMIZATION_GUIDE_2026-09-30.md#13-仍值得向用户或主办方索取的信息)：截止、目标、3.4 系版本、硬件与初始化接口已明确；跨点权重失效合同、精确 fork 与合法 profile 仍有缺口。

当前事实以新任务书注明的核验时间与 SHA 为准；提交前重新只读检查平台状态。

## 目录职责

| 路径 | 用途 |
|---|---|
| `p1/` | 当前 kernel 与历史候选；本轮不批量迁移 |
| `scripts/` | 提交、查询和分析入口；路径保持稳定 |
| `experiments/YYYY-MM-DD/` | 当轮 `candidates/`、`results/`、`notes/` |
| `docs/` | 当前权威状态与操作说明 |
| `reports/` | 历史审计及可执行导出器；保持原位 |
| `benchmarks/` | 历史提交与测量记录；其中“当前”措辞只对文档标注日期有效 |
| `archive/` | 历史 handoff、会话转储、报告索引和旧入口 |
| `logs/`、`benchmarks/`、`sandbox/` | 原始实验记录与工具；保持原位 |
| `比赛信息/`、`1-full.md` | 题目原始资料 |
| `桌面-XPUOJ提交与平台说明/` | 通用平台说明；部分旧认证流程可能已过期 |
| `.secrets/` | 本地凭据；不要输出、移动或提交 |
| `work/repo/` | 本地依赖；不要移动 |

## 历史资料

旧 handoff 互相覆盖且常在开头保留更早状态，只能作为历史证据。迁移后的入口和回滚清单见 [archive/README.md](archive/README.md) 与 [archive/MOVE_MANIFEST.tsv](archive/MOVE_MANIFEST.tsv)。原始内容、目录内部结构和日志均保留。
