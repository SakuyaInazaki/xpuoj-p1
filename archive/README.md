# 历史资料与回退

当前入口是根 [README](../README.md)与 [v926 任务书](../docs/OPTIMIZATION_GUIDE_V926_2026-09-30.md)。历史文件保留原文，其中“当前”、任务分工、链接和版本只对当时有效。

| 目录 | 内容 |
|---|---|
| [p1](p1/README.md) | 本轮从 p1 迁出的 1552 个历史源码、结果与笔记；逐文件 SHA 已记录 |
| [guides](guides/README.md) | 原根目录四份优化指引；旧 slot DN 主线已关闭 |
| `historical-docs/` | STATE / CODE_MAP 在替换前的完整快照 |
| `legacy/` | 历次 README；本轮原文为 `README.pre-v926-handoff-20260930.md` |
| `handoffs/` | 2026-08 交接与累计笔记，内部结构保留 |
| `transcripts/` | 旧长会话转储，可能含旧认证内容，不整篇复制或输出 |
| `reports/` | 历史报告索引；根 reports 的报告本体保持原位 |

本轮 [WORKSPACE_MOVES_2026-09-30.csv](WORKSPACE_MOVES_2026-09-30.csv)有 **1556 项**，记录旧路径、新路径、SHA-256、大小、迁移时间；此前迁移另见 [MOVE_MANIFEST.tsv](MOVE_MANIFEST.tsv)。本轮清理未删除历史候选和原始测量，未修改凭据或依赖。

旧指引内的相对链接仍以迁移前根目录为上下文，历史脚本也可能含旧路径；恢复运行前先读脚本并按迁移表解析，不能直接运行不明提交脚本。可从项目根查询：

```bash
python3 scripts/find_candidate.py kernel_v800
python3 scripts/find_candidate.py 436f0a227678
```

## 回退

项目没有 Git。回退某个历史文件时，先从清单找到 `new_path`，校验 SHA，再确认 `old_path` 不存在，最后移回；不要覆盖后来创建的文件。当前 README/STATE/CODE_MAP 的快照应另存后再恢复。此前清单中的目录 SHA 使用该清单注明的算法，本轮 1556 项均为文件本身 SHA。

生产回退使用 `p1/references/` 中的已测快照，复核目标 SHA 并更新根 README。无需为接手把所有旧候选搬回 p1。
