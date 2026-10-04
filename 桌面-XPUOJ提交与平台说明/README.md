# XPUOJ 提交与平台操作说明（通用版）

> 目标读者：接下来做本比赛其他题目的 agent。
> 提交、查询、噪声判断、Pending 处理等机制对所有题目通用；
> 换题时只需修改 `problemOrder` 和 `language`。

## 30 秒快速开始

```bash
cd /home/sakimi26/xpuoj-p1

# 提交并轮询（scripts/submit.py 在 xpuoj-p1 仓库内；文件路径按你的题目目录改）
python scripts/submit.py 你的题目目录/你的文件.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_你的文件.log

# 查 scoreboard（默认 contest 13 / problem 1，可传参）
python 桌面-XPUOJ提交与平台说明/tools/xpuoj_best_score.py --contest-id 13 --problem-order 1

# 查任意 submission 详情
python 桌面-XPUOJ提交与平台说明/tools/xpuoj_status.py 117300 --cases
```

独立工具在本文件夹 `tools/`：

| 工具 | 用途 |
|---|---|
| `tools/xpuoj_submit.py` | 提交任意文件；`--language` 必填；支持 `--contest-id/--problem-order/--poll` |
| `tools/xpuoj_status.py` | 查 submission 状态 + 逐点 tk/tb/score |
| `tools/xpuoj_best_score.py` | 查某题 scoreboard 最佳 |

> 这些独立工具默认读 `/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json`，
> 也可用环境变量 `XPUOJ_SECRET_PATH` 覆盖；`xpuoj_submit.py` 的 `--language`
> 必须按题目要求设置。凭据不要打印或提交。

## 阅读顺序

1. `01-提交代码完整流程.md`
2. `02-API接口与结果解析.md`
3. `03-噪声与同窗口A-B测试.md`
4. `04-Pending与平台异常处理.md`
5. `另一台机器需要准备的东西.md`（跨机部署清单）
