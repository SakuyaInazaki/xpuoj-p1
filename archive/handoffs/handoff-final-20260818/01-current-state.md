# 01 当前状态（立即必读）

> **2026-08-18 第二轮更新：当前 scoreboard 已变为 116142（raw 74.75 / 64.75），
> 详见 `07-takeover-20260818-2.md`。本文件保留 115950 时代状态作为历史。**

## 平台状态

- 账号：`dpsk-test` / `601119026@qq.com`
- contestId=13, problemOrder=1, language=`triton-dist`
- 凭据：`.secrets/xpuoj.json`（600，不要打印/提交）
- scoreboard 最佳：**115950**
- raw displayScore：**74.08**
- scoreboard 扣罚后：**64.08**
- submissionCount：**584**
- 扣罚：`min((attempt-100)*0.1,10)`，已到上限 10。

## 两个 canonical 的说明（重要）

| 口径 | submission | raw | timeUsed | 文件 | 说明 |
|---|---:|---:|---:|---|---|
| scoreboard 最佳 | 115950 | **74.08** | 44017 | `p1/kernel.py` / `p1/kernel_115950_backup.py` | case2 tb=46.98 异常拉高；实际 tk 略慢 |
| 稳定实际最快 | 115907 | 74.00 | **43538** | `p1/kernel_115907_backup.py` | 建议继续迭代的性能 base |

当前 `p1/kernel.py` 已同步为 115950（与 scoreboard 一致）。
后续若重新开始实验，强烈建议先复制 `p1/kernel_115907_backup.py` 作为 base。

## 115907（稳定 base）逐点

| case | tk(ms) | tb(ms) | 单点分 |
|---:|---:|---:|---:|
| 1 | 6.003 | 18.108 | 75 |
| 2 | 10.172 | 28.943 | 73 |
| 3 | 2.315 | 7.111 | 75 |
| 4 | 1.598 | 5.121 | 76 |
| 5 | 4.158 | 12.770 | 75 |
| 6 | 2.390 | 7.642 | 76 |
| 7 | 3.390 | 10.283 | 75 |
| 8 | 2.308 | 7.280 | 75 |
| 9 | 3.411 | 8.381 | 71 |
| 10 | 2.795 | 7.167 | 71 |
| 11 | 2.015 | 5.666 | 73 |
| 12 | 2.983 | 8.190 | 73 |

## 115950（scoreboard 最佳）逐点

| case | tk(ms) | tb(ms) | 单点分 |
|---:|---:|---:|---:|
| 1 | 6.136 | 18.073 | 74 |
| 2 | 10.301 | **46.980** | 82 |
| 3 | 2.330 | 6.952 | 74 |
| 4 | 1.600 | 5.019 | 75 |
| 5 | 4.223 | 12.620 | 74 |
| 6 | 2.410 | 7.628 | 75 |
| 7 | 3.438 | 10.191 | 74 |
| 8 | 2.321 | 7.103 | 75 |
| 9 | 3.423 | 8.375 | 70 |
| 10 | 2.808 | 6.612 | 70 |
| 11 | 2.023 | 5.633 | 73 |
| 12 | 3.004 | 8.136 | 73 |

115950 的 raw 高完全来自 case2 tb 异常；timeUsed 44017 比 115907 慢约 0.48ms。

## 关键文件

| 文件 | 对应 submission | 说明 |
|---|---|---|
| `p1/kernel.py` | 115950 | scoreboard canonical |
| `p1/kernel_115950_backup.py` | 115950 | 与 kernel.py 一致 |
| `p1/kernel_115907_backup.py` | 115907 | **稳定性能 base，推荐** |
| `p1/kernel_v50_case2_gather_bm256.py` | 115943 | case2 BM256 gather w8；单次 timeUsed 43467 但复测波动 |
| `p1/kernel_v51_fused_gather_quant.py` | 115944/115946 | fused gather+quant 尝试，最终 TLE，勿用 |
| `p1/kernel_115854_backup.py` | 115854 | 上一阶段最佳 |
| `p1/kernel_115738_backup.py` | 115738 | 早期实际最快 |
| `p1/kernel_115705_backup.py` | 115705 | 最初平台最佳 |

## 常用操作

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

解析 submission detail 的逐点 tk/tb 见 `02-competition-and-platform.md`。
