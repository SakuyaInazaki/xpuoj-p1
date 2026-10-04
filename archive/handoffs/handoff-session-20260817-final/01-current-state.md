# 01 当前状态（立即必读）

## 平台状态

- 账号：`dpsk-test` / `601119026@qq.com`
- contestId=13, problemOrder=1, language=`triton-dist`
- 凭据：`.secrets/xpuoj.json`（600，不要打印/提交）
- 平台最佳 submission：**115705**
- raw displayScore：**71.67**
- scoreboard 扣罚后：**61.67**
- submissionCount：**523**
- 扣罚：`min((attempt-100)*0.1,10)`，已到上限 10。

## 两套 canonical 的说明（重要）

| 口径 | submission | raw | timeUsed | 文件 | 说明 |
|---|---:|---:|---:|---|---|
| 平台最佳 | 115705 | **71.67** | 53703 | `p1/kernel.py` / `p1/kernel_gm2.py` | GROUP_M=2；case6 tb=45.559 异常拉高 |
| 实际最快 | 115738 | 71.08 | **51044** | `p1/kernel_fused_gateup_swiglu.py` | FP8 gateup 融合 SwiGLU+amax；tk 总和最低 |

当前 `p1/kernel.py` 已同步为 115705（与 scoreboard 一致）。后续若重新开始实验，建议先把 `p1/kernel_fused_gateup_swiglu.py` 作为 base。

## 115705 逐点

| case | tk(ms) | tb(ms) | 单点分 |
|---:|---:|---:|---:|
| 1 | 7.623 | 17.907 | 70 |
| 2 | 13.065 | 28.979 | 68 |
| 3 | 2.857 | 6.843 | 70 |
| 4 | 1.970 | 4.939 | 71 |
| 5 | 5.140 | 12.641 | 71 |
| 6 | 3.120 | 45.559 | 93 |
| 7 | 4.081 | 10.219 | 71 |
| 8 | 2.858 | 7.167 | 71 |
| 9 | 3.808 | 8.447 | 68 |
| 10 | 3.174 | 7.261 | 69 |
| 11 | 2.352 | 5.643 | 70 |
| 12 | 3.655 | 8.104 | 68 |

## 115738 逐点（推荐 base）

| case | tk(ms) | tb(ms) | 单点分 |
|---:|---:|---:|---:|
| 1 | 6.612 | 17.910 | 73 |
| 2 | 12.228 | 29.539 | 70 |
| 3 | 2.745 | 7.046 | 71 |
| 4 | 1.970 | 5.066 | 72 |
| 5 | 4.902 | 18.897 | 79 |
| 6 | 3.121 | 7.713 | 71 |
| 7 | 3.984 | 10.163 | 71 |
| 8 | 2.845 | 7.145 | 71 |
| 9 | 3.756 | 8.355 | 68 |
| 10 | 3.142 | 6.673 | 67 |
| 11 | 2.362 | 5.621 | 70 |
| 12 | 3.377 | 8.150 | 70 |

## 关键备份文件

| 文件 | 对应 submission | 说明 |
|---|---|---|
| `p1/kernel.py` | 115705 | 平台最佳，当前 local canonical |
| `p1/kernel_gm2.py` | 115705 | 与 kernel.py 一致 |
| `p1/kernel_fused_gateup_swiglu.py` | 115738 | 实际最快 base，推荐 |
| `p1/kernel_115738_backup.py` | 115738 | 备份 |
| `p1/kernel_115696_backup.py` | 115696 | INT8+FP8 swizzle，无 FP8 epilogue 融合 |
| `p1/kernel_115691_backup.py` | 115691 | 仅 FP8 swizzle |
| `p1/kernel_115209_backup.py` | 115209 | 本会话早期 canonical |

## 常用操作

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

解析 submission detail 的逐点 tk/tb 见 `02-competition-and-platform.md`。
