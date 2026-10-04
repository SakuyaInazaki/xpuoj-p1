# 01 当前状态（立即必读）

## 平台状态

- 账号：`dpsk-test` / `601119026@qq.com`
- contestId=13, problemOrder=1, language=`triton-dist`
- 凭据：`/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json`（600，不要打印/提交）
- scoreboard best：**116792**，raw **76.00**，扣罚后 **66.00**
- submissionCount：734；扣罚 `min((attempt-100)*0.1,10)` 已到上限 10。

## 两个 canonical

| 口径 | submission | raw | timeUsed | 文件 | 说明 |
|---|---:|---:|---:|---|---|
| scoreboard best | 116792 | **76.00** | 43123 | `p1/kernel_v145_order32.py` | case5 tb=80.471 异常；不代表真实性能 |
| 实际性能 base | 116882 | 74.08 | **43131** | `p1/kernel.py` / `p1/kernel_116882_backup.py` | 另一次 116879=43198 |
| 上一稳定 base | 116858 | 74.00 | 43315 | `p1/kernel_116858_backup.py` | route E8 BM64/BK64 |
| 更早稳定 base | 116808 | 73.92 | 43232 | `p1/kernel_116808_backup.py` | case2 custom gather |

> 当前 `p1/kernel.py` = 116882（v159）。scoreboard 接口仍返回 116792。
> 继续迭代请从 `p1/kernel_116882_backup.py` 复制候选。

## 116882 / v159 逐点（当前 base 两次提交）

| case | r1 tk | r1 tb | r1 分 | r2 tk | r2 tb | r2 分 | tk 中位数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.870 | 18.094 | 75 | 5.811 | 18.131 | 75 | 5.841 |
| 2 | 10.112 | 28.899 | 74 | 10.118 | 29.019 | 74 | 10.115 |
| 3 | 2.283 | 7.041 | 75 | 2.286 | 7.045 | 75 | 2.284 |
| 4 | 1.591 | 5.039 | 76 | 1.595 | 5.092 | 76 | 1.593 |
| 5 | 4.132 | 12.701 | 75 | 4.117 | 12.689 | 75 | 4.124 |
| 6 | 2.386 | 8.245 | 77 | 2.390 | 7.575 | 76 | 2.388 |
| 7 | 3.411 | 10.307 | 75 | 3.397 | 10.462 | 75 | 3.404 |
| 8 | 2.307 | 7.224 | 75 | 2.310 | 7.209 | 75 | 2.308 |
| 9 | 3.331 | 8.445 | 71 | 3.334 | 8.452 | 71 | 3.333 |
| 10 | 2.737 | 6.735 | 71 | 2.732 | 6.690 | 71 | 2.735 |
| 11 | 2.046 | 5.647 | 73 | 2.043 | 5.670 | 73 | 2.045 |
| 12 | 2.992 | 8.169 | 73 | 2.998 | 8.174 | 73 | 2.995 |

sum(tk 中位数) = 43.165 ms。

## 116792（scoreboard best）逐点

| case | tk | tb | 分 |
|---:|---:|---:|---:|
| 1 | 5.945 | 18.168 | 75 |
| 2 | 10.553 | 28.924 | 73 |
| 3 | 2.268 | 6.993 | 75 |
| 4 | 1.558 | 4.945 | 76 |
| 5 | 4.142 | **80.471** | **95** |
| 6 | 2.369 | 7.569 | 76 |
| 7 | 3.370 | 10.178 | 75 |
| 8 | 2.257 | 7.061 | 75 |
| 9 | 3.304 | 8.296 | 71 |
| 10 | 2.713 | 7.102 | 72 |
| 11 | 1.849 | 5.601 | 75 |
| 12 | 2.795 | 8.145 | 74 |

## 关键文件与 backups

| 文件 | submission | 说明 |
|---|---|---|
| `p1/kernel.py` | 116882 | 当前实际性能 base（v159） |
| `p1/kernel_116882_backup.py` | 116882 | 同上 |
| `p1/kernel_116858_backup.py` | 116858 | route E8 BM64/BK64（v155） |
| `p1/kernel_116808_backup.py` | 116808 | case2 custom gather（v147） |
| `p1/kernel_116773_backup.py` | 116773 | orderW + GM16（v143） |
| `p1/kernel_116767_backup.py` | 116767 | orderW GM8（v142） |
| `p1/kernel_116754_backup.py` | 116754 | fused orderW（v140） |
| `p1/kernel_116735_backup.py` | 116735 | no id conversions（v136） |
| `p1/kernel_116707_backup.py` | 116707 | relaxed atomics + s4（v130） |
| `p1/kernel_116695_backup.py` | 116695 | relaxed atomics（v129） |
| `p1/kernel_116627_backup.py` | 116627 | persistent fused + order token（v115） |
| `p1/kernel_116310_backup.py` | 116310 | 本会话起始 base |
| `p1/kernel_116142_backup.py` | 116142 | 更早 scoreboard 文件 |

## 常用操作

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

同窗口 A/B 必须投当前 `p1/kernel.py` 作对照，不要拿历史时间Used直接比较。
