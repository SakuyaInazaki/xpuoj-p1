# 01 当前状态（立即必读）

## 平台状态

- 账号：`dpsk-test` / `601119026@qq.com`
- contestId=13, problemOrder=1, language=`triton-dist`
- 凭据：`/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json`（600，不要打印/提交）
- scoreboard best：**116961**，raw **76.67**，扣罚后 **66.67**（case5 tb=264.994 异常）
- submissionCount：906；扣罚 `min((attempt-100)*0.1,10)` 已到上限 10。

## Canonical

| 口径 | submission | raw | timeUsed | 文件 | 说明 |
|---|---:|---:|---:|---|---|
| scoreboard best | 116961 | **76.67** | 42225 | `p1/kernel_v165_final_tiled_aggr2.py` | case5 tb 异常，不代表真实性能 |
| **实际性能 base** | **117300** | 73.92 | **43266** | `p1/kernel.py` / `p1/kernel_117300_backup.py` | 当前 v233；复测 117304=43325 |
| 上一稳定 base | 117234 | 74.17 | 42961 | `p1/kernel_117234_backup.py` | token quant BK64 之前 |
| 更早稳定 base | 117218 | 74.17 | 42903 | `p1/kernel_117218_backup.py` | case2 gather BM64/BH128 |

> 当前 `p1/kernel.py` = 117300（v233）。scoreboard 接口仍返回 116961。
> 继续迭代请从 `p1/kernel_117300_backup.py` 或 `p1/kernel.py` 复制候选。

## 117300 / v233 逐点（当前 base 两次提交）

| case | r1 tk | r1 tb | r1 分 | r2 tk | r2 tb | r2 分 | tk 中位数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.872 | 18.040 | 75 | 5.856 | 18.063 | 75 | 5.864 |
| 2 | 10.183 | 28.871 | 73 | 10.276 | 28.817 | 73 | 10.230 |
| 3 | 2.305 | 7.018 | 75 | 2.298 | 7.078 | 75 | 2.302 |
| 4 | 1.604 | 5.032 | 75 | 1.591 | 5.021 | 75 | 1.598 |
| 5 | 4.158 | 12.694 | 75 | 4.159 | 12.651 | 75 | 4.159 |
| 6 | 2.405 | 7.538 | 75 | 2.415 | 7.609 | 75 | 2.410 |
| 7 | 3.422 | 10.280 | 75 | 3.414 | 10.226 | 74 | 3.418 |
| 8 | 2.322 | 7.177 | 75 | 2.317 | 7.212 | 75 | 2.320 |
| 9 | 3.366 | 8.445 | 71 | 3.364 | 8.458 | 71 | 3.365 |
| 10 | 2.702 | 6.703 | 71 | 2.718 | 6.699 | 71 | 2.710 |
| 11 | 1.988 | 5.679 | 74 | 1.989 | 5.650 | 73 | 1.989 |
| 12 | 2.939 | 8.140 | 73 | 2.928 | 8.148 | 73 | 2.934 |

sum(tk 中位数) ≈ 43.299 ms（r1=43.266, r2=43.325）。

## 117234（gather BH256 中间 base）逐点，供对照

tk: 5.887 10.154 2.271 1.573 4.132 2.379 3.361 2.252 3.293 2.728 1.991 2.940
sum=42.961ms；raw74.17。

## 关键文件与 backups

| 文件 | submission | SHA 前 16 | 说明 |
|---|---|---|---|
| `p1/kernel.py` | 117300 | 14329d914f56144d2 | 当前 base v233 |
| `p1/kernel_117300_backup.py` | 117300 | 同上 | 当前 base |
| `p1/kernel_117234_backup.py` | 117234 | ef9eca54091d4b1e | gather BM64/BH256，quant BK128 |
| `p1/kernel_117218_backup.py` | 117218 | 76dddada65c070b9 | gather BM64/BH128 |
| `p1/kernel_117207_backup.py` | 117207 | 252697eaaf23e02b | route E32 BM64 中间 base |
| `p1/kernel_117157_backup.py` | 117157 | 28f81c6fe2224325 | fused GM32 中间 base |
| `p1/kernel_117151_backup.py` | 117151 | 4337f27cbebabcff | route N96 w4 中间 base |
| `p1/kernel_117077_backup.py` | 117077 | 84bedefff4f1deec | SwiGLU BN64 中间 base |
| `p1/kernel_117064_backup.py` | 117064 | 6f8ad21a727b9ce2 | SwiGLU BN128 + N96 中间 base |
| `p1/kernel_117050_backup.py` | 117050 | 9a49ca60915153f2 | route N96 BM64/BK128 中间 base |
| `p1/kernel_117009_backup.py` | 117009 | a35aa9f77c7a7b09 | final tiling BT32/w32 中间 base |
| `p1/kernel_116882_backup.py` | 116882 | 3c99beae2d6533aa | 本会话起始 base v159 |

## 常用操作

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

解析逐点 tk/tb：

```python
import sys, json, base64, re
sys.path.insert(0, "scripts")
from xpuoj_api import Client
c = Client()
d = c.get_detail(117300).json()
for h, v in d["progress"]["testcaseResult"].items():
    m = re.search(r"OJRESULT v1 \S+ (\S+)", v.get("userOutput", ""))
    if m:
        j = json.loads(base64.b64decode(m.group(1)))
        print(v.get("input"), j["tk_time_ms"], j["tb_time_ms"], v.get("displayScore"))
```
