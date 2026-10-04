# 01 当前状态（立即必读）

## Canonical

- Submission：**114970**
- 状态：Accepted，`score=100`
- raw `displayScore`：**69.33**
- scoreboard 扣罚后：**59.33**
- `timeUsed`：58057（约等于 12 点 tk ms 之和 ×1000）
- `memoryUsed`：3570276
- 文件：`p1/kernel.py` = `p1/kernel_case9_int8_both_clean.py`
- SHA-256：`e8b70ee81ef48e40b491a7729321d6619afff0b81d2195c08fcbbce4d4976270`
- 在线代码与本地逐字节一致（已核对）。

## 平台状态

- 账号：`dpsk-test` / `601119026@qq.com`
- 凭据：`.secrets/xpuoj.json`（600，绝不提交/打印）
- contestId：13，problemOrder：1，language：`triton-dist`
- 在线 submissionCount：**447**
- 扣罚：`min((attempt-100)*0.1, 10)`，已到上限 10。
- scoreboard 当前返回 114970 / 59.33；**submission detail 的 displayScore 是 raw**。

## 114970 逐点结果

| case | tk(ms) | tb(ms) | 单点分 |
|---:|---:|---:|---:|
| 1 | 7.703 | 18.203 | 70 |
| 2 | 15.322 | 43.516 | 73 |
| 3 | 2.877 | 6.961 | 70 |
| 4 | 1.961 | 5.008 | 71 |
| 5 | 5.160 | 12.997 | 71 |
| 6 | 3.090 | 7.565 | 70 |
| 7 | 4.083 | 10.243 | 71 |
| 8 | 2.865 | 7.229 | 71 |
| 9 | 5.022 | 8.430 | 62 |
| 10 | 3.963 | 6.737 | 62 |
| 11 | 2.349 | 6.321 | 72 |
| 12 | 3.662 | 8.209 | 69 |

平均 raw = 69.33。

## 关键备份

| 文件 | 说明 |
|---|---|
| `p1/kernel_case9_int8_both_clean.py` | 114970 命名副本（当前 canonical） |
| `p1/kernel_114965_backup.py` | 114965 / 68.67 |
| `p1/kernel_114962_backup.py` | 114962 / 68.50 |
| `p1/kernel_114913_backup.py` | 114913 / 68.33 |
| `p1/kernel_114904_backup.py` | 114904 / 67.33 |
| `p1/kernel_case9_int8_both.py` | 114966 / 68.58，raw tk 总和更低，display 略低 |
| `p1/kernel_best_case9_fp8both.py` | 114971 / 68.75，case9 单点 4.936 |
| `p1/kernel_best_case10_fp8down.py` | 114972 / 67.33，case10 FP8 down 变体 |
| `p1/kernel_hybrid_int8_case3_case12fp8.py` | 114955 / 67.50，case3/12 FP8 组合 |
| `p1/kernel_int8_tiled*.py` | per-tile INT8 失败反例，保留供调试 |

## 常用命令

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

解析 submission detail：

```python
import sys,json,base64,re
sys.path.insert(0,'scripts')
from xpuoj_api import Client
c=Client(); d=c.get_detail(114970).json()
for k,v in d['progress']['testcaseResult'].items():
    uo=v.get('userOutput',''); m=re.search(r'OJRESULT v1 \S+ (\S+)',uo)
    if m:
        j=json.loads(base64.b64decode(m.group(1)))
        print(v.get('input'), j['tk_time_ms'], j['tb_time_ms'], v.get('displayScore'))
```
