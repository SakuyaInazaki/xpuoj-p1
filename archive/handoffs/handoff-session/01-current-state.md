# 01 当前状态（立即必读）

## Canonical

- Submission：**114970**
- Status：Accepted，`score=100`
- `displayScore = 69.33`（raw，scoreboard 扣罚后 59.33）
- `timeUsed = 58057`（平台单位与逐点 ms 之和一致，可视为 us）
- `memoryUsed = 3570276`
- 文件：`p1/kernel.py` = `p1/kernel_case9_int8_both_clean.py`
- SHA-256：
  `e8b70ee81ef48e40b491a7729321d6619afff0b81d2195c08fcbbce4d4976270`

### 检查清单

```bash
cd /home/sakimi26/xpuoj-p1
sha256sum p1/kernel.py p1/kernel_case9_int8_both_clean.py
python -m py_compile p1/kernel.py
python scripts/best_score.py
```

预期两个文件 SHA 相同且为上面这串；py_compile OK。

## 平台状态

- 账号：`dpsk-test` / `601119026@qq.com`
- 凭据：`.secrets/xpuoj.json`（600 权限，绝不打印/提交/复制）
- 比赛 ID：13，problemOrder：1，语言：`triton-dist`
- Scoreboard 接口当前返回 `114970 / 59.33`（raw 69.33 扣罚 10 后）。
- 在线 `submissionCount`：447。
- 扣罚：`min((attempt-100)*0.1, 10)`，已到上限 10。
- 平台保留每题历史最佳，因此后续失败/低分提交不会降低 114970 的最佳。

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

平均单点分 = 69.33。

## 会话内分数演进

上一会话：
- 会话开始 canonical：114382 / 51.08
- 114668 / 51.58（direct allgather v2 + case2 pack）
- 114697 / 52.17（E96 sorted dispatch c4）
- 114706 / 52.50（sorted dispatch v3 hybrid）

本会话：
- 接手 canonical：114785 / 52.67
- 114796 / 55.50（小 NVSHMEM buffer 多 key 缓存）
- 114830 / 56.83（static/full 权重缓存改为 shape key）
- 114839 / 58.25（case3 切 replicated）
- **114845 / 58.58（`_gather_branch_sum` H 混合 BLOCK_H，最终 canonical）**

## 关键备份文件

| 文件 | 含义 |
|---|---|
| `p1/kernel.py` | 当前 canonical，=114970 |
| `p1/kernel_case9_int8_both_clean.py` | 114970 命名副本 |
| `p1/kernel_114965_backup.py` | 114965 备份 |
| `p1/kernel_114962_backup.py` | 114962 备份 |
| `p1/kernel_114913_backup.py` | 114913 备份 |
| `logs/submit_case9_int8_both_clean.log` | 114970 提交终态日志 |

## 常用命令

```bash
cd /home/sakimi26/xpuoj-p1

# 查看榜单缓存（可能不是真实最佳）
python scripts/best_score.py

# 提交并轮询（新实验前先保存候选源码和日志）
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log

# 查 submission detail 并解析 12 点 tk/tb
python - <<'PY'
import sys,json,base64,re
sys.path.insert(0,'scripts')
from xpuoj_api import Client
c=Client(); sid=114970; d=c.get_detail(sid).json()
for h,v in d['progress']['testcaseResult'].items():
    uo=v.get('userOutput',''); m=re.search(r'OJRESULT v1 \S+ (\S+)',uo)
    if m:
        j=json.loads(base64.b64decode(m.group(1)))
        print(v.get('input'), j['tk_time_ms'], j['tb_time_ms'], v.get('displayScore'))
PY
```
