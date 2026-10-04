# XPUOJ P1 MegaMoE 最终交接（本会话完整版，submission 116594-116897）

> 本文件夹汇总从接手 `handoff-session-20260818-final/`（截至 submission 116568）之后，
> 本 agent 从 submission 116594 到 116897 的全部尝试、晋升、踩坑与当前代码状态。
> 上一份：`../handoff-session-20260818-final/`；本会话中间快照：`../handoff-session-20260818-final2/`、`../handoff-session-20260818-final3/`。

## 10 秒结论

- 比赛：XPUOJ contestId=13, problemOrder=1, language=`triton-dist`，账号 `dpsk-test`。
- scoreboard best：**116792，raw 76.00，扣罚后 66.00**。
  - 116792 的 raw 靠 case5 tb=80.471 异常，不是真实性能。
- **实际性能 base / 当前 `p1/kernel.py`：116882（v159），timeUsed 43131（另一次 43198）。**
- v159 从旧 116310（timeUsed 42881 的早窗口成绩）出发，主要晋升链：
  - 116627 v115：fused gateup persistent + order-derived token gather；
  - 116695 v129：所有 activation amax `sem="relaxed"`；
  - 116707 v130：fused gateup `num_stages=4`；
  - 116735 v136：topk_ids 保持 int64；
  - 116754 v140：fused gateup `orderW`（kernel 内按 order 读 route weight）；
  - 116767 v142：case2 SwiGLU 也 `orderW`；
  - 116808 v147：case2 token gather 改 custom order-derived gather；
  - 116858 v155：route E8 BM64/BN16/w4；
  - 116882 v159：route E8 BLOCK_K=128。
- submissionCount：734；扣罚 `min((attempt-100)*0.1,10)` 已到上限 10。
- 本会话共提交 101 次（116594-116897 中属于本 agent 的 id），全部逐点 tk/tb 在
  `submissions_raw_116594-116897.json`。

## 阅读顺序

1. `01-current-state.md` —— 当前文件、SHA、逐点结果、backups
2. `02-competition-and-platform.md` —— 比赛/题目/评分/API/噪声
3. `03-architecture.md` —— 当前 v159 的代码路径与函数行号
4. `04-session-timeline.md` —— 101 次提交完整时间线与结论
5. `05-pitfalls-and-sandbox.md` —— 本会话全部踩坑与禁止事项
6. `06-candidates-and-next-steps.md` —— 候选文件与下一步优先级
7. `submissions_raw_116594-116897.json` —— 101 次提交 meta + 逐点 tk/tb + userError tail

## 接手检查清单

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
# scoreboard 应输出 totalScore 66, submissionId 116792

sha256sum p1/kernel.py p1/kernel_116882_backup.py p1/kernel_116858_backup.py p1/kernel_116808_backup.py
python -m py_compile p1/kernel.py p1/kernel_116882_backup.py
```

关键 SHA-256：

```text
3c99beae2d6533aa95611b676275d81a0746f085ce80fab238fb749479cceba2  p1/kernel.py (=116882, 当前实际 base)
3c99beae2d6533aa95611b676275d81a0746f085ce80fab238fb749479cceba2  p1/kernel_116882_backup.py
56dcabbf892ba9193e16a31f2629d8c14de5ceac96d9c8204b9568d13345a804  p1/kernel_116858_backup.py (=116858, route E8 BM64/BK64)
58691435e51dbfef9b8c0793794de114477261a59cb619139cd4f63d0cddf638  p1/kernel_116808_backup.py (=116808, v147)
```

提交新实验：

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

解析逐点 tk/tb：

```python
import sys, json, base64, re
sys.path.insert(0, "scripts")
from xpuoj_api import Client
c = Client()
d = c.get_detail(116882).json()
for h, v in d["progress"]["testcaseResult"].items():
    m = re.search(r"OJRESULT v1 \S+ (\S+)", v.get("userOutput", ""))
    if m:
        j = json.loads(base64.b64decode(m.group(1)))
        print(v.get("input"), j["tk_time_ms"], j["tb_time_ms"], v.get("displayScore"))
```
