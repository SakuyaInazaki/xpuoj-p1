# XPUOJ P1 MegaMoE 最终交接（本会话完整版，截至 submission 116568）

> 本文件夹是当前 agent 从接手 `handoff-final-20260818/` 以来，完整梳理的交接材料。
> 覆盖 submission 116130-116568 共 49 次提交，包含实验、坑、架构、比赛信息。
> 上一份交接：`../handoff-final-20260818/`；更早历史：`../handoff-session-20260817-final*/`。

## 10 秒结论

- 比赛：XPUOJ contestId=13, problemOrder=1, language=`triton-dist`，账号 `dpsk-test`。
- scoreboard 最佳：**116142，raw 74.75，扣罚后 64.75**。
  - 116142 是 K-constexpr 变体，case4 tb 异常，不能代表真实性能。
- **实际性能 base / 当前 `p1/kernel.py`：116310，timeUsed 42881（复测 43524）**。
  - 116310 = 116142 基础上增加：
    1. FP8 down GEMM 的 B 矩阵 host-side TensorDescriptor TMA；
    2. down activation per-row FP8 scale；
    3. sorted-token gather + per-row amax/scale；
    4. fused gateup row-A scale。
- submissionCount：633；扣罚早已到上限 10。
- 旧稳定 base：115907（timeUsed 43538）；scoreboard 文件备份：116142。

## 阅读顺序

1. `01-current-state.md` —— 当前状态、关键 SHA、逐点结果
2. `02-competition-and-platform.md` —— 比赛/题目/评分/API/评测噪声
3. `03-architecture.md` —— 当前 116310 代码路径与函数行号
4. `04-session-timeline.md` —— 本会话 49 次提交完整时间线与结论
5. `05-pitfalls-and-sandbox.md` —— 全部踩坑与禁止事项
6. `06-candidates-and-next-steps.md` —— 候选文件与下一步优先级
7. `submissions_raw_116130-116568.json` —— 49 次提交 meta + 逐点 tk/tb + userError tail

## 接手检查清单

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
# scoreboard 应输出 totalScore 64.75, submissionId 116142

sha256sum p1/kernel.py p1/kernel_116310_backup.py p1/kernel_116142_backup.py p1/kernel_115907_backup.py
python -m py_compile p1/kernel.py p1/kernel_115907_backup.py
```

关键 SHA-256：

```text
2c8204026ecea36158b62d2a04c566918a4c7e4c0f8d36b8cac42f8cb98f71f7  p1/kernel.py (=116310, 实际性能 base)
2c8204026ecea36158b62d2a04c566918a4c7e4c0f8d36b8cac42f8cb98f71f7  p1/kernel_116310_backup.py
31321bb0116b509203536914e8dc57a03a6b68ff45f235baedea2dc49397d2f4  p1/kernel_116142_backup.py (=116142, scoreboard best)
52a8533badbddceb1ce6ef4cc9548443d4f3e5662e37a23b67f3545d9306032a  p1/kernel_115907_backup.py (=115907, 旧稳定 base)
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
d = c.get_detail(116310).json()
for h, v in d["progress"]["testcaseResult"].items():
    m = re.search(r"OJRESULT v1 \S+ (\S+)", v.get("userOutput", ""))
    if m:
        j = json.loads(base64.b64decode(m.group(1)))
        print(v.get("input"), j["tk_time_ms"], j["tb_time_ms"], v.get("displayScore"))
```
