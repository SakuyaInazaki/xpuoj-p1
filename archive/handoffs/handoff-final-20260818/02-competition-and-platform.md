# 02 比赛、题目、平台、评分

## 账号与提交

- 账号：`dpsk-test`，邮箱 `601119026@qq.com`
- 凭据：`/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json`
- contestId=13, problemOrder=1, language=`triton-dist`
- API 基础：`https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/`
- 提交 payload 由 `scripts/xpuoj_api.py` / `scripts/submit.py` 封装。

## 题目约束

- 单机 4×H800，EP MoE 推理，`dist` 和 NVSHMEM 已初始化。
- 输入只读：
  - `hidden_states [T,H] BF16`
  - `gate_weight [E,H] BF16`（四卡相同）
  - `expert_gate_proj [Ep,I,H] BF16`
  - `expert_up_proj [Ep,I,H] BF16`
  - `expert_down_proj [Ep,H,I] BF16`
  - `output [T,H] BF16` 完整写入
  - `topk` Python int
- owner rank = `e // Ep`，local id = `e % Ep`，`Ep=E/4`。
- 数值语义：
  - route：BF16 GEMM -> BF16 logits -> FP32 softmax -> topk -> FP32 weight 归一化（分母 `max(sum,1e-6)`）。
  - gate/up：BF16 GEMM -> BF16 -> FP32；SwiGLU+route weight 在 FP32。
  - down 输入 activation 转 BF16；down BF16 GEMM -> FP32 累加 -> BF16 output。
- 检查：SQNR≥22dB、输出有限、输入不变、同输入两次运行逐字节一致。
- 禁止：官方 fused MoE/EP、`torch.matmul`；主 GEMM/专家计算必须 Triton 或 triton-dist kernel。

## 12 个测试点

| case | T | H | E | I | topk |
|---:|---:|---:|---:|---:|---:|
| 1 | 16384 | 4096 | 8 | 8192 | 2 |
| 2 | 16384 | 4096 | 8 | 14336 | 2 |
| 3 | 16384 | 2048 | 32 | 2048 | 4 |
| 4 | 16384 | 2048 | 32 | 1024 | 4 |
| 5 | 8192 | 3584 | 64 | 2560 | 8 |
| 6 | 8192 | 3584 | 64 | 1024 | 8 |
| 7 | 16384 | 4096 | 96 | 2048 | 3 |
| 8 | 16384 | 4096 | 96 | 1024 | 3 |
| 9 | 4096 | 4096 | 256 | 2048 | 8 |
| 10 | 4096 | 4096 | 256 | 1536 | 8 |
| 11 | 65536 | 1024 | 32 | 1024 | 2 |
| 12 | 65536 | 1024 | 32 | 2048 | 2 |

## 评分

- 单点 `score = floor(100 * tb / (tb + tk))`。
- submission displayScore = 12 点单点分平均（raw）。
- scoreboard totalScore = raw 扣罚 `min((attempt-100)*0.1,10)`；当前 raw 74.08 -> 64.08。
- 平台保留每题历史最佳；WA/TLE/低分不会降低历史最佳。
- **timeUsed 约等于 12 点 tk(ms) 之和 × 1000**，是判断真实速度的关键，不要只看 displayScore。

## 评测 harness

- 每个 testcase 独立进程，日志 `Running kernel, testcase=<n>, warmup=1, iters=2, testdata_groups=2`。
- 每个 testcase 会多次调用 `run_kernel`；每次 hidden/weight 张量对象可能都是新的，`id(tensor)` 不可作为静态缓存 key。
- hidden_states 每次运行都可能换数据，不能缓存 routing/counts。
- 静态权重只读固定，可按 shape 缓存。
- 500 秒总时间限制；kernel 编译时间也算在内。多次 TLE 是编译或极端慢 kernel 导致的全局超时。

## API 与结果解析

- 登录：`POST auth/login`
- 提交：`POST contest/play/submit`
- 最近提交：`POST contest/play/querySubmissions`
- scoreboard：`POST contest/play/getContestScoreboardMe`
- 详情：`POST submission/getSubmissionDetail` body `{"submissionId":"...","locale":"zh_CN"}`
- 逐点结果在 `progress.testcaseResult` 的每个 testcase：
  - `userOutput` 中 `OJRESULT v1 <hash> <base64json>`，base64 解码得到
    `schema_version / tk_time_ms / tb_time_ms / pass`。
  - `userError` 可能很长；错误诊断优先 grep `Execution error`、`TypeError`、
    `CUDA error`、`SQNR`、`total problem time limit`。
- 解析脚本示例：

```python
import sys, json, base64, re
sys.path.insert(0, "scripts")
from xpuoj_api import Client

c = Client()
d = c.get_detail(115907).json()
for h, v in d["progress"]["testcaseResult"].items():
    m = re.search(r"OJRESULT v1 \S+ (\S+)", v.get("userOutput", ""))
    if m:
        j = json.loads(base64.b64decode(m.group(1)))
        print(v.get("input"), j["tk_time_ms"], j["tb_time_ms"], v.get("displayScore"))
```

## 评测噪声

- 同代码、未改动 case 也会大幅波动；case2/9/11/12 尤其明显。
- 本 agent 会话见到：
  - 115705 case6 tb=45.559 异常；
  - 115836 case1 tb=40.526 异常；
  - 115950 case2 tb=46.980 异常（当前 scoreboard 最佳正是靠它）。
- 判断优化要看 timeUsed/tk 重复信号，不能靠单次 raw/displayScore。
