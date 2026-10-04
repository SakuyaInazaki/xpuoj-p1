# 02 比赛、题目、平台、评分

## 账号与提交

- 账号：`dpsk-test`，邮箱 `601119026@qq.com`
- 凭据：`/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json`
- contestId=13, problemOrder=1, language=`triton-dist`
- API 基础：`https://sd629vuj4f7uh2cscrbe0.apgateway-cn-beijing.volceapi.com/api/`（实际前缀以 `scripts/xpuoj_api.py` 为准）
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
- scoreboard totalScore = raw 扣罚 `min((attempt-100)*0.1,10)`；当前已扣 10。
- 当前 raw 71.67 -> scoreboard 61.67。
- 平台保留每题历史最佳；WA/TLE/低分不会降低历史最佳。

## 评测 harness

- 每个 testcase 独立进程，日志 `Running kernel, testcase=<n>, warmup=1, iters=2, testdata_groups=2`。
- 每个 testcase 会多次调用 `run_kernel`；每次 hidden/weight 张量对象可能都是新的，`id(tensor)` 不可作为静态缓存 key。
- hidden_states 每次运行都可能换数据，不能缓存 routing/counts。
- 静态权重只读固定，可按 shape 缓存；本会话所有 full/local weight 低精度缓存都按 shape key。

## API 与结果解析

- 登录：`POST auth/login`
- 提交：`POST contest/play/submit`
- 最近提交：`POST contest/play/querySubmissions`
- 详情：`POST submission/getSubmissionDetail` body `{"submissionId":"...","locale":"zh_CN"}`
- scoreboard：`POST contest/play/getContestScoreboardMe`
- `userOutput` 中 `OJRESULT v1 <hash> <base64json>`，base64 解码得到 `schema_version/tk_time_ms/tb_time_ms/pass`。
- `userError` 里每个 testcase 的 stderr、SQNR、determinism、traceback 是诊断最重要来源。
- 诊断提交时 `userError` 可能是 `{"data":..., "omittedLength":...}`，需要先取 `data` 字段再 grep。
- `timeUsed` 数值约等于 12 点 tk(ms) 之和，可快速核对真实速度；不要只看 displayScore。

## 评测噪声

- 同代码、未改动的 case 也会大幅波动；case2/9/11/12 尤其明显。
- 本会话见到过 case6 tb=45.559 这种异常基线，直接让 115705 display 虚高。
- 判断优化要看多次重复的 timeUsed/tk 信号，而不是单次 displayScore。
