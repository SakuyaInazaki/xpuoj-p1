# 02 比赛、题目与评测平台

## 账号与提交

- 账号：`dpsk-test`，邮箱 `601119026@qq.com`。
- 凭据：`/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json`。
- contestId=13，problemOrder=1，语言 `triton-dist`。
- 提交 payload 由 `scripts/xpuoj_api.py` / `scripts/submit.py` 封装。
- 平台保留每题历史最佳；低分/WA 不会降低历史最佳。

## 题目核心约束

- 单机 4×H800，EP MoE 推理，`dist` 已初始化，NVSHMEM 已初始化。
- 参数均为只读 BF16：`hidden_states [T,H]`、`gate_weight [E,H]`、`expert_gate_proj [Ep,I,H]`、`expert_up_proj [Ep,I,H]`、`expert_down_proj [Ep,H,I]`。
- `output [T,H] BF16` 完整写入；`topk` 为 Python int。
- owner rank = `e // Ep`，local id = `e % Ep`。
- 数值语义：
  - route GEMM：BF16 输入→BF16 logits→FP32 softmax。
  - topk weights FP32 归一化，分母 `max(sum,1e-6)`。
  - gate/up GEMM：BF16 输入→BF16→FP32；SwiGLU+route weight 在 FP32。
  - down 输入 act 转 BF16；down 输出 BF16→FP32 按 token 累加→BF16 写 output。
- 检查：SQNR≥22dB；输出有限；输入不变；同一输入两次运行逐字节一致。
- 禁止：官方 fused MoE/EP、`torch.matmul`；主 GEMM/专家计算必须是 Triton 或 triton_dist kernel。

## 12 个实际测试点

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
- submission `displayScore` = 12 点单点分的平均（raw）。
- scoreboard `totalScore` = raw 扣罚：`min((attempt-100)*0.1, 10)`；当前已扣 10。
- 因此目标 raw 70 对应的 scoreboard 是 60。

## 评测 harness 行为（本会话诊断确认）

每个 testcase 独立运行，日志中有：

```text
Running kernel, testcase=<n>, warmup=1, iters=2, testdata_groups=2
```

含义：
- 同一 testcase 进程内会多次调用 `run_kernel`（warmup 1 次 + 2 iters × 2 data groups）。
- 每次调用拿到的 hidden/weight 张量对象可能是新的，`id(tensor)` 每次都变。
- **因此静态权重缓存必须按 shape 作 key，不能按 id。**
- 每个 testcase 进程独立，跨 case 的缓存不会共享；大 buffer 缓存通常只需覆盖同一 shape 的多次调用。

## API

- 基础：`https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/`
- 登录：`POST auth/login`。
- 提交：`POST contest/play/submit`，body 见 `scripts/submit.py`。
- 列表：`POST contest/play/querySubmissions`。
- 详情：`POST submission/getSubmissionDetail`，body `{"submissionId":"...","locale":"zh_CN"}`。
- scoreboard：`POST contest/play/getContestScoreboardMe`，`scripts/best_score.py`。
- `userOutput` 中 `OJRESULT v1 <hash> <base64json>`，base64 解码得到 `schema_version/tk_time_ms/tb_time_ms/pass`。
- `userError` 包含每个 testcase 的 stderr/SQNR/determinism/异常 traceback，是做诊断提交时最重要的信息来源。

## 评测噪声

- 同架构、完全未改的 case 也会大幅波动；case9 尤其明显，本会话见过 4.88~39.3ms。
- 单次总分不可全信，应看同一优化多次重复的单点 tk 信号。
- 不要用低分轮否定或重复提交同一代码；除非有新的确定优化。
