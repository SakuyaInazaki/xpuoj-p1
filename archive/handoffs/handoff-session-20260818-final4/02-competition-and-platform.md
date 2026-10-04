# 02 比赛、题目、平台、评分

## 账号与提交

- 账号：`dpsk-test`，邮箱 `601119026@qq.com`
- 凭据：`/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json`（600，不要打印/提交）
- contestId=13，problemOrder=1，language=`triton-dist`
- API 基础：`https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/`
- 提交封装：`scripts/xpuoj_api.py`、`scripts/submit.py`
- 当前 submissionCount：734；扣罚 `min((attempt-100)*0.1,10)` 已到上限 10。
- 平台保留每题历史最佳；WA/TLE/低分不会降低 scoreboard，所以 scoreboard 可能停在异常 raw 上。

## 题目约束

- 单机 4×H800，EP MoE 推理；`dist` 与 NVSHMEM 已初始化。
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
  - gate/up：BF16 GEMM -> BF16 -> FP32；SwiGLU + route weight 在 FP32。
  - down 输入 activation 转 BF16；down BF16 GEMM -> FP32 累加 -> BF16 output。
- 检查：SQNR≥22dB、输出有限、输入不变、同输入两次运行逐字节一致。
- 禁止：官方 fused MoE/EP、`torch.matmul`；主 GEMM/专家计算必须 Triton 或 triton_dist kernel。

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
- submission `displayScore` = 12 点单点分平均（raw）。
- scoreboard `totalScore` = raw - `min((attempt-100)*0.1,10)`；当前 76 - 10 = 66。
- **timeUsed ≈ 12 点 tk(ms) 之和 × 1000**；判断真实速度看 timeUsed/tk，不要只看 raw。

## 评测 harness

- 每个 testcase 独立进程：
  `Running kernel, testcase=<n>, warmup=1, iters=2, testdata_groups=2`
- 同一 testcase 会多次调用 `run_kernel`；hidden/weight 张量对象可能是新的，
  **缓存 key 必须按 shape，不能按 id(tensor)**。
- hidden_states 每次可能换数据，不能缓存 routing/counts。
- 静态权重只读固定，可按 shape 缓存。
- 总时间限制 500 秒；编译时间也算在内。TLE 可能来自编译慢或极端慢 kernel，不一定是死循环。
- 每个 testcase 独立进程意味着 `_DIAG_CALLS[0]==1` 这类“第二次调用 raise”的诊断在每个 case 都会触发一次。

## API 与结果解析

- 登录：`POST auth/login`
- 提交：`POST contest/play/submit`
- 最近提交：`POST contest/play/querySubmissions`
- scoreboard：`POST contest/play/getContestScoreboardMe`
- 详情：`POST submission/getSubmissionDetail`，body `{"submissionId":"...","locale":"zh_CN"}`
- 逐点结果在 `progress.testcaseResult`：
  - `userOutput` 中 `OJRESULT v1 <hash> <base64json>`，base64 解码得到
    `schema_version / tk_time_ms / tb_time_ms / pass`。
  - `userError` 诊断优先 grep：`Execution error`、`TypeError`、`CUDA error`、
    `SQNR`、`total problem time limit`、`Pointer argument`、`PassManager`、
    `TensorGuardError`、`OutOfResources`、`DETERMINISM FAIL`。

## 评测噪声（本会话再次确认，非常重要）

- 同代码同窗口波动可达 ±0.5ms 以上；case2/9/11/12 尤其明显。
- 本会话异常 tb：
  - 116627 case8 tb=11.282（正常约 7.2）；
  - 116675/116716/116792/116839 等 raw 高分基本都靠 tb 异常；
  - 116792 case5 tb=80.471 直接成为 scoreboard best，但 timeUsed 43123 并不是最快。
- 同代码复测差异：
  - v115：42753 / 43220；
  - v129：42664 / 43537；
  - v140：42343 / **41808**；
  - v159：43198 / 43131。
- 判断优化必须同窗口 A/B：候选前后或旁边投当前 `p1/kernel.py` 作对照。
- 不要重复提交同一文件去碰 tb 异常。
