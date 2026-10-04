# 04 沙箱限制、已踩坑、必须注意的细节

## 沙箱 / Language validation

- **禁止 try/except**：114631 直接 Language validation 崩掉。导入可选模块不能 try 包裹。
- 禁止 `torch.matmul`；禁止函数式 `torch.argsort`、`torch.repeat_interleave`、`torch.cumsum`；
  可用对应 tensor method（`x.argsort(stable=True)`、`x.repeat_interleave(...)`、`x.cumsum(...)`）。
- 禁止 `tensor.clamp_min`；用 `torch.maximum`。
- 禁止 `tensor.data_ptr()`；缓存 key 用 `id(tensor)`。
- 禁止导入 `nvshmem` 模块；NVSHMEM 能力通过 `triton_dist` 的 `libshmem_device` / utils 使用。
- 禁止 `scatter_reduce_`（TensorGuard）。
- 全局非字面量赋值会 Language validation 失败；本会话新增的全局缓存 dict、字面量常量没问题。
- `torch.topk` 的 k 必须位置传参。
- `torch.cuda.Stream/Event` 可用，但本会话及前序实验均未因 stream overlap 获得收益。

## NVSHMEM / direct communication 坑（本会话新踩）

1. **官方 `fast_allgather(push2d)` 不可用**：
   - 114635 首轮 SQNR/determinism 通过，正式计时段 case5/6/9/10 全部
     `tk_time_ms=0.0, pass=false`。不要再尝试。

2. **自写 direct all-gather 的两种写法有本质区别**：
   - v1：先 `recv_buf[rank_block].copy_(x)`，kernel 跳过本地 put -> 正式计时段失败。
   - v2：kernel 写全部 rank block（**包括本地 rank**），无前置 copy -> 稳定。
   - 每个 direct all-gather 后必须调用 `nvshmem_barrier_all_on_stream()`。

3. **不要把多个 direct all-gather kernel launch 合并到一个 barrier**：
   - 114743 将 hidden+pack 两个 AG 后只做一次 barrier，case5 packed route 出现
     SQNR=-1.4dB、determinism fail。保持“每个 AG 一个 barrier”。

4. **sorted dispatch 槽位必须加 local-expert 前缀**：
   - v1 漏掉 `- global_base[(e//Ep)*Ep]`，不同 local expert 写入同一段 buffer，
     A2A 全 case SQNR 约 0dB。正确公式：
     ```
     slot_base[e] = global_base[e] - global_base[(e//Ep)*Ep]
                  + sum_{src<rank} counts_all[src,e]
     ```
   - `counts_all` 是 `all_gather_into_tensor` 输出，第一维是 source rank。

5. **sorted dispatch chunk 参数是 shape 敏感的**：
   - E96：c8 慢，c4 好，c2 又变差。
   - E32：c4 明显差于 c8。
   - E8：c32 稳定，c64 单点好但不稳定，c128 不如 c32/c64。
   - 不要随意全局改 chunks。

6. **direct dispatch 的 ordering invariant 很值钱**：
   - 对 source-block direct dispatch + `order2/inv_order2` 路径，down 返回行序等于
     `send_meta` 顺序，因此源 rank 可省掉回传 meta A2A，只做本地 `send_meta.argsort()`。
   - sorted dispatch 也保持同样 invariant，已用于 E8/E96 回传。

## 评测 / 提交流程

- 提交 payload：
  ```json
  {
    "contestId": 13,
    "problemOrder": 1,
    "content": {
      "language": "triton-dist",
      "code": "<code>",
      "compileAndRunOptions": {}
    }
  }
  ```
- `scripts/submit.py <file> --poll --interval 10 --timeout 1800`
  每次只轮询最近 5 个 submission；并发提交可能让目标 ID 跌出窗口。本会话均为串行提交。
- `scripts/best_score.py` 查的是 scoreboard 缓存，长期返回 113512/46.58；
  **真实最佳看 `POST /api/submission/getSubmissionDetail`**。
- `userOutput` 中 `OJRESULT v1 <hash> <base64json>` 解 base64 得到
  `{"schema_version":2,"tk_time_ms":...,"tb_time_ms":...,"pass":...}`。
- `timeUsed` 约等于 12 个 case 的 `tk_time_ms` 之和（乘以 1000 后的整数），可用于快速核对。

## 评测波动

- 同架构、甚至完全未改动的 case，tk 可以在很大范围波动。
  本会话多次出现单个 case 异常：case4 20.65ms、case5 18.87ms、case9 10.18ms 等。
- 判断优化是否有效应看**同一优化多次重复的单点 tk 信号**，不要只看一次 displayScore。
- 平台保留每题历史最佳，失败/低分提交不会降低 114706 的最佳。
- 当前已 369 次提交，扣罚上限；但不要为碰波动重复提交同一代码。

## 正确性细节

- 预热阶段可以缓存静态权重；`hidden_states` 在正式计时阶段会换多组，不能按 shape 缓存路由结果或 counts。
- 最终归并必须 FP32 累加后转 BF16，才能稳定满足 SQNR≥22dB。
- 输入只读；不要原地修改 `hidden_states` 或权重。
- 确定性要求对同一输入两次运行逐字节一致；所有 argsort 必须 `stable=True`；
  避免用 atomic 顺序影响最终求和顺序的路径。
