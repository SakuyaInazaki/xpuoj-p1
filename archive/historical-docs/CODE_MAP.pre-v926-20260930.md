# `p1/kernel.py` 代码导航

适用基线：`p1/kernel.py` SHA-256 `dd46bdebb7be2eed2f1ebe1106be258789bde35e4ee6c9421d5756b163f426b9`。行号只对应这个冻结版本；实验终态与在途队列以 [STATE.md](STATE.md) 为准。

文件保留了若干同名历史定义；Python 以最后一次定义为准。下表均指最后生效项，尤其 `_get_full_weights` 应从 [4194 行](../p1/kernel.py#L4194) 阅读，而不是 4164 行。相同规则还适用于 `_prepare_moe_metadata`（[3821](../p1/kernel.py#L3821)）、`_get_static_cache`（[4023](../p1/kernel.py#L4023)）和 `_run_kernel_a2a`（[4041](../p1/kernel.py#L4041)）。

## 顶层分发

| 入口 | 行号 | 作用 |
|---|---:|---|
| [`run_kernel`](../p1/kernel.py#L5948) | 5948–6200 | 唯一外部入口；5960–5965 设置 call/已知 shape 状态。 |
| replicated 分发 | 5967–6014 | 六组静态 shape 进入 `_run_replicated`。 |
| [`_run_replicated`](../p1/kernel.py#L5517) | 5517–5946 | 全专家权重复制路径：route → 排序/metadata → MD → DN → branch gather。 |
| common：小 `topk` | 6023–6043 | 取本地静态缓存；`k<=4` 进入 `_run_kernel_a2a_sorted` 或最后生效的 `_run_kernel_a2a`。 |
| common：其余 | 6045–6200 | route/all-gather、本地专家 MD/DN、`index_add_`，最后 `reduce_scatter_tensor`。 |

## replicated 主链

| 阶段 | 行号 | 当前入口 |
|---|---:|---|
| 全量权重准备 | 5528–5554 | c9 low-memory FP8 用 [`_get_full_fp8_weights_lowmem`](../p1/kernel.py#L3514)；其余先取最后生效的 [`_get_full_weights`](../p1/kernel.py#L4194)，按 shape 再取 FP8。 |
| route | 5556–5575 | 稳态可走 [`_route_full`](../p1/kernel.py#L978)；另一路为 [`_route_gemm_softmax`](../p1/kernel.py#L1810)；首调保留 torch 组合。 |
| 稳定排序与 metadata | 5577–5632 | [`_counting_sort_order`](../p1/kernel.py#L732)；最终 metadata helper 为 [`_prepare_moe_metadata`](../p1/kernel.py#L3821)。 |
| MD 分发 | 5634–5799 | c9/c10 的 E256 q8 路径用 [`_fgs_tma1_host`](../p1/kernel.py#L1567) 在 1579 launch [`_fgs_tma1_kernel_gq`](../p1/kernel.py#L1492)；c5/c7 等 E≤96 q8 路径用 [`_fgs_tma1_intq_host`](../p1/kernel.py#L5340)，按 `_GA`/`K` 在 5352、5364 或 5374 launch。BF16-act host 为 [`_fgs_tma1_int_host`](../p1/kernel.py#L5421)，E8 稳态在 5680 直接 launch。 |
| DN 分发 | 5800–5939 | 主量化 host 为 [`_dn_tma2_host`](../p1/kernel.py#L1129) → `_dn_tma2_kernel` 1136；FP8-output 变体为 [`_dn_tma2_f8_host`](../p1/kernel.py#L4280) → `_dn_tma2_f8_kernel` 4288；BF16-act 变体入口在 4414/4482。 |
| 最终归并 | 5941–5946 | FP8 down 用 [`_gather_branch_sum_f8`](../p1/kernel.py#L4338)，其余用 [`_gather_branch_sum`](../p1/kernel.py#L513)。 |

- call 3 的 c4/c6/c8/c11 均在 5733–5739 进入 intq q8 MD：c4/c6/c8 由 `_GA` 落到 5352 的 kernel，c11 因 `K<=2048` 落到 5364；两侧 `tl.dot` 操作数均为 FP8。
- 四案生成的 `act_q8+act_rowscl` 在 5814–5823 先于 BF16-act DN 分支被消费，并进入 FP8×FP8 的 `_dn_tma2_f8_host`；c4/c11 虽满足 `use_bf16a_dn`，5834 后分支仍被覆盖。

## common 通信与归并

- 最后生效的 [`_run_kernel_a2a`](../p1/kernel.py#L4041) 在 4091–4109 交换计数/权重/元数据，4144–4156 回传并按 token 求和；sorted 变体入口为 [`_run_kernel_a2a_sorted`](../p1/kernel.py#L2063)。
- common `k>4` 路径在 6055–6079 all-gather token 与 route，6112–6152 执行本地 grouped MD/DN 并写 `partial`，6175–6200 用 `reduce_scatter_tensor` 返回各源 rank。
- 自定义通信 host 入口集中在 [`_direct_a2a`](../p1/kernel.py#L74)、[`_direct_allgather`](../p1/kernel.py#L124) 与 [`_direct_allgather_flat`](../p1/kernel.py#L175)。

## 活跃权重缓存

| 缓存/helper | 行号 | 键 |
|---|---:|---|
| `_FULL_WEIGHT_CACHE` / 最后生效的 `_get_full_weights` | 4191–4219 | 三个权重 shape；执行三次跨 rank all-gather。 |
| `_FULL_FP8_CACHE` / `_get_full_fp8_weights` | 3488–3508 | 三个权重 shape。 |
| `_FULL_FP8_LOWMEM_CACHE` / `_get_full_fp8_weights_lowmem` | 3511–3561 | 三个权重 shape。 |
| `_STATIC_CACHE` / 最后生效的 `_get_static_cache` | 4020–4037 | Python `id`、shape 与 `topk`；缓存本地拼接 `gate_up`。 |
| `_STATIC_INT8_GATEUP_CACHE`、`_STATIC_INT8_DOWN_CACHE` | 3585–3610 | shape；down 另含 `topk`。 |
| `_BNORM_CACHE` / `_gu_bnorm`，`_INT_GU_CACHE` / `_get_int_gu` | 5457–5514 | 量化权重 shape；后者另含 granularity。 |

这些派生权重缓存的更新边界见 [STATE.md](STATE.md)：题面只保证同一测试点内权重静态，不能把 shape 命中解释为跨测试点权重恒定。
