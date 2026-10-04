# 02 本会话时间线与全部尝试

会话开始时 canonical 是 submission **114382 / 51.08**。本会话共产生 37 个 submission ID
（114627 至 114758，中间不全连续）。所有分数为 submission detail 的 `displayScore`。

## 阶段 A：packweight / no-meta 路线（114627-114635）

| ID | 文件 | 结果 | 说明 |
|---:|---|---:|---|
| 114627 | `kernel_c128_packe8_nometa96.py` | Accepted 48.08 | E8 weight+local int64 单路 A2A，去掉回传 meta；E96 只发 int32 local。case2 tk 22.996 明显优于 114382 的 25.732。 |
| 114629 | `kernel_c128_packe8_case2_nometa.py` | Accepted 49.67 | 仅 case2 启用 pack+nometa，其余与 114382 相同。case2 tk 24.15；case5 本轮异常 18.87ms。 |
| 114631 | `kernel_fastag_k4gt.py`（try/except 版） | WA 0 | **沙箱不接受 try/except**，Language validation 直接崩。 |
| 114635 | `kernel_fastag_k4gt.py`（直接导入） | WA 33.08 | 官方 `fast_allgather(push2d)` 首轮正确，正式计时段 k>4 全部 `tk=0, pass=false`；放弃。 |

关键结论：
- case2 的 E8 packweight+no-meta 多次重复验证快 1.6~2.8ms，是有效优化。
- 不要写 try/except；不要使用官方 fast_allgather push2d。

## 阶段 B：自写 NVSHMEM direct all-gather（114651-114668）

| ID | 文件 | 结果 | 说明 |
|---:|---|---:|---|
| 114651 | `kernel_directag_k4gt.py`（v1：先 copy 本地块、kernel 跳过本地 put） | WA 33.92 | 首轮正确，正式计时段 k>4 全部失败。 |
| 114658 | `kernel_directag_k4gt_v2.py`（kernel 写全部 rank block，无前置 copy） | Accepted 50.67 | v2 稳定。逐点见 JSON。 |
| 114663 | `kernel_c128_packe8e96_nometa.py` | Accepted 47.83 | E8/E96 全 pack；本轮 case4 异常 26.24ms。 |
| 114668 | `kernel_directag_v2_case2pack.py` | **Accepted 51.58** | direct AG v2 + 仅 case2 pack/nometa。**第一次晋升 canonical**。 |

关键结论：
- direct all-gather v1 的“前置 local copy + 跳过本地 put”写法会在正式计时段挂掉；
  v2 必须让 kernel 写全部 rank block（含本地），每个 AG 后接 `nvshmem_barrier_all_on_stream()`。
- 114668 的 direct AG 替换 k>4 hidden allgather 有效。

## 阶段 C：sorted direct dispatch（114681-114706）

| ID | 文件 | 结果 | 说明 |
|---:|---|---:|---|
| 114681 | `kernel_directag_v2_case2pack_diag.py` | WA 0（预期） | 第二次调用 raise，拿到较干净 phase profile。 |
| 114684 | `kernel_sorted_dispatch.py`（v1） | WA 31.83 | A2A 全 fail。原因：槽位计算漏加“local expert 前缀”，不同 local expert 写入重叠。 |
| 114692 | `kernel_sorted_dispatch_v2.py` | Accepted 50.17 | 修复 slot base 后正确。E8/E32 改善，E96 c8 慢。 |
| 114693 | `kernel_sorted_dispatch_e32only.py` | Accepted 49.17 | 仅 E32 sorted c8；case3 7.757（114668 为 8.215）。 |
| 114694 | `kernel_sorted_e8case1_e32.py` | Accepted 49.67 | E32 + E8 case1 sorted；case1/3 有改善但整体噪声低。 |
| 114697 | `kernel_sorted_e96only_c4.py` | **Accepted 52.17** | E96 sorted chunks 从 8 改成 4 明显变好。**第二次晋升 canonical**。 |
| 114701 | `kernel_sorted_hybrid_v1.py` | Accepted 50.75 | E96 c4 + E32 c8 + E8 case1 c32。case3 7.37 很好，其他噪声拖低。 |
| 114702 | `kernel_sorted_e32_c4.py` | Accepted 49.33 | E32 c4 明显差（case3 10.16），c8 保持。 |
| 114704 | `kernel_sorted_e96c4_e32c8.py` | Accepted 47.25 | 本轮 case4/case12 异常，分数不可信。 |
| 114706 | `kernel_sorted_v3_hybrid.py` | **Accepted 52.50** | v3：删 local/src put，E8/E96 只 put token+weight，E32 用计数本地重建 src；E96 c4 / E32 c8 / E8 case1 c32；case2 保留旧 pack。**最终 canonical**。 |

关键结论：
- sorted dispatch 思路：all_gather 全局专家计数 -> 计算每个 global expert 在目标 rank
  本地专家序列中的精确槽位 -> 源 rank 直接 put 到最终排序位置。
- 槽位公式必须包含 local-expert 前缀：
  `slot_base = global_base[e] - global_base[(e//Ep)*Ep] + counts_all[:rank,e].sum()`。
- v1 漏掉中间减项，SQNR 约 0dB。
- E96 chunks 扫描：c8 < c4 < c2，c4 最优。
- E32 c4 差于 c8；E8 c32 暂保持。

## 阶段 D：chunks 扫描与变体（114711-114734）

| ID | 文件 | 结果 | 说明 |
|---:|---|---:|---|
| 114711 | `kernel_sorted_v3_e96c2.py` | Accepted 49.67 | E96 c2，case7/8 差于 c4。 |
| 114712 | `kernel_sorted_v3_e8c64.py` | Accepted 49.67 | E8 case1 c64；case1 14.74 单点很好，其他噪声。 |
| 114713 | `kernel_sorted_v3_e8c128.py` | Accepted 52.33 | case1 16.74，不如 c32/c64 稳定。 |
| 114718 | `kernel_sorted_v4_e32pack.py` | Accepted 51.67 | E32 weight/meta 打包 int64；case3 8.14，未超 v3。 |
| 114721 | `kernel_sorted_v4_e8c64_e32pack.py` | Accepted 50.33 | 组合后噪声低。 |
| 114727 | `kernel_sorted_select_ag.py` | Accepted 48.42 | k>4 先 sort 小索引再单次 token gather（理论少一次大 gather）。 |
| 114728 | `kernel_sorted_v3_e8c64_e32c16.py` | Accepted 49.75 | c16 无明显收益。 |
| 114730 | `kernel_sorted_select_ag_e8c64.py` | Accepted 47.42 | 噪声大。 |
| 114732 | `kernel_sorted_e8both_c64.py` | Accepted 47.83 | E8 全 sorted c64；本轮 case4 异常 20.65ms。 |
| 114734 | `kernel_sorted_e8both_c64_selectsort.py` | Accepted 51.00 | 组合仍受波动影响。 |

## 阶段 E：route direct all-gather（114741-114758）

| ID | 文件 | 结果 | 说明 |
|---:|---|---:|---|
| 114741 | `kernel_directag_routes.py` | Accepted 52.33 | k>4 ids/weights/pack 全部改为自写 flat direct AG。case6 6.54 明显改善。 |
| 114743 | `kernel_directag_routes_onebarrier.py` | WA 48.08 | 多个 AG launch 后只做一次 barrier：case5 packed route SQNR/determinism 失败。 |
| 114752 | `kernel_directag_routes_nonpack.py` | Accepted 51.50 | 仅非 packed k>4 路由 direct AG；case6 7.34 继续改善。 |
| 114755 | `kernel_directag_routes_selectsort.py` | Accepted 48.25 | direct AG routes + 单次 token gather；本轮波动大。 |
| 114758 | `kernel_directag_routes_case6.py` | Accepted 49.75 | 仅 case6 启用 route direct AG；case6 6.84，第三次重复改善。 |

## 本会话逐点数据

完整逐点 `tk/tb/displayScore/status` 已存到 `session_submissions_raw.json`。
