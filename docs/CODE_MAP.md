# 当前 v12 源码导航

对应 [p1/kernel.py](../p1/kernel.py) SHA `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`，共7701行。以下均为此 SHA 的 **def 行号**，有重复定义时列出最终有效版本。不是早期5331行清理版；那次清理未晋升。旧导航保存在 [历史 CODE_MAP](../archive/historical-docs/CODE_MAP.pre-v926-20260930.md)。

当前十二案全走 replicated：预热收集全专家静态权重，稳态各卡只处理自己的 X；路由 → 专家分组/metadata → GQ → MD（GU + SwiGLU + 路由加权 + FP8 ACT）→ DN → fin。稳态没有 token 跨卡 dispatch/return。有效分支仍受 `_CALLN` / `_GA` 影响。

| 函数 | v12 行号 | 用途与新方向 |
|---|---:|---|
| `_route_full_kernel` / `_route_full` | 987 / 1051 | router、top-k、归一化 |
| `_gq1p_tm_kernel` | 811 | 量化并写 compact sorted branch |
| `_gq1p_tm_params_kernel` / `_gq1p_tm_params` | 838 / 872 | 已晋升 A 的 AH/WI/ACT_SCALE producer |
| `_gq1p_tok_kernel` | 893 | token-only 量化，含 `_GA` 支路 |
| `_fgs_tma1_kernel_gq_tiled` / `_fgs_tma1_host_tiled` | 1694 / 1868 | c9/c10 tiled GU；旧 B 已实测无收益 |
| `_get_full_fp8_weights_lowmem_tiled` | 3845 | c9/c10 静态权重收集、量化、tile 打包 |
| `_prepare_moe_metadata` | 4173 | 最终有效 metadata 包装，依赖 BM128 |
| `_get_full_weights` | 4546 | 最终有效 full-weight 缓存 |
| `_fgs_t1i_mdq_tma_kernel` | 5624 | 原 TMA-A MD、量化与 ACT 收尾 |
| `_fgs_t1i_mdq_kernel_g` / `_fgs_tma1_intq_host` | 5800 / 5899 | token Q gather MD；**S1 consumer 与 S2 c4 的另一个 producer** |
| `_fgs_t1i_mdq_tma_pre_kernel` / `_fgs_tma1_intq_host_pre` | 5949 / 6013 | H1024 的 A consumer，flatten；**S0 I constexpr** |
| `_fgs_t1i_mdq_tma_pre_nf_kernel` / `_fgs_tma1_intq_host_pre_nf` | 6035 / 6099 | H2048 的 non-flatten A consumer；**S2 c4 必须与 MDg 同时处理** |
| `_fgs_t1i_mdq_pre_kernel` / `_fgs_tma1_intq_host_pre_far` | 6121 / 6183 | H3584/4096 regular-A consumer |
| `_gu_bnorm` | 6291 | 静态 GU bound |
| `_run_replicated` | 6332 | 全链分发；`_dir_a` 条件在6344，DN/fin 在后半段 |
| `run_kernel` | 6875 | 对外入口；6887 增调用计数，6891 设置 `_GA` |
| `_counting_sort_order_invpad` | 7221 | 同时产生 compact INV 与 INV_PAD |
| `_counting_sort_order_packed_compact` | 7410 | c9/c10 compact sort |
| `_get_down_static_scale` | 7468 | 非 tiled DN 静态 C 与 folded B scale |
| `_dn_tma2_f8_pad_static_kernel` / host | 7497 / 7544 | static padded DN；**S2 成对修改 ACT 输入地址**；R1 无正常收益、R3 降级 |
| `_dn_tma2_f8_tiled_static_kernel` / host | 7590 / 7681 | c10 tiled static DN，不能与非 tiled B 混用 |
| `_get_down_static_scale_tiled` | 7650 | c10 tiled 静态 C/folded scale |
| `_gather_branch_sum_f8_kernel` | 4733 | FP8 Down + DSCL 按 j 顺序累加到 BF16；本轮 S1/S2 保持 |
| `_gather_branch_sum_f8_padded` | 7287 | 用 INV_PAD、显式 T 分发 fin |

当前 ACT、AH/WI/ACT_SCALE 是 compact；token-only Q 路径的输入 scale 则按 token 存储。c3–c8/c11/c12 的 DN 输出和 DSCL 是 padded，c9/c10 保持 compact。S2 只把选定 producer 的 ACT 改 padded，动态 ACT 行 scale 仍 compact；host 必须按实际 producer 的布局配对 DN 读址，不能仅凭 shape 推断。S1 保留 token Q，同时在 GQ 生成 compact AH/WI/ACT_SCALE，不能把两种 scale 串用。

当前 c11 DN 参数：BM128/BN256/BK128、GM8、grid132、w8/s4、FLAT=True。当前 c4 MDg 的 outer stages=2、launch stages=3；pre_nf outer=2、launch=4/maxnreg232；DN launch=3。S2 首发保留这些差异。所有新方案与数值边界见 [最新指引](OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)。

函数定义位置不等于调用顺序。上游 Triton3.4 的 JIT cache key 包含起始行，但日志证明线上会改写 host 源码，**提交行号不等于已确认的 JIT 实际行号**。新增 helper 优先追加尾部以减少无关扰动；记录文本、行号、特化参数，不把 AST 相同写成缓存身份相同。guide-candidates 的行号绑定 SID152976，不能套本表。改文件后重新核 SHA。
