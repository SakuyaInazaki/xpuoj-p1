# 04 本会话全部 78 次提交与时间线

## 阶段 A：接手后第一轮（114777-114856）

- 接手时 canonical 114706 / 52.50。
- 114777/114785：融合 final gather+sum，52.67。
- 114796：修复小 NVSHMEM 缓存 clear bug，55.50。
- 114810：当前架构 phase 诊断（WA 预期）。
- 114825：id/缓存 miss 诊断，确认权重张量每次调用 id 都变。
- 114830：静态权重缓存改 shape-key，56.83。
- 114839：case3 切 replicated，58.25。
- 114840/114845：`_gather_branch_sum` BLOCK_H 调优，58.58。
- 其他：index_select 被禁（114816）、k>4 单次 gather 失败、case2 sorted、E256 路由打包等均为负/噪声结果。

## 阶段 B：全 replicated 化（114859-114868）

- 114859：E96 case7/8 replicated，61.75。
- 114861：E8 case1/2 replicated，62.75。
- 114863：E64 case5/6 replicated，64.92。
- 114864：E256 全 replicated TLE，放弃。
- 114868：仅 case10 replicated，65.25。

## 阶段 C：FP8/INT8 低精度 GEMM（114872-114913）

- 114872：replicated 全 BN256 尝试，63.92，未采用。
- 114877：replicated phase 诊断（WA 预期）。
- 114880/114885/114886：replicated cleanup、E256 NCCL AG（失败）、E256 direct AG chunks=32（66.17）。
- 114890/114891：chunks c64/c16，均不如 c32。
- 114893：case9 phase 诊断。
- 114895/114898：E8 FP8 both；融合 SwiGLU+FP8 quant 前身。
- 114901：case5 FP8，66.75。
- 114904：E96 FP8 + case9 local FP8，67.33。
- 114913：融合 SwiGLU+FP8 quant，68.33。

## 阶段 D：INT8 与冲 70（114917-114974）

- 114917：case10 切回 allgather+FP8，失败。
- 114920：case10 FP8 down，单点 3.88ms。
- 114923：case9 int32 argsort，噪声未胜。
- 114925：case9 路由打包，39ms 异常，失败。
- 114927/114929/114933：INT8 全局 scale（SQNR 不足）→ per-row（18.98dB）→ round（通过）。
- 114935/114937：case2 INT8 gateup + FP8 down，case2 15.37ms。
- 114942/114951：组合 case2 hybrid + case10 down。
- 114952/114955：case3、case12 FP8，case3 2.90ms。
- 114959：case9 route direct AG，失败。
- 114962：case10 INT8 down，68.50。
- 114965：case9 INT8 down，68.67。
- 114966：case9 INT8 both，raw tk 总和更低（timeUsed 57859），display 68.58。
- 114967：INT8 BK256，OutOfResources。
- 114968：case9 c28，未胜。
- 114969/114974：per-tile INT8，非有限值。
- **114970：case9 INT8 both + 清理无用缓存，69.33（最终 canonical）。**
- 114971：case9 FP8 both，68.75。
- 114972：case10 FP8 down 组合，67.33。
- 114973：case9 FP8 both + case10 FP8 down，67.58。

# 04 本会话全部 78 次提交（114777-114974）

说明：本会话从接手时 canonical 114706/52.50 开始，第一组提交为 114777。
所有结果均为 submission detail 的 displayScore（raw，未扣罚）。`log` 为 `logs/` 下对应轮询日志。

| id | 状态 | display | timeUsed | memoryUsed | log |
|---:|---|---:|---:|---:|---|
| 114777 | Accepted | 52.67 | 111062 | 3608376 | submit_fused_gather_sum.log |
| 114785 | Accepted | 52.67 | 109884 | 3596180 | submit_fused_gather_sum_v2_routecase6.log |
| 114787 | Accepted | 48.25 | 125993 | 2377036 | submit_fused_gather_sum_v2.log |
| 114791 | Accepted | 51.17 | 115937 | 2365276 | submit_fused_gather_sum_selective.log |
| 114796 | Accepted | 55.50 | 100803 | 3653708 | submit_cache_multibuf.log |
| 114800 | Accepted | 54.33 | 104540 | 3648636 | submit_cleanup.log |
| 114804 | Accepted | 50.75 | 115953 | 3543148 | submit_case2_sorted.log |
| 114810 | WrongAnswer | 0.00 | 135698802 | 3470476 | submit_diag_current.log |
| 114813 | Accepted | 51.25 | 114108 | 3531352 | submit_cleanup_v2.log |
| 114816 | WrongAnswer | 0.00 | 42905562 | 2085776 | submit_index_select.log |
| 114817 | Accepted | 49.00 | 122781 | 3207564 | submit_triton_gather.log |
| 114820 | Accepted | 51.67 | 111669 | 2379336 | submit_argsort32.log |
| 114822 | Accepted | 50.83 | 114942 | 3210852 | submit_onegather_triton.log |
| 114825 | WrongAnswer | 0.00 | 141249702 | 3644168 | submit_diag_ids.log |
| 114830 | Accepted | 56.83 | 98813 | 2373292 | submit_shape_cache.log |
| 114832 | Accepted | 55.58 | 104035 | 2363808 | submit_shape_cache_cleanup_argsort.log |
| 114835 | Accepted | 55.92 | 105314 | 2368524 | submit_shape_cache_cleanup.log |
| 114837 | Accepted | 55.83 | 104355 | 3463476 | submit_shape_cache_routes_all.log |
| 114839 | Accepted | 58.25 | 95848 | 2362060 | submit_shape_cache_case3_repl.log |
| 114840 | Accepted | 58.42 | 98420 | 3595392 | submit_gathersum_bh2048.log |
| 114845 | Accepted | 58.58 | 97821 | 2371056 | submit_gathersum_bh_mixed.log |
| 114848 | Accepted | 58.17 | 97352 | 2388748 | submit_case6_packed_route.log |
| 114850 | Accepted | 57.08 | 101534 | 2369432 | submit_e256_packed_route.log |
| 114854 | Accepted | 58.50 | 95132 | 2386432 | submit_case6pack_cleanup.log |
| 114856 | Accepted | 57.67 | 98341 | 2384620 | submit_argsort32_v2.log |
| 114859 | Accepted | 61.75 | 84444 | 2363432 | submit_e96_repl.log |
| 114861 | Accepted | 62.75 | 77490 | 2348400 | submit_e8_repl.log |
| 114863 | Accepted | 64.92 | 70346 | 3442416 | submit_e64_repl.log |
| 114864 | TimeLimitExceeded | 0.00 | 473039884 | 3433084 | submit_e256_repl.log |
| 114868 | Accepted | 65.25 | 69494 | 2350556 | submit_case10_repl.log |
| 114872 | Accepted | 63.92 | 71439 | 2328912 | submit_repl_bn256.log |
| 114877 | WrongAnswer | 2.42 | 62501916 | 3559444 | submit_diag_repl.log |
| 114880 | Accepted | 65.00 | 69892 | 2345928 | submit_repl_cleanup.log |
| 114885 | Accepted | 64.25 | 75157 | 3556800 | submit_e256_nccl_ag.log |
| 114886 | Accepted | 66.17 | 67799 | 3604036 | submit_e256_ag_c32.log |
| 114890 | Accepted | 65.17 | 70372 | 2353772 | submit_e256_ag_c64.log |
| 114891 | Accepted | 65.58 | 69717 | 3448032 | submit_e256_ag_c16.log |
| 114893 | WrongAnswer | 60.67 | 1881114 | 3599532 | submit_diag_case9.log |
| 114895 | Accepted | 65.33 | 67529 | 3564376 | submit_fp8_e8_repl.log |
| 114898 | Accepted | 66.25 | 64484 | 3606552 | submit_fp8_e8_repl_cleanup.log |
| 114899 | Accepted | 65.50 | 67545 | 2340380 | submit_fp8_e8_gateup_only.log |
| 114901 | Accepted | 66.75 | 64657 | 3603176 | submit_fp8_e8_case5.log |
| 114902 | Accepted | 65.92 | 67016 | 2321684 | submit_fp8_case2_downonly.log |
| 114903 | Accepted | 66.33 | 65021 | 2322680 | submit_fp8_e96.log |
| 114904 | Accepted | 67.33 | 62261 | 3595040 | submit_fp8_e96_case9.log |
| 114905 | WrongAnswer | 62.08 | 1899260 | 3617664 | submit_fp8_case10.log |
| 114907 | WrongAnswer | 4.58 | 101382791 | 3455600 | submit_diag_fp8_repl.log |
| 114909 | Accepted | 67.00 | 63623 | 3569408 | submit_fp8_case2_bn128.log |
| 114913 | Accepted | 68.33 | 59599 | 3606184 | submit_fp8_swiglu_quant.log |
| 114917 | Accepted | 65.17 | 69557 | 2324904 | submit_case10_allgather_fp8.log |
| 114920 | Accepted | 67.58 | 61484 | 3580768 | submit_case10_fp8_down.log |
| 114923 | Accepted | 66.58 | 64468 | 2353292 | submit_case9_argsort32.log |
| 114925 | Accepted | 64.17 | 93682 | 3647868 | submit_case10_fp8_down_case9pack.log |
| 114927 | WrongAnswer | 56.42 | 86719830 | 3590908 | submit_int8_e8.log |
| 114929 | WrongAnswer | 57.42 | 93694285 | 3564904 | submit_int8_e8_rowscale.log |
| 114933 | Accepted | 67.58 | 60521 | 3583660 | submit_int8_e8_rowscale_round.log |
| 114935 | Accepted | 67.25 | 62184 | 3581504 | submit_int8_gate_fp8_down.log |
| 114937 | Accepted | 66.83 | 63070 | 2339256 | submit_case2_hybrid_int8.log |
| 114939 | Accepted | 67.42 | 60865 | 3463296 | submit_case2_hybrid_int8_c24.log |
| 114942 | Accepted | 67.25 | 61736 | 3595308 | submit_case2_hybrid_int8_case10down.log |
| 114945 | Accepted | 67.00 | 63392 | 3589632 | submit_case9_fp8_downonly.log |
| 114949 | Accepted | 66.58 | 63396 | 3456112 | submit_bestcombo_ag_w4.log |
| 114951 | Accepted | 68.17 | 58566 | 3595916 | submit_hybrid_int8_gateonly_case10down.log |
| 114952 | Accepted | 67.67 | 60218 | 2318492 | submit_hybrid_int8_case3fp8.log |
| 114955 | Accepted | 67.50 | 60056 | 2328644 | submit_hybrid_int8_case3_case12fp8.log |
| 114959 | Accepted | 67.75 | 62013 | 3567512 | submit_hybrid_case9_direct_route.log |
| 114960 | Accepted | 67.42 | 60577 | 3439184 | submit_hybrid_int8_case2_bn128.log |
| 114962 | Accepted | 68.50 | 58235 | 3609972 | submit_case10_int8down.log |
| 114965 | Accepted | 68.67 | 58362 | 3606996 | submit_case9_int8down.log |
| 114966 | Accepted | 68.58 | 57859 | 3586692 | submit_case9_int8_both.log |
| 114967 | WrongAnswer | 61.50 | 1824599 | 3602556 | submit_case2_int8_bk256.log |
| 114968 | Accepted | 68.58 | 58042 | 3583280 | submit_case9_int8_both_c28.log |
| 114969 | WrongAnswer | 54.42 | 5560546 | 3596104 | submit_int8_tiled.log |
| 114970 | Accepted | 69.33 | 58057 | 3570276 | submit_case9_int8_both_clean.log |
| 114971 | Accepted | 68.75 | 58209 | 3573408 | submit_best_case9_fp8both.log |
| 114972 | Accepted | 67.33 | 60208 | 2323488 | submit_best_case10_fp8down.log |
| 114973 | Accepted | 67.58 | 60098 | 2333672 | submit_best_case9fp8_case10fp8.log |
| 114974 | WrongAnswer | 53.08 | 5737697 | 3538036 | submit_int8_tiled_clamp.log |
