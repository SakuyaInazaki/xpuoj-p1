# 04 本 agent 会话完整时间线（submission 115818-115966）


本 agent 从 `handoff-session-20260817-final/`（115705 平台最佳 / 115738 实际最快）接手，
共 61 次提交（115818-115966），最终 scoreboard 最佳 115950（raw 74.08），
稳定实际最快 115907（raw 74.00 / timeUsed 43538）。

## 阶段概览

1. **115818-115825：修复 persistent + 去 mask**
   - 修复 FP8 down persistent launch 语法；删 custom GEMM 的 K/N mask；
   - 115825 raw 71.42 / timeUsed 48912。
2. **115834：自写 activation amax kernel**
   - 单遍 read + atomic_max 替代 `a.abs().max()` 物化；
   - 115834 raw 71.83 / timeUsed 47532，本会话最大单项收益之一。
3. **115839-115850：case2 FP8 化 + 小 case FP8 化**
   - case2 gateup INT8 -> plain FP8；case4/6/11 也切 FP8；
   - 115850 raw 72.58 / timeUsed 45801。
4. **115854：case2 SwiGLU BN256**
   - 115854 raw 72.92 / timeUsed 45372。
5. **115905-115907：gather 与 amax 融合**
   - 新增 `_gather_tokens_amax_kernel`；case2 保持 hybrid；
   - 115907 raw 74.00 / timeUsed 43538，成为稳定 base。
6. **115943-115966：边际扫描**
   - case2 gather 变体、BM64 metadata、down slots、split gateup、route 变体等；
   - 多数变慢或 TLE；115950 靠 case2 tb 异常成为 scoreboard 最佳。

## 全部提交表

| id | 状态 | display | timeUsed | 日志 | 说明 |
|---:|---|---:|---:|---|---|
| 115818 | Accepted | 70.92 | 50115 | submit_fp8_down_persistent_v2.log | 修复 FP8 down persistent launch bug |
| 115819 | Accepted | 71.08 | 49629 | submit_nokmask_v2.log | 删 custom GEMM K-remainder mask |
| 115823 | Accepted | 71.25 | 49588 | submit_nokmask_persistdown_v3.log | nokmask + persistent down |
| 115825 | Accepted | 71.42 | 48912 | submit_nomasks_v4.log | 删 N-col/quant mask |
| 115830 | Accepted | 71.17 | 49689 | submit_v5_route_swiglu_nomask.log | route/SwiGLU 去 mask（变慢） |
| 115833 | Accepted | 71.58 | 48902 | submit_v6_fused_persistent.log | fused gateup persistent（打平） |
| 115834 | Accepted | 71.83 | 47532 | submit_v7_amax_kernel.log | 自写 FP8 activation amax kernel |
| 115835 | Accepted | 71.17 | 49799 | submit_v8_int32_sort.log | route ids int32 argsort（变慢） |
| 115836 | Accepted | 72 | 49734 | submit_v9_bf16_persistent.log | 官方 BF16 num_sms=132（实际变慢，raw 靠 tb 异常） |
| 115839 | Accepted | 71 | 48864 | submit_v10_case2_plain_fp8.log | case2 gateup 改 plain FP8 |
| 115841 | Accepted | 72.17 | 46168 | submit_v11_amax_case2fp8.log | amax + case2 plain FP8 |
| 115843 | Accepted | 71.75 | 47225 | submit_v12_quant_bk256.log | quant BLOCK_K=256（变慢） |
| 115848 | TimeLimitExceeded | 0 | 473340389 | submit_v13_case2_fused_fp8.log | case2 full-fused FP8（TLE） |
| 115850 | Accepted | 72.58 | 45801 | submit_v15_fp8_smallbf16.log | case4/6/11 也切 FP8 |
| 115854 | Accepted | 72.92 | 45372 | submit_v16_swiglu_bn256.log | case2 SwiGLU BN256（上阶段最佳） |
| 115856 | Accepted | 72.33 | 48199 | submit_v17_case2_fused_seq.log | case2 两段式 fused gateup（慢） |
| 115859 | TimeLimitExceeded | 0 | 473638073 | submit_v18_swiglu_bn512.log | case2 SwiGLU BN512（TLE） |
| 115860 | Accepted | 72.75 | 45777 | submit_v19_case2_gm16.log | case2 gateup GROUP_M=16（慢） |
| 115861 | Accepted | 69.17 | 52889 | submit_v20_fused_s2.log | fused gateup num_stages=2（慢） |
| 115866 | Accepted | 72.67 | 46065 | submit_v21_sigmoid.log | SwiGLU 改 tl.sigmoid（慢） |
| 115876 | Accepted | 72.5 | 45779 | submit_v23_swiglu_bm64.log | case2 SwiGLU BM64（慢） |
| 115878 | Accepted | 72.75 | 46599 | submit_v24_case2_down_bn128.log | case2 down BN128（慢） |
| 115879 | Accepted | 71.58 | 48664 | submit_v25_case2_gateup_bn128.log | case2 gateup BN128（慢） |
| 115892 | WrongAnswer | 0 | 102890396 | submit_diag_v16_phases.log | phase 诊断 v16 |
| 115895 | Accepted | 72.92 | 45501 | submit_v26_case2_gateup_nonpersistent.log | case2 gateup nonpersistent（慢） |
| 115896 | WrongAnswer | 35.83 | 47858949 | submit_v27_route_e256_bm64.log | route E256 BM64（有 bug，WA） |
| 115898 | WrongAnswer | 0 | 69654854 | submit_diag_v16_case2_phases.log | case2 细 phase 诊断 |
| 115899 | Accepted | 72.67 | 47787 | submit_v29_case2_gateup_s2.log | case2 gateup s2（慢） |
| 115900 | Accepted | 72.08 | 48169 | submit_v28_case2_gateup_bk64.log | case2 gateup BK64（慢） |
| 115903 | Accepted | 72.67 | 45864 | submit_v30_quant_bm256.log | quant BM256（慢） |
| 115905 | Accepted | 73.33 | 43978 | submit_v31_gather_amax.log | 全 case gather+amax（有效） |
| 115907 | Accepted | 74 | 43538 | submit_v32_gather_amax_hybrid.log | hybrid gather+amax（稳定最佳） |
| 115908 | Accepted | 73.08 | 44556 | submit_v33_route_e256_bk128.log | route E256 BK128（慢） |
| 115911 | Accepted | 73.25 | 44748 | submit_v34_gather_bh256.log | gather BH256（慢） |
| 115915 | Accepted | 34.25 | 299912 | submit_v35_bt_fixed.log | B 转置修复 stride（正确但极慢） |
| 115918 | Accepted | 73.42 | 44100 | submit_v36_gather_w4.log | gather w4（慢） |
| 115919 | WrongAnswer | 67.58 | 1822368 | submit_v37_case2_gateup_bm64.log | case2 gateup BM64 未同步 metadata（WA） |
| 115922 | Accepted | 73.75 | 44101 | submit_v38_swiglu_w16.log | case2 SwiGLU w16（慢） |
| 115923 | Accepted | 73.5 | 44131 | submit_v39_route_e256_bm64bk128.log | route E256 BM64/BK128（慢） |
| 115924 | Accepted | 72.92 | 44936 | submit_v40_down_bn128_h1024.log | down H1024 BN128（慢） |
| 115927 | TimeLimitExceeded | 0 | 474235374 | submit_v42_case2_fused_bn64.log | case2 full-fused BN64（TLE） |
| 115929 | Accepted | 73.33 | 46617 | submit_v43_swiglu_w4.log | case2 SwiGLU w4（慢） |
| 115930 | Accepted | 73.42 | 44083 | submit_v44_case2_xamax.log | case2 用 x 的 amax（慢） |
| 115933 | Accepted | 73.42 | 43938 | submit_v46_gather_noamax_xamax.log | gather 不带 amax + 单独 x amax（慢） |
| 115935 | Accepted | 73.92 | 44122 | submit_v47_case2_down_nonpersistent.log | case2 down nonpersistent（慢） |
| 115937 | Accepted | 73.83 | 43916 | submit_v48_case2_gateup_gm4.log | case2 gateup GM4（慢） |
| 115939 | WrongAnswer | 6.08 | 46859481 | submit_v49_fused_bm64_meta.log | fused BM64+metadata64（tuple bug） |
| 115942 | WrongAnswer | 0 | 39444914 | submit_v49_fused_bm64_meta_r2.log | fused BM64+metadata64（TLE） |
| 115943 | Accepted | 73.83 | 43467 | submit_v50_case2_gather_bm256.log | case2 gather BM256/H128/w8（单次最快 43467） |
| 115944 | WrongAnswer | 6.17 | 85191148 | submit_v51_fused_gather_quant.log | fused gather+FP8 quant（pointer bug） |
| 115946 | TimeLimitExceeded | 0 | 473757890 | submit_v51_fused_gather_quant_r2.log | fused gather+FP8 quant（TLE） |
| 115950 | Accepted | 74.08 | 44017 | submit_v52_case2_gather_bm256_w4.log | case2 gather BM256/H128/w4（scoreboard 最佳，tb 异常） |
| 115951 | Accepted | 73.5 | 44023 | submit_v53_gather_bm256_all.log | 全 case gather BM256（慢） |
| 115954 | Accepted | 73.08 | 44542 | submit_v54_case2_gather_bm256_bh64.log | case2 gather BM256/H64（慢） |
| 115956 | Accepted | 73.5 | 44571 | submit_v55_case2_gather_bm512.log | case2 gather BM512/H64（慢） |
| 115957 | Accepted | 73.25 | 44517 | submit_v50_case2_gather_bm256_r2.log | v50 复测（波动） |
| 115959 | WrongAnswer | 6.25 | 20374306 | submit_v56_case2_gateup_bm64_meta.log | case2 plain GEMM BM64+metadata64（TLE） |
| 115961 | Accepted | 73.5 | 44406 | submit_v58_down_persistent264.log | down persistent grid264（慢） |
| 115964 | Accepted | 73.5 | 44092 | submit_v59_route_e256_bn64.log | route E256 BN64（慢） |
| 115965 | Accepted | 73.33 | 45099 | submit_v60_down_slots.log | down 写 [T,k,H] slot + final 顺序归约（慢） |
| 115966 | TimeLimitExceeded | 0 | 473610119 | submit_v61_case2_split_gateup.log | case2 gate/up 拆两个 GEMM（TLE） |

详细逐点 tk/tb 和 userError tail 见 `submissions_raw_all.json`。