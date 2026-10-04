# 04 本会话完整时间线（submission 116944-117310）

## 阶段概览

1. **116944-117019：final gather tiling 主线**
   - 发现历史判为“已最优”的 `_gather_branch_sum` 仍可 2D tiling。
   - v162/v163/v164/v171/v172 逐步晋升；最终 case1/2 BT8/BH1024/w8，其他 BT32/BH1024/w32。
2. **117021-117029：case2 gateup/route 盲扫**
   - gateup grid264 无稳定收益；route N96 BK128 暂不采用。
3. **117030-117046：case2 结构化失败**
   - 直接 FP8 gate/up 输出有 Python bug；dual interleave/amax 明显慢；BN256/w16 OOR。
4. **117050-117084：route + case2 SwiGLU 细化**
   - route N96 BM64/BK128 有效；N96 w4 有效；case2 SwiGLU BN128 -> BN64 有效；down w16/w4 慢。
5. **117134-117180：fused gateup 与 route 收尾**
   - GM32/s4 有效；GM64/GM32+s3/grid264 慢；E8 s4/BM32、E64 变体无稳定收益。
6. **117182-117244：case2 token gather 与 token quant**
   - case2 gather BM64 -> BM64/BH256 两窗稳定晋升。
   - token quant BK64 经 4 窗 tie-break（3 窗 case2 更快）晋升。
   - row gather BM64/BH256 两/三窗互斥，关闭。
7. **117287-117310：收尾盲扫**
   - E32 BN16、H1024 final/gather 变体、token quant BM64/BK64 均无稳定收益。

## 判断纪律（本会话最重要的经验）

- 第一窗候选大幅领先经常是噪声；必须同窗口 A/B 至少两对，必要时 3-4 窗。
- 只有当前 `p1/kernel.py` 同窗口对照才可信。
- raw/tb 异常不能作为晋升依据；只看 tk/timeUsed。

# 04 本会话完整时间线（submission 116944-117310）

本会话共 172 次提交（含同窗口对照）。完整逐点数据见 `submissions_raw_116944-117310.json`。

| id | 状态 | raw | timeUsed | 日志 | 备注 |
|---:|---|---:|---:|---|---|
| 116944 | Accepted | 74.50 | 42308 | submit_v162_final_tiled.log | v162 final gather BT8/BH512 首试，有效 |
| 116950 | Accepted | 74.33 | 43228 | submit_v162_control.log | 旧 v159 同窗口对照 [对照] |
| 116952 | Accepted | 75.08 | 41651 | submit_v163_final_tiled_hblock.log | v163 final tiling 保留原 BH；大赚，有效 |
| 116953 | Accepted | 73.92 | 43708 | submit_v163_control_v162.log | v162 对照 [对照] |
| 116954 | Accepted | 74.00 | 43137 | submit_v163_final_tiled_hblock_r2.log | v163 复测 |
| 116955 | Accepted | 73.75 | 43772 | submit_v163_r2_control.log | v159 对照 [对照] |
| 116957 | Accepted | 74.00 | 42839 | submit_v164_final_tiled_aggr.log | v164 final aggr；晋升中间态 |
| 116960 | Accepted | 73.83 | 43278 | submit_v164_control_v163.log | v163 对照 [对照] |
| 116961 | Accepted | 76.67 | 42225 | submit_v165_final_tiled_aggr2.log | v165 全 BT8/BH1024；case5 tb 异常成 scoreboard best |
| 116972 | Accepted | 74.50 | 42182 | submit_v167_final_tiled_nomask.log | v167 final no-mask；无额外收益 |
| 116976 | Accepted | 75.00 | 42157 | submit_v167_control_v164.log | v164 对照复投 [对照] |
| 116981 | Accepted | 74.17 | 43233 | submit_v168_route_e32_bk128.log | v168 route E32 BK128；初步有效 |
| 116982 | Accepted | 73.67 | 43444 | submit_v168_control.log | v168 对照 [对照] |
| 116983 | Accepted | 73.75 | 43281 | submit_v169_route_e64_bk128.log | v169 route E64 BK128 首窗 |
| 116984 | Accepted | 75.00 | 41597 | submit_v169_control_v168.log | v168 对照（快窗） [对照] |
| 116986 | Accepted | 73.58 | 43589 | submit_v168_r2_control_first.log | v168 反向窗对照 [对照] |
| 116987 | Accepted | 74.00 | 43043 | submit_v168_r2_candidate.log | v168 反向窗候选 |
| 116989 | Accepted | 74.00 | 42770 | submit_v169_r2_candidate.log | v169 r2 |
| 116990 | Accepted | 74.00 | 43466 | submit_v169_r2_control_v168.log | v168 对照 [对照] |
| 116992 | Accepted | 76.25 | 41712 | submit_v170_final_bt16_w16.log | v170 final BT16/w16；有效 |
| 116996 | Accepted | 74.17 | 42980 | submit_v170_control.log | v169 对照 [对照] |
| 116999 | Accepted | 74.25 | 43001 | submit_v170_r2_control_first.log | v170 反向窗对照 |
| 117000 | Accepted | 74.08 | 42873 | submit_v170_r2_candidate.log | v170 反向窗候选 |
| 117001 | Accepted | 74.92 | 41788 | submit_v171_final_bt16_keep_case12.log | v171 final BT16 + case1/2 保持；晋升中间态 |
| 117002 | Accepted | 74.08 | 43126 | submit_v171_control_v170.log | v170 对照 [对照] |
| 117003 | Accepted | 74.33 | 42293 | submit_v171_r2.log | v171 r2 |
| 117008 | Accepted | 74.08 | 42998 | submit_v171_r2_control_v169.log | v169 对照 [对照] |
| 117009 | Accepted | 74.50 | 42779 | submit_v172_final_bt32_w32.log | v172 final BT32/w32；晋升中间态 **采用** |
| 117011 | Accepted | 74.25 | 42884 | submit_v172_control_v171.log | v171 对照 [对照] |
| 117013 | Accepted | 73.67 | 43551 | submit_v172_r2_control_first.log | v172 反向窗对照 [对照] |
| 117015 | Accepted | 74.08 | 42985 | submit_v172_r2_candidate.log | v172 反向窗候选 |
| 117018 | Accepted | 73.83 | 44160 | submit_v173_final_bt64_w32.log | v173 final BT64/w32；慢，关闭 |
| 117019 | Accepted | 73.75 | 43547 | submit_v173_control_v172.log | v172 对照 [对照] |
| 117021 | Accepted | 74.00 | 43183 | submit_v174_gateup_grid264.log | v174 case2 gateup grid264 首窗 |
| 117022 | Accepted | 73.75 | 43478 | submit_v174_control_v172.log | v172 对照 [对照] |
| 117023 | Accepted | 73.75 | 43626 | submit_v174_r2_control_first.log | v174 反向窗对照 [对照] |
| 117024 | Accepted | 73.50 | 43669 | submit_v174_r2_candidate.log | v174 反向窗候选；关闭 |
| 117025 | Accepted | 74.25 | 42919 | submit_v175_route_n96_bk128.log | v175 route N96 BK128 首窗 |
| 117026 | Accepted | 73.67 | 43449 | submit_v175_control.log | v172 对照 [对照] |
| 117027 | Accepted | 73.50 | 43510 | submit_v175_r2_control_first.log | v175 反向窗对照 [对照] |
| 117029 | Accepted | 73.92 | 43102 | submit_v175_r2_candidate.log | v175 反向窗候选；暂不采用 |
| 117030 | WrongAnswer | 68.25 | 1831266 | submit_v176_case2_fp8gateup.log | v176 case2 gate/up 直接 FP8；UnboundLocal bug WA |
| 117032 | TimeLimitExceeded | 0.00 | 2147483647 | submit_v177_case2_fp8gateup_fixed.log | v177 修复后候选；最终 TLE，未作 base |
| 117038 | Accepted | 73.58 | 43603 | submit_v178_case12_bt32.log | v178 case1/2 也 BT32；无差 |
| 117041 | Accepted | 73.75 | 43611 | submit_v178_control.log | v172 对照 |
| 117043 | Accepted | 73.92 | 45383 | submit_v179_case2_dual_interleave.log | v179 case2 dual interleave；case2 13.286 严重慢 |
| 117045 | Accepted | 73.75 | 45294 | submit_v181_case2_dual_interleave_amax.log | v181 case2 dual+amax；case2 12.446 慢 |
| 117046 | WrongAnswer | 67.92 | 1808724 | submit_v180_case2_dual_interleave_bn256w16.log | v180 dual BN256/w16；OutOfResources |
| 117050 | Accepted | 73.67 | 43403 | submit_v184_route_n96_bm64_bk128.log | v184 route N96 BM64/BK128；晋升中间态 **采用** |
| 117051 | Accepted | 73.58 | 43563 | submit_v186_final_h1024_bt64.log | v186 final H1024 BT64；无收益 |
| 117055 | Accepted | 73.92 | 43186 | submit_v192_swiglu_bn128.log | v192 case2 SwiGLU BN128；有效 |
| 117058 | Accepted | 73.75 | 43649 | submit_v184_control.log | v184 对照 [对照] |
| 117061 | Accepted | 73.58 | 43619 | submit_v192_control_v184.log | v192 对照 v184 [对照] |
| 117064 | Accepted | 74.00 | 43529 | submit_v196_base_swiglu_bn128.log | v196 = v184+BN128 组合 |
| 117065 | Accepted | 73.67 | 43521 | submit_v196_control.log | v184 对照 [对照] |
| 117068 | Accepted | 73.75 | 43407 | submit_v197_route_n96_bm64_bn64.log | v197 N96 BM64/BN64 首窗 |
| 117069 | Accepted | 73.58 | 43573 | submit_v197_control.log | v197 对照 [对照] |
| 117072 | Accepted | 73.92 | 43642 | submit_v197_r2_control_first.log | v197 反向窗对照 [对照] |
| 117073 | Accepted | 73.58 | 43663 | submit_v197_r2_candidate.log | v197 反向窗候选；无稳定收益 |
| 117077 | Accepted | 74.75 | 42795 | submit_v199_swiglu_bn64.log | v199 SwiGLU BN64；晋升中间态 **采用** |
| 117078 | Accepted | 73.67 | 43645 | submit_v199_control.log | v199 对照 [对照] |
| 117083 | Accepted | 73.75 | 43831 | submit_v194b_down_w16_only.log | v194b down w16；慢 |
| 117084 | Accepted | 73.67 | 43598 | submit_v194b_control.log | v194b 对照 [对照] |
| 117134 | Accepted | 73.67 | 43599 | submit_v199_r2_control_first.log | v199 反向窗对照 [对照] |
| 117135 | Accepted | 73.92 | 43137 | submit_v199_r2_candidate.log | v199 反向窗候选 **采用** |
| 117137 | Accepted | 74.08 | 43335 | submit_v206_swiglu_bn32.log | v206 SwiGLU BN32；case2 变慢，关闭 |
| 117138 | Accepted | 73.83 | 43373 | submit_v206_control.log | v199 对照 [对照] |
| 117141 | Accepted | 74.25 | 42624 | submit_v203_final_h3584_bh512.log | v203 H3584 BH512 旧 base 版 |
| 117142 | Accepted | 74.17 | 43258 | submit_v203b_final_h3584_bh512.log | v203b H3584 BH512 新 base 版 |
| 117143 | Accepted | 74.08 | 43170 | submit_v203b_control.log | v203b 对照 [对照] |
| 117145 | Accepted | 74.33 | 42999 | submit_v198b_route_n96_bm128_bn64.log | v198b N96 BM128/BN64 首窗 |
| 117146 | Accepted | 73.58 | 43702 | submit_v198b_control.log | v198b 对照 [对照] |
| 117148 | Accepted | 73.50 | 43724 | submit_v198b_r2_control_first.log | v198b 反向窗对照 [对照] |
| 117149 | Accepted | 73.92 | 43203 | submit_v198b_r2_candidate.log | v198b 反向窗候选；无稳定收益 |
| 117151 | Accepted | 75.75 | 43002 | submit_v205b_route_n96_w4.log | v205b N96 w4；晋升中间态 **采用** |
| 117153 | Accepted | 73.75 | 43709 | submit_v205b_control.log | v205b 对照 [对照] |
| 117154 | Accepted | 73.67 | 43707 | submit_v205b_r2_control_first.log | v205b 反向窗对照 [对照] |
| 117155 | Accepted | 74.17 | 43179 | submit_v205b_r2_candidate.log | v205b 反向窗候选 **采用** |
| 117157 | Accepted | 74.25 | 42966 | submit_v188b_fused_gm32.log | v188b fused GM32；晋升中间态 **采用** |
| 117158 | Accepted | 73.67 | 43517 | submit_v188b_control.log | v188b 对照 [对照] |
| 117159 | Accepted | 73.58 | 43686 | submit_v188b_r2_control_first.log | v188b 反向窗对照 [对照] |
| 117160 | Accepted | 74.17 | 43215 | submit_v188b_r2_candidate.log | v188b 反向窗候选 **采用** |
| 117161 | Accepted | 73.67 | 43458 | submit_v191b_route_e8_s4.log | v191b route E8 s4；慢 |
| 117162 | Accepted | 74.08 | 43166 | submit_v191b_control.log | v191b 对照 [对照] |
| 117164 | Accepted | 74.33 | 43306 | submit_v200b_gateup_w16.log | v200b case2 gateup w16；case2 变慢 |
| 117165 | Accepted | 73.67 | 43750 | submit_v200b_control.log | v200b 对照 [对照] |
| 117166 | Accepted | 56.17 | 108642 | submit_v193c_down_w4_only.log | v193c down w4；108642 明显慢 |
| 117167 | Accepted | 73.58 | 43751 | submit_v193c_control.log | v193c 对照 [对照] |
| 117168 | Accepted | 73.92 | 43400 | submit_v207_fused_gm64.log | v207 fused GM64；慢 |
| 117169 | Accepted | 74.00 | 43255 | submit_v207_control.log | v207 对照 [对照] |
| 117170 | Accepted | 74.00 | 43170 | submit_v208_route_e8_bm32.log | v208 route E8 BM32 首窗 |
| 117171 | Accepted | 73.50 | 43757 | submit_v208_control.log | v208 对照 [对照] |
| 117172 | Accepted | 74.08 | 43222 | submit_v208_r2_control_first.log | v208 反向窗对照 [对照] |
| 117173 | Accepted | 74.83 | 43688 | submit_v208_r2_candidate.log | v208 反向窗候选；无稳定收益 |
| 117174 | Accepted | 74.17 | 42850 | submit_v209_token_quant_bk64.log | v209 token quant BK64 首窗 |
| 117175 | Accepted | 73.58 | 43753 | submit_v209_control.log | v209 对照 [对照] |
| 117177 | Accepted | 73.92 | 43219 | submit_v209_r2_control_first.log | v209 反向窗对照 [对照] |
| 117178 | Accepted | 73.83 | 43383 | submit_v209_r2_candidate.log | v209 反向窗候选；暂不采用 |
| 117179 | Accepted | 73.92 | 43444 | submit_v210_fused_gm32_s3.log | v210 GM32+s3；慢 |
| 117180 | Accepted | 74.00 | 43246 | submit_v210_control.log | v210 对照 [对照] |
| 117182 | WrongAnswer | 0.00 | 216291654 | submit_diag_v199_phases.log | 当前 base phase 诊断 WA（userError 有 phase 数据） |
| 117183 | Accepted | 73.83 | 43450 | submit_v211_token_quant_bk32.log | v211 token quant BK32；case2 变慢 |
| 117184 | Accepted | 74.08 | 43228 | submit_v211_control.log | v211 对照 [对照] |
| 117188 | Accepted | 74.67 | 42887 | submit_v212_token_quant_bm64.log | v212 token quant BM64 首窗 |
| 117189 | Accepted | 73.67 | 43743 | submit_v212_control.log | v212 对照 [对照] |
| 117190 | Accepted | 74.42 | 43255 | submit_v212_r2_control_first.log | v212 反向窗对照 [对照] |
| 117191 | Accepted | 73.75 | 43382 | submit_v212_r2_candidate.log | v212 反向窗候选；不采用 |
| 117193 | Accepted | 73.67 | 43711 | submit_v213_fused_grid264.log | v213 fused grid264；慢 |
| 117194 | Accepted | 74.00 | 43189 | submit_v213_control.log | v213 对照 [对照] |
| 117195 | Accepted | 74.75 | 42484 | submit_v214_final_case910_bt16.log | v214 case9/10 final BT16 首窗 |
| 117196 | Accepted | 73.75 | 43711 | submit_v214_control.log | v214 对照 [对照] |
| 117197 | Accepted | 74.17 | 43159 | submit_v214_r2_control_first.log | v214 反向窗对照 [对照] |
| 117198 | Accepted | 74.25 | 43128 | submit_v214_r2_candidate.log | v214 反向窗候选；两窗互斥 |
| 117199 | Accepted | 74.00 | 44978 | submit_v215_gateup_bk256_s2.log | v215 gateup BK256/s2；慢 |
| 117200 | Accepted | 73.67 | 43704 | submit_v215_control.log | v215 对照 [对照] |
| 117201 | Accepted | 74.17 | 43110 | submit_v216_route_e64_w4.log | v216 route E64 w4 首窗 |
| 117202 | Accepted | 73.42 | 43755 | submit_v216_control.log | v216 对照 [对照] |
| 117203 | Accepted | 74.25 | 43284 | submit_v216_r2_control_first.log | v216 反向窗对照 [对照] |
| 117204 | Accepted | 74.50 | 43531 | submit_v216_r2_candidate.log | v216 反向窗候选；互斥 |
| 117205 | Accepted | 74.00 | 43673 | submit_v217_route_e64_bm64_w4.log | v217 E64 BM64/w4；慢 |
| 117206 | Accepted | 74.08 | 43222 | submit_v217_control.log | v217 对照 [对照] |
| 117207 | Accepted | 74.17 | 42980 | submit_v218_route_e32_bm64.log | v218 route E32 BM64；晋升中间态 **采用** |
| 117208 | Accepted | 73.58 | 43722 | submit_v218_control.log | v218 对照 [对照] |
| 117209 | Accepted | 73.58 | 43732 | submit_v218_r2_control_first.log | v218 反向窗对照 [对照] |
| 117211 | Accepted | 74.00 | 43235 | submit_v218_r2_candidate.log | v218 反向窗候选 **采用** |
| 117212 | Accepted | 74.58 | 42335 | submit_v219_case2_dual_normal.log | v219 case2 dual normal；case2 10.543 慢 |
| 117213 | Accepted | 75.08 | 43364 | submit_v219_control.log | v219 对照 [对照] |
| 117214 | Accepted | 74.25 | 43363 | submit_v220_route_e32_bm32.log | v220 E32 BM32；慢 |
| 117215 | Accepted | 73.92 | 43225 | submit_v220_control.log | v220 对照 [对照] |
| 117216 | Accepted | 75.33 | 43429 | submit_v221_route_e64_bm64_w8.log | v221 E64 BM64/w8；慢 |
| 117217 | Accepted | 74.17 | 43213 | submit_v221_control.log | v221 对照 [对照] |
| 117218 | Accepted | 74.17 | 42903 | submit_v223_case2_gather_bm64.log | v223 case2 gather BM64；晋升中间态 **采用** |
| 117219 | Accepted | 73.50 | 43732 | submit_v223_control.log | v223 对照 [对照] |
| 117220 | Accepted | 73.83 | 43754 | submit_v223_r2_control_first.log | v223 反向窗对照 [对照] |
| 117221 | Accepted | 73.83 | 43206 | submit_v223_r2_candidate.log | v223 反向窗候选 **采用** |
| 117222 | Accepted | 74.08 | 43340 | submit_v224_case2_gather_bm32.log | v224 case2 gather BM32；case2 变慢 |
| 117223 | Accepted | 74.08 | 43207 | submit_v224_control.log | v224 对照 [对照] |
| 117224 | Accepted | 74.25 | 42813 | submit_v225_gather_row_amax_bm64.log | v225 row gather BM64 首窗 |
| 117225 | Accepted | 74.25 | 43350 | submit_v225_control.log | v225 对照 [对照] |
| 117226 | Accepted | 74.08 | 43131 | submit_v225_r2_control_first.log | v225 反向窗对照 [对照] |
| 117227 | Accepted | 74.08 | 43362 | submit_v225_r2_candidate.log | v225 反向窗候选；不采用 |
| 117228 | Accepted | 74.00 | 43192 | submit_v226_gather64_quant64.log | v226 gather64+quant BM64 首窗 |
| 117229 | Accepted | 73.67 | 43691 | submit_v226_control.log | v226 对照 [对照] |
| 117230 | Accepted | 74.00 | 43172 | submit_v226_r2_control_first.log | v226 反向窗对照 [对照] |
| 117231 | Accepted | 73.58 | 43750 | submit_v226_r2_candidate.log | v226 反向窗候选；不采用 |
| 117234 | Accepted | 74.17 | 42961 | submit_v227_case2_gather_bm64_bh256.log | v227 case2 gather BM64/BH256；晋升中间态 **采用** |
| 117235 | Accepted | 73.50 | 43726 | submit_v227_control.log | v227 对照 [对照] |
| 117236 | Accepted | 73.33 | 43741 | submit_v227_r2_control_first.log | v227 反向窗对照 [对照] |
| 117237 | Accepted | 74.42 | 43233 | submit_v227_r2_candidate.log | v227 反向窗候选 **采用** |
| 117239 | Accepted | 74.00 | 43390 | submit_v228_case2_gather_bm64_bh512.log | v228 case2 gather BH512；case2 变慢 |
| 117240 | Accepted | 74.00 | 43228 | submit_v228_control.log | v228 对照 [对照] |
| 117241 | Accepted | 74.17 | 42949 | submit_v229_gather_row_amax_bm64_bh256.log | v229 row gather BM64/BH256 首窗 |
| 117242 | Accepted | 73.83 | 43372 | submit_v229_control.log | v229 对照 [对照] |
| 117243 | Accepted | 74.00 | 43245 | submit_v229_r2_control_first.log | v229 反向窗对照 [对照] |
| 117244 | Accepted | 74.00 | 43457 | submit_v229_r2_candidate.log | v229 反向窗候选；不采用 |
| 117287 | Accepted | 74.17 | 43471 | submit_v230_route_e32_bn16.log | v230 route E32 BN16；慢 |
| 117288 | Accepted | 74.33 | 43272 | submit_v230_control.log | v230 对照 |
| 117289 | Accepted | 74.33 | 42864 | submit_v231_final_h1024_w16.log | v231 final H1024 w16 首窗 |
| 117290 | Accepted | 73.83 | 43783 | submit_v231_control.log | v231 对照 [对照] |
| 117292 | WrongAnswer | 6.08 | 72588446 | submit_v232_gather_h1024_bh256.log | v232 H1024 gather BH256；缺 amax 定义 bug WA |
| 117293 | Accepted | 73.92 | 43286 | submit_v232_control.log | v232 对照 [对照] |
| 117296 | Accepted | 74.67 | 42323 | submit_v232b_gather_h1024_bh256.log | v232b 修复；首窗大赚 |
| 117298 | Accepted | 73.42 | 43699 | submit_v232b_r2_control_first.log | v232b 反向窗对照 [对照] |
| 117299 | Accepted | 74.17 | 43236 | submit_v232b_r2_candidate.log | v232b 反向窗候选；互斥，不采用 |
| 117300 | Accepted | 73.92 | 43266 | submit_v233_token_quant_bk64.log | v233 token quant BK64 第 3 窗；晋升当前 base **采用** |
| 117301 | Accepted | 73.75 | 43701 | submit_v233_control.log | v233 对照 [对照] |
| 117303 | Accepted | 73.75 | 43721 | submit_v233_r4_control_first.log | v233 第 4 窗对照 [对照] |
| 117304 | Accepted | 73.75 | 43325 | submit_v233_r4_candidate.log | v233 第 4 窗候选；3/4 窗 case2 更快 **采用** |
| 117306 | Accepted | 73.50 | 43781 | submit_v234_gather_row_bm64_bh256.log | v234 row gather BM64/BH256 第 3 窗；慢，关闭 |
| 117307 | Accepted | 74.00 | 43202 | submit_v234_control.log | v234 对照 [对照] |
| 117309 | Accepted | 73.92 | 43384 | submit_v235_token_quant_bm64_bk64.log | v235 token quant BM64/BK64；case2 变慢 |
| 117310 | Accepted | 74.25 | 43208 | submit_v235_control.log | v235 对照 [对照] |
