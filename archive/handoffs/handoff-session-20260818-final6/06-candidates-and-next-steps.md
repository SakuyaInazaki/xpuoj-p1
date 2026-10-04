# 06 候选文件、当前差距、下一步

## 推荐 base

- **`p1/kernel.py` / `p1/kernel_117300_backup.py` = 117300 = v233：当前实际性能 base。**
- scoreboard 文件：`p1/kernel_v165_final_tiled_aggr2.py`（116961，raw 靠异常，不推荐性能 base）。
- 上一稳定：`p1/kernel_117234_backup.py`、`p1/kernel_117218_backup.py`。

## 当前 raw 下一档差距（117300/117304 两窗均值）

| case | tk 均值 | 典型 tb | 当前分 | 下一档需 tk <= | 缺口 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.864 | 18.05 | 75 | 5.700 | 0.164 |
| 2 | 10.229 | 28.84 | 73 | 10.134 | **0.095** |
| 3 | 2.301 | 7.05 | 75 | 2.226 | 0.076 |
| 4 | 1.598 | 5.03 | 75 | 1.587 | **0.010** |
| 5 | 4.159 | 12.67 | 75 | 4.002 | 0.157 |
| 6 | 2.410 | 7.57 | 75 | 2.392 | **0.018** |
| 7 | 3.418 | 10.25 | 74 | 3.418 | **0.000**（边界） |
| 8 | 2.320 | 7.19 | 75 | 2.272 | 0.048 |
| 9 | 3.365 | 8.45 | 71 | 3.287 | 0.078 |
| 10 | 2.710 | 6.70 | 71 | 2.606 | 0.104 |
| 11 | 1.989 | 5.67 | 74 | 1.888 | 0.100 |
| 12 | 2.933 | 8.14 | 73 | 2.861 | 0.072 |

case2 仍是绝对最大单点，但离下一档只差约 0.095ms；case4/6/7 也都在边界附近。

## 本会话关键候选文件

| 文件 | 最终状态 | 结果摘要 |
|---|---|---|
| `p1/kernel_v172_final_bt32_w32.py` | 已融入 base | final tiling BT32/w32 |
| `p1/kernel_v184_route_n96_bm64_bk128.py` | 已融入 base | route N96 BM64/BK128 |
| `p1/kernel_v192_swiglu_bn128.py` | 已被 BN64 替代 | SwiGLU BN256->128 有效 |
| `p1/kernel_v199_swiglu_bn64.py` | 已融入 base | SwiGLU BN64 |
| `p1/kernel_v205b_route_n96_w4.py` | 已融入 base | route N96 w4 |
| `p1/kernel_v188b_fused_gm32.py` | 已融入 base | fused GM32/s4 |
| `p1/kernel_v218_route_e32_bm64.py` | 已融入 base | route E32 BM64 |
| `p1/kernel_v223_case2_gather_bm64.py` | 已融入 base | case2 gather BM64 |
| `p1/kernel_v227_case2_gather_bm64_bh256.py` | 已融入 base | case2 gather BM64/BH256 |
| `p1/kernel_v233_token_quant_bk64.py` | **当前 base** | case2 token quant BK64 |

## 明确不要重试

- case2 dual gateup：`v179/v180/v181/v219`（慢/OOR）。
- case2 直接 FP8 gate/up：`v176/v177`（v177 最终 TLE）。
- case2 SwiGLU BN32：`v206`（case2 变慢）。
- case2 gather BM32/BH512：`v224/v228`。
- non-case2 gather BM64/BH256：`v229/v234`（3 窗 2 慢）。
- token quant BM64/BK64：`v235`；token quant BK32：`v211`。
- down w4/w16：`v193c/v194b`。
- fused grid264/GM64/GM32+s3：`v213/v207/v210`。
- route E8 s4/BM32、E64 BM64/w4/w8、E32 BM32/BN16：`v191b/v208/v216/v217/v221/v220/v230`。
- final H1024 BT64/w16、H3584 BH512、case9/10 BT16：`v186/v231/v203b/v214`。
- gateup grid264/BK256/s2/w16：`v174/v215/v200b`。

## 下一步优先级

1. **case2 plain gateup GEMM 结构性优化（最大单点）**。
   - 当前 10.18-10.28ms。BM/BN/BK/GM/num_warps/stages/grid 已扫尽。
   - dual/multi-dot、interleave、amax epilogue、FP8 C 输出均已失败。
   - 建议下一步：真实 profiling（NCU/SASS），看 SM 占用、register spill、L2 命中；
     或研究每 CTA 多 N-tile 流水（需同时控制 register 与 num_stages，v180 OOR 的教训）。
2. **边界分数微调**。
   - case4/6/7 离下一档只有 0.00-0.02ms；任何全 case 微优化都可能直接加分。
   - 可谨慎复测（不是重投同一文件）：
     - `p1/kernel_v214_final_case910_bt16.py`（case9/10 final BT16，两窗互斥，值得第 3 窗）；
     - `p1/kernel_v216_route_e64_w4.py`（E64 route w4，case6 两窗均为 -0.015~-0.032）；
     - `p1/kernel_v232b_gather_h1024_bh256.py`（H1024 gather BH256，首窗大赚、反向小亏，值得第 3 窗）。
   - 必须同窗口投当前 `p1/kernel.py` 对照，至少 2 对。
3. **不要做**：
   - 重复提交当前 base 碰 tb；
   - 继续盲扫已关闭参数；
   - 在 Pending 期间反复重投同一候选。

## 已备好但未投/未采纳候选

- `p1/kernel_v193b_down_w4_only.py`、`p1/kernel_v194b_down_w16_only.py`（已证慢，勿投）。
- `p1/kernel_v182_case2_dual_interleave_amax_bn256w16.py`、
  `p1/kernel_v183_case2_dual_interleave_amax_bf16.py`（dual 线已关，勿投）。
- `p1/kernel_v189_final_h3584_bh512.py`（含 H1024 BT64，已证无收益，勿投）。
- `p1/kernel_v195_base_h1024bt64.py`、`p1/kernel_v196_base_swiglu_bn128.py`（中间态，已被替代）。

## 诊断方法

- phase 诊断模板：`p1/kernel_diag_v199_phases.py`（117182，WA 但 userError 中有各 case phase）。
- 生成新诊断：复制当前 base，插入 `_diag_mark` / `_diag_raise`，注意只在第二次调用 raise。
- 错误优先查 `progress.testcaseResult[*].userError`。
