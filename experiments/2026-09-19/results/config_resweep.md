# 2026-09-19 v836 新路径启动参数重扫（c9/c10 tiled md+dn，c8/c4 `_g`）

基线 `p1/kernel.py` = v836，SHA-256 `d849cd97cc8a44e79d1cf42c76beff276410032c2af7f64c3f80586de5974de7`（未修改）。
每对 = 候选 + 紧邻同窗锚（锚 = 未改动的 `p1/kernel.py`）。全部候选只改 launch kwarg /
host 侧整数，不动任何 kernel 源码。

## Step 1 现值

`_fgs_tma1_host_tiled` → `_fgs_tma1_kernel_gq_tiled[(132,)]`（唯一调用点 = `use_case9_lowmem_fp8`
= `E==256 and I in (2048,1536)`，即 c9/c10 专用）：
`BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=32, KTOP=_GA[1], EPP=_EPP[0](=2), KT=K//128,
num_warps=8, num_stages=4, maxnreg=232`；kernel 内外层 `tl.range(..., num_stages=2)`。

`_dn_tma2_f8_host_tiled` → `_dn_tma2_f8_tiled_kernel[(132,)]`（唯一调用点同为 c9/c10）：
`BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_M = 4 if K==8192 else (8 if N==1024 else 32)`
⇒ E=256 几何（K=I=2048/1536, N=H=4096）取 **GROUP_M=32**；`FLAT=(K<=2560)` ⇒ **True**；
`num_stages = 3 if K==14336 else 4` ⇒ **4**；`num_warps=8`。
注意：任务书写的是 `_dn_tma2_f8_host`，但 v836 里 c9/c10 的 dn 走的是 **`_dn_tma2_f8_host_tiled`**
（line 6335，`use_case9_lowmem_fp8` 分支）。R3 因此改 tiled 版。
`_dn_tma2_f8_host`（非 tiled）承接 c8（K=1024,N=4096,E=96）与 c4（K=1024,N=2048,E=32）：
两者 `GROUP_M=32`（K≠8192 且 N≠1024）、`FLAT=True`、`num_stages=4`。

`_fgs_tma1_intq_host` → `_fgs_t1i_mdq_kernel_g[(132,)]`（`_GA[0]` 分支，c3–c8 的 md）：
`BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M = 8 if K<=1024 else 32, KTOP=_GA[1],
num_warps=8, num_stages=3`，**未设 maxnreg**。host 内可用 `G`(=E) 与 `I` 做几何门控。

R1(+1) 跳过理由：该 kernel 每级流水缓冲 = a[128,128]fp8 16KB + bg[128,128]fp8 16KB +
bu[128,128]fp8 16KB = 48KB，`num_stages=5` ⇒ 240KB > H100 的 227KB 动态 smem 上限。

## 候选清单（全部 = v836 + 单一改动，且门控到精确几何）

| 编号 | 文件 | SHA-256（短） | 改动 | 触碰案 |
|---|---|---|---|---|
| R1 | `r1_md_s3.py` | `4bc5b18b` | tiled md `num_stages` 4→3 | 9,10 |
| R1+ | —（跳过） | — | `num_stages=5` 需 240KB smem > 227KB | — |
| R2a | `r2_md_gm16.py` | `07b29644` | tiled md `GROUP_M` 32→16 | 9,10 |
| R2b | `r2_md_gm8.py` | `e540ef51` | tiled md `GROUP_M` 32→8 | 9,10 |
| R3a | `r3_dn_gm8.py` | `235668eb` | tiled dn `GROUP_M` 32→8（门控 K∈(2048,1536) & N==4096） | 9,10 |
| R3b | `r3_dn_gm16.py` | `ed9e6764` | tiled dn `GROUP_M` 32→16（同上门控） | 9,10 |
| R4a | `r4_md_reg200.py` | `ae7ba71c` | tiled md `maxnreg` 232→200 | 9,10 |
| R4b | `r4_md_reg255.py` | `10930245` | tiled md `maxnreg` 232→255 | 9,10 |
| R5 | `r5_c8_g_s4reg232.py` | `81d994be` | `_g` s3→4 + maxnreg=232，门控 `G==96 and I==1024` | 8 |
| R6a | `r6_c4_g_s4.py` | `79f09562` | `_g` s3→4，门控 `G==32 and I==1024` | 4 |
| R6b | `r6_c4_g_s4reg232.py` | `72afb435` | `_g` s3→4 + maxnreg=232，门控 `G==32 and I==1024` | 4 |
| R7 | `r7_c910_mdgm8_dngm8.py` | `127f5705` | R2b + R3a 合并：tiled md GROUP_M 32→8 且 tiled dn GROUP_M 32→8 | 9,10 |

R1/R2/R4 改的 `_fgs_tma1_host_tiled` 与 R3 改的 `_dn_tma2_f8_host_tiled` 都只有一个调用点
（`use_case9_lowmem_fp8`），因此天然只影响 c9/c10；R3 仍加了显式几何门控。
R5/R6 用 host 局部变量 `_gg` 门控，非目标几何的 `num_stages` 仍为 3 且不传 `maxnreg`
（`**({} )` 展开为空）⇒ 其余案启动参数逐字不变。

## SID 对照（每发后立即追加）

| 对 | 候选 | 候选 SID | 锚 SID | 终态 |
|---|---|---|---|---|
| pair1 | R1 md_s3 | 146300 | 146301 | Accepted / Accepted |
| pair1 | R2a md_gm16 | 146302 | 146303 | Accepted / Accepted |
| pair1 | R2b md_gm8 | 146304 | 146305 | Accepted / Accepted |
| pair1 | R3a dn_gm8 | 146306 | 146307 | Accepted / Accepted |
| pair1 | R3b dn_gm16 | 146308 | 146309 | Accepted / Accepted |
| pair1 | R4a md_reg200 | 146358 | 146359 | Accepted / Accepted |
| pair1 | R4b md_reg255 | 146360 | 146361 | Accepted / Accepted |
| pair1 | R5 c8_g_s4reg232 | 146362 | 146363 | Accepted / Accepted |
| pair1 | R6a c4_g_s4 | 146364 | 146365 | Accepted / Accepted |
| pair1 | R6b c4_g_s4reg232 | 146367 | 146369 | Accepted / Accepted |
| pair1 | R7 md_gm8+dn_gm8 | 146370 | 146371 | Accepted / Accepted |
| pair2 | R2b md_gm8 | 146372 | 146373 | Accepted / Accepted |
| pair2 | R3a dn_gm8 | 146375 | 146376 | Accepted / Accepted |
| pair2 | R5 c8_g_s4reg232 | 146379 | 146380 | Accepted / Accepted |
| pair2 | R4a md_reg200 | 146384 | 146386 | Accepted / Accepted |
| pair2 | R4b md_reg255 | 146388 | 146389 | Accepted / Accepted |
| pair2 | R2a md_gm16 | 146393 | 146394 | Accepted / Accepted |
| pair2 | R7 md_gm8+dn_gm8（候选首发 HTTP 500 未产生 SID，锚 146395 先落地后补发候选） | 146396 | 146395 | Accepted / Accepted |
| pair3 | R5 c8_g_s4reg232（加固） | 146398 | 146399 | Accepted / Accepted |
| pair3 | R4a md_reg200（破平） | 146402 | 146403 | Accepted / Accepted |

## 配对结果（`scripts/mnorm.py` 口径：未触碰案最小二乘拟合机器项 m，残差 = 真实效应，+ = 变慢）

| 候选 | 对 | 候选/锚 SID | m | 触碰案残差 | 触碰案 Δtk(ms) | SQNR |
|---|---|---|---|---|---|---|
| R1 md_s3 | p1 | 146300 / 146301 | +1.11% | c9 +1.65%, c10 +2.63% | +0.046, +0.064 | 全同 |
| R2a md_gm16 | p1 | 146302 / 146303 | −1.32% | c9 +0.05%, c10 −0.81% | −0.004, −0.032 | 全同 |
| R2b md_gm8 | p1 | 146304 / 146305 | +1.13% | c9 +0.22%, c10 −1.86% | +0.010, −0.023 | 全同 |
| R3a dn_gm8 | p1 | 146306 / 146307 | −1.32% | c9 +0.41%, c10 −0.93% | +0.005, −0.034 | 全同 |
| R3b dn_gm16 | p1 | 146308 / 146309 | +0.92% | c9 +1.05%, c10 +1.88% | +0.030, +0.047 | 全同 |
| R4a md_reg200 | p1 | 146358 / 146359 | −0.92% | c9 −1.27%, c10 +0.10% | −0.036, −0.009 | 全同 |
| R4b md_reg255 | p1 | 146360 / 146361 | −1.20% | c9 −1.43%, c10 +0.32% | −0.041, −0.008 | 全同 |
| R5 c8_g_s4reg232 | p1 | 146362 / 146363 | −1.31% | c8 −1.97% | −0.030 | 全同 |
| R6a c4_g_s4 | p1 | 146364 / 146365 | +1.24% | c4 +0.22% | +0.012 | 全同 |
| R6b c4_g_s4reg232 | p1 | 146367 / 146369 | −1.20% | c4 +2.60% | +0.012 | 全同 |
| R7 md_gm8+dn_gm8 | p1 | 146370 / 146371 | −1.34% | c9 −1.62%, c10 −0.45% | −0.047, −0.025 | 全同 |
| R2b md_gm8 | p2 | 146372 / 146373 | +1.11% | c9 +1.36%, c10 +0.41% | +0.039, +0.021 | 全同 |
| R3a dn_gm8 | p2 | 146375 / 146376 | +1.20% | c9 −0.35%, c10 +1.28% | −0.004, +0.039 | 全同 |
| R5 c8_g_s4reg232 | p2 | 146379 / 146380 | −1.44% | c8 −1.81% | −0.028 | 全同 |
| R4a md_reg200 | p2 | 146384 / 146386 | −1.47% | c9 −0.01%, c10 +1.10% | −0.006, +0.004 | 全同 |
| R4b md_reg255 | p2 | 146388 / 146389 | −1.34% | c9 +0.77%, c10 −0.41% | +0.014, −0.024 | 全同 |
| R2a md_gm16 | p2 | 146393 / 146394 | −1.30% | c9 −1.59%, c10 +0.19% | −0.046, −0.012 | 全同 |
| R7 md_gm8+dn_gm8 | p2 | 146396 / 146395 | +1.21% | c9 +1.21%, c10 +1.35% | +0.035, +0.040 | 全同 |
| R5 c8_g_s4reg232 | p3 | 146398 / 146399 | −1.33% | c8 −1.96% | −0.030 | 全同 |
| R4a md_reg200 | p3 | 146402 / 146403 | −1.05% | c9 −0.70%, c10 +0.89% | −0.022, +0.005 | 全同 |

（R7 pair2 的候选首发 POST 返回 HTTP 500 `INTERNAL_SERVER_ERROR`，未产生 SID，也不是样例槽 TLE；
锚 146395 已先落地，候选随后补发为 146396，两发相邻同窗，仍按一对处理。
本轮 **没有任何样例槽 TimeLimitExceeded**：所有候选都只改 launch kwarg / host 侧整数，
与 09-19 早些时候「改 c9/c10 内核源码首发即冷编译 TLE」的事实一致。）

## 汇总（池化口径交叉校验）

20 发未改动 `p1/kernel.py` 锚提供的噪声地板（同码、跨机、全天）：
`c4 0.844±0.51%  c8 1.3625±0.51%  c9 2.5277±0.75%  c10 1.9562±0.85%`（tk 均值±总体标准差）。
把每个候选的同案 tk 池化后与该锚均值比：

| 候选 | 对数 | 池化 tk | vs 锚均值 | σ | 配对残差均值 | 判定 |
|---|---|---|---|---|---|---|
| R1 md_s3 | 1 | c9 2.5630 / c10 2.0000 | +1.40% / +2.24% | +1.9 / +2.6 | c9 +1.65%, c10 +2.63% | NEGATIVE |
| R2a md_gm16 | 2 | c9 2.5170 / c10 1.9575 | −0.42% / +0.07% | −0.8 / +0.1 | c9 −0.77%, c10 −0.31%（两对反号） | NEUTRAL |
| R2b md_gm8 | 2 | c9 2.5460 / c10 1.9445 | +0.72% / −0.60% | +1.4 / −1.0 | c9 +0.79%, c10 −0.73%（两对反号） | NEUTRAL |
| R3a dn_gm8 | 2 | c9 2.5105 / c10 1.9550 | −0.68% / −0.06% | −1.3 / −0.1 | c9 +0.03%, c10 +0.18%（两对反号） | NEUTRAL |
| R3b dn_gm16 | 1 | c9 2.5340 / c10 1.9760 | +0.25% / +1.01% | +0.3 / +1.2 | c9 +1.05%, c10 +1.88% | NEGATIVE |
| R4a md_reg200 | 3 | c9 2.5163 / c10 1.9657 | −0.45% / +0.48% | −1.0 / +1.0 | c9 −0.66%（三对同号），c10 +0.70% | NEUTRAL（配对口径够线但池化只有 −1.0σ，不足以动生产） |
| R4b md_reg255 | 2 | c9 2.5090 / c10 1.9355 | −0.74% / −1.06% | −1.4 / −1.8 | c9 −0.33%（反号），c10 −0.05% | NEUTRAL |
| **R5 c8_g_s4reg232** | **3** | **c8 1.3380** | **−1.80%** | **−6.2** | **c8 −1.91%（三对同号）** | **PROMOTE-CANDIDATE** |
| R6a c4_g_s4 | 1 | c4 0.8540 | +1.14% | +2.2 | c4 +0.22% | NEUTRAL |
| R6b c4_g_s4reg232 | 1 | c4 0.8500 | +0.66% | +1.3 | c4 +2.60% | NEGATIVE |
| R7 md_gm8+dn_gm8 | 2 | c9 2.5295 / c10 1.9510 | +0.07% / −0.27% | +0.1 / −0.4 | c9 −0.21%, c10 +0.45%（两对反号） | NEUTRAL |

全部候选 12 案 SQNR 与锚逐案同值（c4 23.19 / c8 23.12 / c9 23.13 / c10 23.11）。

### c10 的 78 分门槛（tk ≤ tb·22/78，用各发自己的 tb）

28 个 c10 读数（14 候选 + 14 锚）**无一跨线**。最接近的是锚 146395：tk 1.918、tb 6.789、
门槛 1.915，差 **0.003 ms**；候选侧最好的是 R4b pair2（146388）tk 1.928、tb 6.741、
门槛 1.901，差 **0.027 ms（1.40%）**。在 tb 抽签最有利的一发（tb 6.818，门槛 1.923）下，
候选 tk 1.943 仍差 0.020 ms（1.03%）。结论：c10 要稳定吃到 78 分，需要 **真实 −1.5% 以上**
且不依赖 tb 抽签的 tk 下降；本轮 11 个启动参数候选无一提供这个量级。

## 结论

- 唯一可晋升项：**R5**（`_fgs_t1i_mdq_kernel_g` 在 `G==96 and I==1024` 上 `num_stages` 3→4
  且 `maxnreg=232`），c8 稳定 −1.80%/−0.0245 ms，三对同号、−6.2σ、SQNR 不变。
  这正是 09-15 `v838_g_s4reg232` 被整体否决时「仅 c8 约 −1%」的那一项，按几何门控后 c5/c6 完全不受影响。
- c9/c10 的 tiled md/dn 启动参数（num_stages / GROUP_M / maxnreg，共 8 个候选、17 对）
  **全部落在噪声内**：现行 `num_stages=4, GROUP_M=32, maxnreg=232`（md）与 `GROUP_M=32,
  num_stages=4`（dn）在该机箱上已是局部最优，s3 与 dn GM16 明确更差。
- c4 的 `_g` s4（含/不含 maxnreg）为中性到负，`_g` 在 c4 上维持 s3。
- 生产 `p1/kernel.py` 未改动，SHA-256 仍为 `d849cd97…4de7`。
