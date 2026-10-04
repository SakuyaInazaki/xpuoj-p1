# 2026-09-20 f16-acc BM256 md（目标：c3–c8 `_fgs_t1i_mdq_kernel_g`）

生产 `p1/kernel.py` = v837，SHA-256 `81d994bea8e0a22e8cd95d7b5dfe7a519a95968d822be54bcddaa7d0423f7f91`，**本轮未改动**。
本轮 **0 次提交**（Step 0 证据挖掘即触发关闭判据，按任务书 "do not build"）。

## 判定：CLOSED-BY-EVIDENCE（全几何）

f16 累加器 + 大 tile 这条线在本 fork 上已有**三层独立实测**，全部远坏于 +3% 阈值，
其中最关键的一条就是**逐字对应的 BM256 / BN128 / s3 / 8 warps 配置**。

---

## 证据一（决定性）：09-02 沙箱微基准 D52 — 真机 H800、原生 fp8、无 metadata/padding 混淆

来源：`archive/handoffs/handoff-session-20260821/session-notes.md` 圈68–69（bench19a v5/v6）。
单位 = effective TFLOPS，越高越快；括号为对 K2（现役 FP8 f32-acc BM128/BN128）之比。

| 档 | 配置 | TF | 比 K2 |
|---|---|---|---|
| K2 | **FP8 f32-acc BM128/BN128（基线）** | 1392.8 / 1404.2 | 1.00 |
| K3 | f16acc BM128/BN128 | 1119 | **0.80**（纯 lowering 税 −20%） |
| **K5** | **f16acc BM256/BN128 s3** | **1276** | **0.91（即 +9.9% 慢）** |
| K9 | f16acc BM256/BK64 s4 | 1105 | 0.79 |
| K10 | f16acc BM256 s2 | 852 | 0.61 |
| K16 | f32 BM128 s2 | 918.6 | 0.65（stages 比占用率重要） |
| K14 | f32acc BM256/BN128 nw16 | 138.8 | 0.10（寄存器溢出） |
| K17 | f32 BM128/BN256 nw16 | 74.7 | 0.05 |
| K6 | BM128/BN256 | — | smem 245760 > 232448，编不出来 |

**D52 原文封线**：「fp32 累加 BM256 无论 8/16 warps 都溢出，fp16 累加自带 −20%，
最好的 BM256 变体仍比 BM128 基线慢 9%，2 CTA/SM 变体全部大幅变慢。
**翻案条件：Triton 版本更新后 f16 累加 wgmma 原生化（K3 ≥ K2）**。」

⇒ 大 tile 的 L2 收益实测只有 **+14%**（1276/1119），f16 lowering 税 **−20%**，净 −9%。
⇒ **翻案条件未满足**：09-19 版本复探 SID 146133 确认判题机仍为 Triton 3.4 系、
`gluon_wgmma=no`、`warp_specialize=no`（用户确认官方口径 3.4）。

## 证据二：09-16 平台正式提交 — f16-acc BN512 在 L2-bound 案上的机器归一化残差

两条 md 内核族各一次干净读数（Accepted、12/12、SQNR 与锚**逐位相同**）：

| 候选 | SID | 内核 / 改动 | 锚 | 机器项 | 残差 |
|---|---|---|---|---|---|
| `f16acc_bn256_w4.py` (`f5aa462a`) | **145177** | c1/c2 `_fgs_tma2_int_pm_q8_kernel`：acc_g/acc_u → f16 + `out_dtype=tl.float16`；BN 128→256（BN_eff 256→**512**）；s4→s2（s3 装不下：80 KB/stage）；maxnreg 232；权重量化 target 448→4.0（E==8） | 145178 (v835) | +1.15% | **c1 +23.01% / c2 +25.89%** |
| 同上 | 145177 | 同上 | 145179 (`g_s4_only`) | −0.68% | **c1 +26.65% / c2 +30.19%** |
| `f16_bn512_c1112.py` (`31806ba8`) | **145181** | c11/c12 `_fgs_t1i_mdq_tma_kernel`：acc → f16（`F16` constexpr 门控）；BN 128→256（acc 是 `[BM,2*BN]` ⇒ BN_eff 256→**512**）；s4→s2；maxnreg 232→255；权重 target 4.0（G==32 且 I==1024） | 145178 (v835) | +1.27% | **c11 +42.76% / c12 +50.09%** |

- SQNR：145177 全 12 案与锚逐位相同（c1 23.12 / c2 23.14）；145181 c11 23.29 vs 锚 23.31、c12 23.26 vs 23.25。
  ⇒ **f16 累加的数值不是问题，速度是问题。**
- 无 OOR、无编译失败；acc 字节数与基线相同（2×128×256 f16 = 2×128×128 f32 = 128 KB ⇒ 8 warps 下仍 128 reg/thread），
  因此不是新增 spill 导致。

## 证据三：09-12 E256-c10 MD-only custom（已在 STATE「已关闭」表内）

`E256 L2-FP16acc BM256`：N64 epilogue split 把 255 regs/64 spills 降到 **217 regs / 0 spill**、
比旧 half 快 9.5%，**三组仍比 FP8 慢 14.1–20.8%**。这是**零 spill 的 BM256 f16-acc 直接读数**。

---

## 从未取得读数的 BM256 尝试（09-16，全部死在数学/编译，不构成正面证据）

`_fgs_tma2_int_pm_q8_kernel`（c1/c2），BM 128→256、BN=128、BK=128、8 warps、maxnreg=232，
TMA A-box 改 [256,128]：

| SID | 文件 | 配置 | 终态 |
|---|---|---|---|
| 145104 | `c1c2_f16acc_bm256.py` | BM256 **s4** | WA：`OutOfResources: shared memory, Required 262176 > 232448`（64 KB/stage × 4） |
| 145106 | `c1c2_f16acc_bm256_s3.py` | BM256 **s3**（量化 target 仍 448） | WA：oracle SQNR 23.79 dB **OK**、determinism OK，但 `correctness: target output has non-finite values` ⇒ **f16 累加器溢出**（fp8 amax=448 ⇒ 单积最大 448²=2.0e5 > 65504，K=2048 随机游走必 inf） |
| 145108 / 145110 | `..._s3_a8` / `..._s3_a8_bt8` | BM256 s3 + 激活/权重 target→8 | WA：SQNR **1.08 dB**（rescale 实现有 bug） |
| 145116 / 145120 / 145123 / 145126 / 145128 / 145176 | bn256 / bn128 + target 8/2/1 组 | — | TLE 或 SQNR 1.08 / 0.00 dB。其中 145126 是 **BN=128 基线 tile** + target8，同样 1.08 dB ⇒ 确证 bug 在 rescale 本身，与 tile 无关 |

**溢出的正确解法已知**：权重量化 target 448→4.0（145177 / 145181 用的就是这个，SQNR 逐位不变）。
所以若将来翻案，BM256/s3 是可建的；**卡住它的不是数学，是 −20% 的 f16 lowering 税**。

---

## 对任务书假设的逐条核对

1. **「fork 如何累加」**：`tl.dot(a, b.T, acc, out_dtype=tl.float16)` —— 本 fork 支持该签名，
   语义是**整条 K 链单一 f16 累加**，没有 per-block fp32 汇总。
   ⇒ 09-12 CPU 筛选的「K32 分块 f16 + 跨块 f32，最低 23.0069 dB」**不适用于这里**；
   真机行为见 145106（target 448 直接 non-finite）。
2. **「smem 够用」**：对 `_fgs_t1i_mdq_kernel_g` 的 BM256 估算正确 —— A[256,128] fp8 32 KB +
   B box[2*BN=256,128] fp8 32 KB = 64 KB/stage，s3 = 192 KB ≤ 227 KB。s4 不行（145104 实测 262176）。
3. **「f16 lowering 税只有 +2.5%」**：该数字来自 c1 在**旧 tile**上的读数；
   D52 的隔离微基准给出的是 **−20%**（K3/K2 = 0.80），与 145177/145181 的 +23~50% 一致。
4. **「c10 不能决定 L2-bound 案」**：同意；但 c1/c2 是 L2-bound 案，145177 的 +23~26% 正是在它们身上测的。
5. **算术强度账**：BM256 把每单位输出的 A+B 流量降 33.3%，BN512 降 16.7%。
   BN512 实测 +23~26%（c1/c2）。即便把 33.3% 的流量收益全额兑现，也补不上 −20% 的 lowering 税
   × 现役 measured/L2-floor 仅 1.0–1.13×（L2 墙上只有 0–13% 可回收）。

## 翻案条件（唯一）

判题机 Triton 升级到 f16 累加 wgmma 原生化（微基准 K3 ≥ K2）。
在此之前不要再建 f16 累加器 / BM256 / BN256 / BN512 的 md 变体。
