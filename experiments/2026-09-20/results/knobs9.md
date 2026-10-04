# 2026-09-20 knobs9 — ten gated launch-parameter candidates on the v840 anchor

Anchor for every pair = the unchanged `p1/kernel.py` **v840**
(sha256 `ae961a07a7c1ca61385632673b2dc649f9635fe6e364322a990017ad341b4671` = v839 + c4 dn `num_stages` 3).

Geometry map. dn host `_dn_tma2_f8_host` sees K=I, N=H; md host `_fgs_tma1_intq_host` sees K=H, I=intermediate, G=E.
`_GASET` = c3..c10, so the `_g` md branch serves c3–c8 (c9/c10 never reach this host), the TMA md branch
(`K<=2048`, `_GA[0]==0`) serves c11/c12, and c1/c2 (E==8) use `_fgs_tma2_int_pm_q8_kernel`.
The tiled dn host is reached only by `use_case9_lowmem_fp8 = (E==256 and I in (2048,1536))` = c9/c10.

## Current values read out of v840 before changing anything
| launch | line | current value |
|---|---|---|
| `_dn_tma2_f8_host` GROUP_M | 4591 | `8 if (K==1024 and N==3584) else (4 if K==8192 else (8 if N==1024 else 32))` → **c1 = 4** |
| `_dn_tma2_f8_host` num_stages | 4593 | `3 if (K==1024 and N==3584) else (3 if (K==1024 and N==2048) else (3 if K==14336 else 4))` |
| `_dn_tma2_f8_host` maxnreg | — | **absent** (no maxnreg on this launch) |
| `_dn_tma2_f8_host_tiled` num_stages | 4672 | `3 if K==14336 else 4` → **c9/c10 = 4** |
| `_g` md (`_fgs_t1i_mdq_kernel_g`) | 5864-5865 | `num_stages = 4 if _gg else 3`, `maxnreg=232` only if `_gg` (`_gg = G==96 and I==1024` = c8) |
| TMA md (`_fgs_t1i_mdq_tma_kernel`) | 5876-5878 | `GROUP_M = 8 if K<=1024 else 32` → **c11/c12 = 8**; num_stages=4, maxnreg=232 |
| `_fgs_tma2_int_pm_q8_kernel` (c1/c2) | 6193-6195 | `GROUP_M=16, EPP=_EPP[0], num_warps=8, **num_stages=4**, maxnreg=232` |

g7 smem arithmetic: BLOCK_M=BLOCK_N=BLOCK_K=128. Per stage the pipeline holds one A tile
(128×128 fp8 = 16 KiB) and one B tile (128×128 int8 = 16 KiB) ⇒ 32 KiB/stage; 4 stages = 128 KiB,
which already fits the 227 KiB H800 limit, so the adjacent value that certainly fits is the **smaller**
one: 3 stages = 96 KiB. (5 stages = 160 KiB would also fit numerically but the brief asks for the
adjacent value when the current one is 4 ⇒ 3.)

## Candidates (each `diff p1/kernel.py <cand>` = exactly one changed/added line, verified)
| cand | change | gate (case) | sha12 |
|---|---|---|---|
| g1_c4_dn_reg232 | dn: add `maxnreg=232` | K==1024 and N==2048 (c4) | 79c9370e89c6 |
| g2_c1_dn_reg232 | dn: add `maxnreg=232` | K==8192 (c1) | 86946134d1e4 |
| g3_c1_dn_gm2 | dn GROUP_M 4→2 | K==8192 (c1) | d8cf5847e067 |
| g4_c1_dn_gm8 | dn GROUP_M 4→8 | K==8192 (c1) | 9d71335de455 |
| g5_c910_dn_tiled_s3 | tiled dn num_stages 4→3 | K<=2048 in the tiled host (c9+c10) | ed168fd1d8f5 |
| g6_c1112_md_gm4 | TMA md GROUP_M 8→4 | K<=1024 (c11+c12) | cb35d187c104 |
| g7_c12_q8md_stages | q8 md num_stages 4→3 | the E==8 q8 launch (c1+c2) | 8a4dce5822d1 |
| g8_c7_md_reg232 | `_g` md: add `maxnreg=232` (num_stages stays 3) | G==96 and I==2048 (c7) | ca1e2b8f98a7 |
| g9_c5_md_reg232 | `_g` md: add `maxnreg=232` (num_stages stays 3) | G==64 and I==2560 (c5) | 52cd276da3ba |
| g10_c3_md_reg232 | `_g` md: add `maxnreg=232` (num_stages stays 3) | G==32 and I==2048 (c3) | 8a4234ec51ff |

## Estimator (`scratchpad/knobs9/strat3.py`)
Machine index M excludes the touched case(s) **and c4 and c6** (c6 differs v837↔v838↔v839,
c4 differs v839↔v840). Anchor pool for the touched case's OLS = only anchors whose code on that case is
identical to v840: all v837+v838+v839 anchors for every case except c4 (v840 anchors only) and c6
(v839+v840). While the v840 anchor count for c4 is small the slope is borrowed from the full pool
(c4's M-sensitivity is version independent; M explains only ~12% of c4 variance) and only the intercept
comes from the v840 anchors; the noise scale (resid sd) also comes from the full pool.
Promotion rule: stratified z_of_mean ≤ −3, ≥75% of runs negative, SQNR equal on all 12 cases.

## SID log
- kpair1 g1_c4_dn_reg232 (touched c4): cand SID=146731, anchor SID=146733
- kpair2 g2_c1_dn_reg232 (touched c1): cand SID=146736, anchor SID=146737

### kpair1 — g1_c4_dn_reg232 (touched c4), cand 146731 / anchor 146733
raw c4 tk 0.8420 (anchor) → 0.8270 (cand). SQNR EQUAL on all 12; all 12 cases score 100.
Stratified (v840 anchor pool n=1, slope borrowed): M=0.9912, pred 0.8378, resid **−0.0108 ms (−1.286%), z=−2.09**.
pts 85/85; 86 would need tk ≤ 0.8195 — not crossed (best so far 0.8270).
⇒ strong first residual, queue more pairs.
- kpair3 g3_c1_dn_gm2 (touched c1): cand SID=146741, anchor SID=146742

### kpair2 — g2_c1_dn_reg232 (touched c1), cand 146736 / anchor 146737
raw c1 tk 4.6130 (anchor) → 4.7410 (cand). SQNR EQUAL on all 12; all 12 score 100.
c1 fit: anchors n=95, tk = −2.7050 + 7.3839·M, resid sd 0.0205 ms (0.439%).
Stratified: M=1.0062, pred 4.7244, resid **+0.0166 ms (+0.352%), z=+0.81**. pts 79/79, 80 needs ≤4.4727 — no.
⇒ first residual ≥ +0.3%; one more pair decides (2-strike rule).
- kpair4 g4_c1_dn_gm8 (touched c1): cand SID=146745, anchor SID=146746

### kpair3 — g3_c1_dn_gm2 (touched c1), cand 146741 / anchor 146742
raw c1 tk 4.7360 (anchor) → 4.6480 (cand) — but the candidate ran on a much faster machine (M=0.9901).
Stratified: pred 4.6067, resid **+0.0413 ms (+0.897%), z=+2.02**. SQNR EQUAL; all 12 score 100.
pts 79/79, 80 needs ≤4.5040 — no. ⇒ first residual ≥ +0.3% (strike 1).
- kpair5 g5_c910_dn_tiled_s3 (touched c9+c10): cand SID=146752, anchor SID=146754

### kpair4 — g4_c1_dn_gm8 (touched c1), cand 146745 / anchor 146746
raw c1 tk 4.7290 (anchor) → 4.6010 (cand). SQNR EQUAL; all 12 score 100.
Stratified: M=0.9927, pred 4.6262, resid **−0.0252 ms (−0.544%), z=−1.24**. pts 79/79, 80 needs ≤4.5070 — no.
⇒ lead (≤ −0.3%), queue more pairs.
- kpair6 g6_c1112_md_gm4 (touched c11+c12): cand SID=146757, anchor SID=146758

### kpair5 — g5_c910_dn_tiled_s3 (touched c9+c10), cand 146752 / anchor 146754
raw c9 2.5060→2.5010, c10 1.9470→1.9280. SQNR EQUAL; all 12 score 100. M=0.9909.
c9 fit: n=98, resid sd 0.0190 ms (0.753%) → resid **−0.0088 ms (−0.352%), z=−0.46**; pts 77/77, 78 needs ≤2.3794 — no.
c10 fit: n=98, resid sd 0.0138 ms (0.709%) → resid **−0.0116 ms (−0.600%), z=−0.84**; pts 77/77, 78 needs ≤1.9022 — no.
⇒ both touched cases are leads (≤ −0.3%), queue more pairs.
- kpair7 g7_c12_q8md_stages (touched c1+c2): cand SID=146763, anchor SID=146764

### kpair6 — g6_c1112_md_gm4 (touched c11+c12), cand 146757 / anchor 146758
raw c11 0.9660→0.9780, c12 1.5630→1.5900. SQNR EQUAL; all 12 score 100. M=1.0088.
c11 fit: n=99, resid sd 0.0020 ms (0.207%) → resid **+0.0024 ms (+0.242%), z=+1.18**; pts 85/85, 86 needs ≤0.9098 — no.
c12 fit: n=99, resid sd 0.0031 ms (0.199%) → resid **+0.0012 ms (+0.077%), z=+0.39**; pts 83/83, 84 needs ≤1.5522 — no.
⇒ not a lead (both ≥ 0). c11/c12 are the two quietest cases (resid sd ≈0.2%), so a real −0.3% would have shown.
No further pairs allocated.
- kpair8 g8_c7_md_reg232 (touched c7): cand SID=146768, anchor SID=146769

### kpair7 — g7_c12_q8md_stages (touched c1+c2), cand 146763 / anchor 146764
`_fgs_tma2_int_pm_q8_kernel` num_stages 4→3 (96 KiB of pipeline smem vs 128 KiB — both fit).
raw c1 4.7360→4.6470, c2 8.0620→7.9230, but the candidate ran on the faster machine (M=0.9928).
SQNR EQUAL; all 12 score 100.
c1: pred 4.6232, resid **+0.0238 ms (+0.515%), z=+1.07**; pts 79/79, 80 needs ≤4.5178 — no.
c2: pred 7.8626, resid **+0.0604 ms (+0.768%), z=+1.57**; pts 78/78, 79 needs ≤7.8646 — no.
⇒ both touched cases ≥ +0.3% (strike 1); fewer stages hurts the long-K q8 md.
- kpair9 g9_c5_md_reg232 (touched c5): cand SID=146776, anchor SID=146777

### kpair8 — g8_c7_md_reg232 (touched c7), cand 146768 / anchor 146769
raw c7 2.2800→2.3330 (M=1.0093, candidate on the slower machine). SQNR EQUAL; all 12 score 100.
c7 fit: n=101, resid sd 0.0055 ms (0.239%) → resid **+0.0034 ms (+0.146%), z=+0.62**.
pts 81/81, 82 needs ≤2.2291 — no. ⇒ not a lead; no further pairs.
- kpair10 g10_c3_md_reg232 (touched c3): cand SID=146780, anchor SID=146781

### kpair9 — g9_c5_md_reg232 (touched c5), cand 146776 / anchor 146777
raw c5 2.8860→2.8160. SQNR EQUAL; all 12 score 100. M=0.9927.
c5 fit: n=102, resid sd 0.0103 ms (0.362%) → resid **−0.0089 ms (−0.314%), z=−0.86**.
pts 81/81, 82 needs ≤2.7810 — no. ⇒ marginal lead (exactly at the −0.3% bar).
- kpair11 g1_c4_dn_reg232 (touched c4, 2nd pair): cand SID=146785, anchor SID=146786

### kpair10 — g10_c3_md_reg232 (touched c3), cand 146780 / anchor 146781
raw c3 1.4050→1.4230 (M=1.0087). SQNR EQUAL; all 12 score 100.
c3 fit: n=103, resid sd 0.0057 ms (0.408%) → resid **+0.0006 ms (+0.042%), z=+0.11**.
pts 82/82, 83 needs ≤1.4204 — no. ⇒ dead flat; no further pairs.

## Round 1 complete (one pair each, 20 submissions). Leads = first residual ≤ −0.3%
| cand | touched | resid % (z) | lead? |
|---|---|---|---|
| g1_c4_dn_reg232 | c4 | −1.286 (−2.09) | **yes (strongest)** |
| g2_c1_dn_reg232 | c1 | +0.352 (+0.81) | no (strike 1) |
| g3_c1_dn_gm2 | c1 | +0.897 (+2.02) | no (strike 1) |
| g4_c1_dn_gm8 | c1 | −0.544 (−1.24) | **yes** |
| g5_c910_dn_tiled_s3 | c9 / c10 | −0.352 (−0.46) / −0.600 (−0.84) | **yes** |
| g6_c1112_md_gm4 | c11 / c12 | +0.242 (+1.18) / +0.077 (+0.39) | no |
| g7_c12_q8md_stages | c1 / c2 | +0.515 (+1.07) / +0.768 (+1.57) | no (strike 1) |
| g8_c7_md_reg232 | c7 | +0.146 (+0.62) | no |
| g9_c5_md_reg232 | c5 | −0.314 (−0.86) | **yes (marginal)** |
| g10_c3_md_reg232 | c3 | +0.042 (+0.11) | no |

Round-2 allocation of the remaining ~15 pairs, ranked by |effect|/resid-sd
(g1 2.10, g4 1.25, g9 0.87, g5-c10 0.85, g5-c9 0.47): g1 +6, g4 +5, g5 +2, g9 +2.
g5 and g9 cannot reach z ≤ −3 inside this budget (they would need n≈13 and n≈12); their extra pairs
are sign-consistency checks only.
- kpair12 g4_c1_dn_gm8 (touched c1, 2nd pair): cand SID=146790, anchor SID=146791

### kpair11 — g1_c4_dn_reg232 (touched c4, pair 2), cand 146785 / anchor 146786
raw c4 0.8410→0.8210. SQNR EQUAL; all 12 score 100. M=0.9907.
c4 fit now uses its own v840 anchor pool (n=11): tk = 0.3602 + 0.4763·M, resid sd 0.0037 ms (0.437%).
resid **−0.0111 ms (−1.331%), z=−3.03**.
**Crossed the next integer line**: c4 pts 86 (cand) vs 85 (anchor); with its own tb=5.074 the 86 line is
tk ≤ 0.8260 and the candidate did 0.8210 — a crossing on merit, not a tb lottery
(the anchor's tb=4.874 would have needed ≤0.7934).
**g1 pooled n=2: mean −0.0082 ms (−0.976%), z_of_mean −3.16, 2/2 negative.**
- kpair13 g1_c4_dn_reg232 (touched c4, 3rd pair): cand SID=146795, anchor SID=146796

### kpair12 — g4_c1_dn_gm8 (touched c1, pair 2), cand 146790 / anchor 146791
raw c1 4.7470→4.6430 (M=0.9915). SQNR EQUAL; all 12 score 100.
resid **+0.0243 ms (+0.526%), z=+1.17**. pts 79/79, 80 needs ≤4.6215 — no (missed by 0.0215 ms).
**g4 pooled n=2: mean −0.0004 ms (−0.008%), z_of_mean −0.03, 1/2 negative** — pair 4 was noise.
Reallocating: g1 gets the bulk of what is left; g4 keeps one more pair as a tie-breaker.
- kpair14 g1_c4_dn_reg232 (touched c4, 4th pair): cand SID=146800, anchor SID=146801

### kpair13 — g1_c4_dn_reg232 (touched c4, pair 3), cand 146795 / anchor 146796
raw c4 0.8360→0.8440 (M=1.0090). SQNR EQUAL; all 12 score 100.
resid **+0.0043 ms (+0.517%), z=+1.11**. pts 85/85, 86 needs ≤0.8040 — no.
**g1 pooled n=3: mean −0.0046 ms (−0.545%), z_of_mean −2.02, 2/3 negative.**
- kpair15 g5_c910_dn_tiled_s3 (touched c9+c10, 2nd pair): cand SID=146805, anchor SID=146806

### kpair14 — g1_c4_dn_reg232 (touched c4, pair 4), cand 146800 / anchor 146801
raw c4 0.8430→0.8360 (M=0.9928). SQNR EQUAL; all 12 score 100.
resid **+0.0022 ms (+0.267%), z=+0.57**. pts 87(cand — tb lottery, tb≈5.6)/85(anchor); 88 needs ≤0.7797 — no.
**g1 pooled n=4: mean −0.0030 ms (−0.353%), z_of_mean −1.53, 2/4 negative.**
Remaining budget goes almost entirely to g1 (the only candidate that can still reach a verdict);
g4 gets no more pairs (pooled 0.00% at n=2), g5 finishes its second pair, g9 gets one more.
- kpair16 g1_c4_dn_reg232 (touched c4, 5th pair): cand SID=146811, anchor SID=146812

### kpair15 — g5_c910_dn_tiled_s3 (touched c9+c10, pair 2), cand 146805 / anchor 146806
raw c9 2.5220→2.5150, c10 1.9500→1.9480 (M=0.9923). SQNR EQUAL; all 12 score 100.
c9 resid **+0.0027 ms (+0.106%), z=+0.14**; c10 resid **+0.0065 ms (+0.337%), z=+0.49**. No crossings.
**g5 pooled n=2: c9 −0.132% (z_of_mean −0.25, 1/2 negative); c10 −0.135% (z_of_mean −0.28, 1/2 negative).**
⇒ the pair-1 signal did not reproduce; g5 is NEUTRAL, no further pairs.
- kpair17 g1_c4_dn_reg232 (touched c4, 6th pair): cand SID=146817, anchor SID=146819

### kpair16 — g1_c4_dn_reg232 (touched c4, pair 5), cand 146811 / anchor 146812
raw c4 0.8320→0.8370 (M=1.0067). SQNR EQUAL; all 12 score 100.
resid **−0.0022 ms (−0.264%), z=−0.60**. pts 85/85, 86 needs ≤0.8107 — no.
**g1 pooled n=5: mean −0.0026 ms (−0.312%), z_of_mean −1.59, 3/5 negative.**
- kpair18 g1_c4_dn_reg232 (touched c4, 7th pair): cand SID=146823, anchor SID=146824

### kpair17 — g1_c4_dn_reg232 (touched c4, pair 6), cand 146817 / anchor 146819
raw c4 0.8440→0.8310 (M=0.9940). SQNR EQUAL; all 12 score 100.
resid **−0.0031 ms (−0.367%), z=−0.83**. pts 85/85, 86 needs ≤0.8099 — no.
**g1 pooled n=6: mean −0.0028 ms (−0.334%), z_of_mean −1.86, 4/6 negative (67%).**
- kpair19 g1_c4_dn_reg232 (touched c4, 8th pair): cand SID=146828, anchor SID=146829

### kpair18 — g1_c4_dn_reg232 (touched c4, pair 7), cand 146823 / anchor 146824
raw c4 0.8400→0.8310 (M=0.9960). SQNR EQUAL; all 12 score 100.
resid **−0.0039 ms (−0.472%), z=−1.11**. pts 85/85, 86 needs ≤0.8224 — no (missed by 0.0086 ms).
**g1 pooled n=7: mean −0.0029 ms (−0.352%), z_of_mean −2.19, 5/7 negative (71%).**
- kpair20 g1_c4_dn_reg232 (touched c4, 9th pair): cand SID=146832, anchor SID=146833

### kpair19 — g1_c4_dn_reg232 (touched c4, pair 8), cand 146828 / anchor 146829
raw c4 0.8310→0.8400 (M=0.9932). SQNR EQUAL; all 12 score 100.
resid **+0.0063 ms (+0.759%), z=+1.57**. pts 85/85, 86 needs ≤0.8247 — no.
**g1 pooled n=8: mean −0.0016 ms (−0.188%), z_of_mean −1.10, 5/8 negative (63%).**
The two early −1.3/−1.4% runs are looking like outliers; the running mean has decayed
−0.98 (n=2) → −0.55 (n=3) → −0.35 (n=4) → −0.31 (n=5) → −0.33 (n=6) → −0.35 (n=7) → −0.19 (n=8).

### Remaining-budget plan (10 submissions = 5 pairs)
g1 cannot reach z ≤ −3 from here, so the last pairs buy the two things still worth buying:
a clean 2-strike close on the three candidates that were ≥ +0.3% on their first pair (g2, g3, g7),
and two more g1 pairs to settle its pooled estimate.
- kpair21 g2_c1_dn_reg232 (touched c1, 2nd pair): cand SID=146836, anchor SID=146837

### kpair20 — g1_c4_dn_reg232 (touched c4, pair 9), cand 146832 / anchor 146833
raw c4 0.8420→0.8460 (M=1.0102). SQNR EQUAL; all 12 score 100.
resid **+0.0058 ms (+0.695%), z=+1.33**. pts 85/85, 86 needs ≤0.8170 — no.
**g1 pooled n=9: mean −0.0016 ms (−0.187%), z_of_mean −1.07, 5/9 negative (56%).**
- kpair22 g3_c1_dn_gm2 (touched c1, 2nd pair): cand SID=146841, anchor SID=146842

### kpair21 — g2_c1_dn_reg232 (touched c1, pair 2), cand 146836 / anchor 146837
raw c1 4.7450→4.6640 (M=0.9942). SQNR EQUAL; all 12 score 100.
resid **+0.0247 ms (+0.533%), z=+1.19**. pts 79/79, 80 needs ≤4.5345 — no.
**g2 pooled n=2: mean +0.0218 ms (+0.465%), z_of_mean +1.49, 0/2 negative.**
⇒ both pairs ≥ +0.3% → 2-strike stop, verdict **NEGATIVE**.
- kpair23 g7_c12_q8md_stages (touched c1+c2, 2nd pair): cand SID=146847, anchor SID=146848

### kpair22 — g3_c1_dn_gm2 (touched c1, pair 2), cand 146841 / anchor 146842
raw c1 4.7690→4.6950 (M=0.9902). SQNR EQUAL; all 12 score 100.
resid **+0.0848 ms (+1.840%), z=+4.02**. pts 82(cand — tb lottery)/79(anchor); 83 needs ≤4.4673 — no.
**g3 pooled n=2: mean +0.0637 ms (+1.361%), z_of_mean +4.27, 0/2 negative.**
⇒ both pairs ≥ +0.3% → 2-strike stop, verdict **NEGATIVE** (clearly worse than GROUP_M 4).
- kpair24 g1_c4_dn_reg232 (touched c4, 10th pair): cand SID=146851, anchor SID=146852

### kpair23 — g7_c12_q8md_stages (touched c1+c2, pair 2), cand 146847 / anchor 146848
raw c1 4.6060→4.7540, c2 7.8500→8.1410 (M=1.0076). SQNR EQUAL; all 12 score 100.
c1 resid **+0.0119 ms (+0.251%), z=+0.52**; c2 resid **+0.0661 ms (+0.819%), z=+1.74**. No crossings.
**g7 pooled n=2: c1 +0.408% (z_of_mean +1.19, 0/2 negative); c2 +0.799% (z_of_mean +2.37, 0/2 negative).**
⇒ c2 has both pairs ≥ +0.3% → 2-strike stop, verdict **NEGATIVE** (num_stages 4 is right for the q8 md).
- kpair25 g1_c4_dn_reg232 (touched c4, 11th pair): cand SID=146856, anchor SID=146857

### kpair24 — g1_c4_dn_reg232 (touched c4, pair 10), cand 146851 / anchor 146852
raw c4 0.8340→0.8450 (M=1.0104). SQNR EQUAL; all 12 score 100.
resid **+0.0039 ms (+0.462%), z=+0.77**. pts 85/85, 86 needs ≤0.8055 — no.
**g1 pooled n=10: mean −0.0004 ms (−0.045%), z_of_mean −0.23, 5/10 negative (50%).**

### kpair25 — g1_c4_dn_reg232 (touched c4, pair 11), cand 146856 / anchor 146857
raw c4 0.8390→0.8340 (M=0.9942). SQNR EQUAL; all 12 score 100.
resid **+0.0000 ms (+0.001%), z=+0.00**. pts 85/85, 86 needs ≤0.8154 — no.

## g1 FINAL — 11 runs (all recomputed against the final v840 anchor pool, n=25)
c4 fit: tk = 0.4052 + 0.4313·M, resid sd **0.0050 ms (0.594%)**.

| run | M | tk | pred | resid | z | pts | reached 86 |
|---|---|---|---|---|---|---|---|
| 146731 | 0.9907 | 0.8270 | 0.8325 | −0.0055 (−0.656%) | −1.10 | 85 | no |
| 146785 | 0.9903 | 0.8210 | 0.8323 | −0.0113 (−1.356%) | −2.27 | **86** | **yes, on merit** |
| 146795 | 1.0086 | 0.8440 | 0.8402 | +0.0038 (+0.453%) | +0.77 | 85 | no |
| 146800 | 0.9925 | 0.8360 | 0.8332 | +0.0028 (+0.332%) | +0.56 | 87 | tb lottery |
| 146811 | 1.0064 | 0.8370 | 0.8392 | −0.0022 (−0.268%) | −0.45 | 85 | no |
| 146817 | 0.9939 | 0.8310 | 0.8338 | −0.0028 (−0.339%) | −0.57 | 85 | no |
| 146823 | 0.9959 | 0.8310 | 0.8347 | −0.0037 (−0.444%) | −0.75 | 85 | no |
| 146828 | 0.9932 | 0.8400 | 0.8335 | +0.0065 (+0.775%) | +1.30 | 85 | no |
| 146832 | 1.0100 | 0.8460 | 0.8408 | +0.0052 (+0.618%) | +1.05 | 85 | no |
| 146851 | 1.0104 | 0.8450 | 0.8409 | +0.0041 (+0.483%) | +0.82 | 85 | no |
| 146856 | 0.9942 | 0.8340 | 0.8340 | +0.0000 (+0.001%) | +0.00 | 85 | no |

**MEAN −0.0003 ms (−0.035%), z_of_mean −0.19, 5/11 negative (45%), SQNR equal on all 12 in all 11 runs.**
The running mean decayed monotonically once the two early outliers were diluted:
−0.98 (n=2) → −0.55 → −0.35 → −0.31 → −0.33 → −0.35 → −0.19 → −0.19 → −0.05 → −0.035 (n=11).
⇒ **NEUTRAL**. The first-pair −1.29% was a false lead; c4's dn does not care about `maxnreg=232`.

## SUMMARY (knobs9) — 25 pairs, 50 submissions, anchor = v840 in every pair
| cand | change + gate | sha12 | pair SIDs (cand/anchor) | stratified Δ% per run | z_of_mean (n) | % negative | crossed next line? | SQNR same | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **g1_c4_dn_reg232** | dn `maxnreg=232`, gate K==1024 and N==2048 (c4) | 79c9370e89c6 | 146731/733, 146785/786, 146795/796, 146800/801, 146811/812, 146817/819, 146823/824, 146828/829, 146832/833, 146851/852, 146856/857 | −0.656, −1.356, +0.453, +0.332, −0.268, −0.339, −0.444, +0.775, +0.618, +0.483, +0.001 | **−0.035 (−0.19, 11)** | 5/11 = 45% | 146785 reached c4 = 86 on merit (tb 5.074, needed ≤0.8260, got 0.8210); 146800 hit 87 on a tb outlier | yes (11/11) | **NEUTRAL** |
| g2_c1_dn_reg232 | dn `maxnreg=232`, gate K==8192 (c1) | 86946134d1e4 | 146736/737, 146836/837 | +0.352, +0.533 | **+0.452 (+1.35, 2)** | 0/2 | no | yes (2/2) | **NEGATIVE** (2-strike) |
| g3_c1_dn_gm2 | dn GROUP_M 4→2, gate K==8192 (c1) | d8cf5847e067 | 146741/742, 146841/842 | +0.897, +1.840 | **+1.340 (+4.01, 2)** | 0/2 | no (82 pts once, tb lottery) | yes (2/2) | **NEGATIVE** |
| g4_c1_dn_gm8 | dn GROUP_M 4→8, gate K==8192 (c1) | 9d71335de455 | 146745/746, 146790/791 | −0.542, +0.526 | **+0.001 (+0.00, 2)** | 1/2 | no | yes (2/2) | **NEUTRAL** |
| g5_c910_dn_tiled_s3 | tiled dn num_stages 4→3, gate K<=2048 (c9+c10) | ed168fd1d8f5 | 146752/754, 146805/806 | c9 −0.370/+0.106; c10 −0.608/+0.337 | c9 **−0.123 (−0.24, 2)**; c10 **−0.152 (−0.32, 2)** | 1/2 each | no | yes (2/2) | **NEUTRAL** |
| g6_c1112_md_gm4 | TMA md GROUP_M 8→4, gate K<=1024 (c11+c12) | cb35d187c104 | 146757/758 | c11 +0.255; c12 +0.054 | c11 +1.21 (n=1); c12 +0.27 (n=1) | 0/1 | no | yes (1/1) | **NEUTRAL** (c11/c12 sd ≈0.2%, so a real −0.3% would have shown) |
| g7_c12_q8md_stages | q8 md num_stages 4→3 (c1+c2) | 8a4dce5822d1 | 146763/764, 146847/848 | c1 +0.515/+0.251; c2 +0.768/+0.819 | c1 **+0.399 (+1.11, 2)**; c2 **+0.795 (+2.37, 2)** | 0/2 each | no | yes (2/2) | **NEGATIVE** (2-strike on c2) |
| g8_c7_md_reg232 | `_g` md `maxnreg=232` (s3 kept), gate G==96 and I==2048 (c7) | ca1e2b8f98a7 | 146768/769 | +0.172 | +0.73 (n=1) | 0/1 | no | yes (1/1) | **NEUTRAL** |
| g9_c5_md_reg232 | `_g` md `maxnreg=232` (s3 kept), gate G==64 and I==2560 (c5) | 52cd276da3ba | 146776/777 | −0.310 | −0.87 (n=1) | 1/1 | no | yes (1/1) | **NEUTRAL / unresolved** (only lead never re-tested; needs n≈12) |
| g10_c3_md_reg232 | `_g` md `maxnreg=232` (s3 kept), gate G==32 and I==2048 (c3) | 8a4234ec51ff | 146780/781 | +0.035 | +0.09 (n=1) | 0/1 | no | yes (1/1) | **NEUTRAL** |

**Nothing reaches the promotion bar (z_of_mean ≤ −3 with ≥75% negative). v840 stands unchanged.**

### What this round settles
- `maxnreg=232` on the **dn** launch is worthless at both geometries tried: c4 (n=11, −0.04%) and c1 (n=2, +0.45%).
  The earlier whole-file cap-232 readings (c4 −0.78%, c1 −0.80%) were machine artefacts, not the cap.
- `maxnreg=232` on the **`_g` md** launch is flat at c7 (+0.17) and c3 (+0.04); c5 (−0.31) is the only
  residual lead left anywhere in this round and it is one run deep.
- dn `GROUP_M` at c1 is already optimal at 4: 2 is much worse (+1.34%), 8 is a wash (0.00%).
- The tiled dn (c9/c10) does not want `num_stages` 3 — the pair-1 signal vanished on replication.
- The c11/c12 TMA md wants GROUP_M 8, not 4.
- The c1/c2 q8 md wants `num_stages` 4, not 3 (c2 +0.80%, z +2.37).
- SQNR was identical to the anchor on all 12 cases in all 25 pairs, and every case scored 100 everywhere.

### Cheapest follow-up if this line continues
g9 (`_g` md `maxnreg=232` gated to c5) is the only unresolved lead: one run at −0.31% against a 0.36%
resid sd. Settling it to z ≤ −3 needs ≈12 pairs (24 submissions).

### Provenance audit (run after the last pair)
Every one of the 25 candidate submissions hashes to its intended `g*_` file, and every one of the
25 anchor submissions hashes to **ae961a07a7c1** (v840), including the last one at 04:38:35Z.
NOTE: at ~04:42Z (after this round's last submission) another line edited `p1/kernel.py` line 883 to
`num_warps=2 if H == 1024 else 4` (new sha256 `0f07305bfe3f…`). Nothing in knobs9 was submitted from
that file; every knobs9 anchor predates the edit. Any future pair must re-baseline on the new file.
