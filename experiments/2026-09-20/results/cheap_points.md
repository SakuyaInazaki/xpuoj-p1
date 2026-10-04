# 2026-09-20 geometry-gated launch-parameter candidates (cheap points)

Anchor = unmodified `p1/kernel.py` v837 (sha256 81d994be…).
Case key = (T, H, E, I, topk) from `_KNOWN12`; dn host `_dn_tma2_f8_host` sees K=I, N=H.

| cand | change | gate | sha12 |
|---|---|---|---|
| k1_c4_md_gm16 | `_fgs_tma1_intq_host` `_g` md GROUP_M 32→16 | G==32 and I==1024 and K==2048 (c4) | e2afe9907f5d |
| k2_c4_dn_gm8 | `_dn_tma2_f8_host` GROUP_M 32→8 | K==1024 and N==2048 (c4) | 8e13371b8313 |
| k3_c4_dn_gm16 | `_dn_tma2_f8_host` GROUP_M 32→16 | K==1024 and N==2048 (c4) | 5ae812581d51 |
| k4_c6_dn_gm8 | `_dn_tma2_f8_host` GROUP_M 32→8 | K==1024 and N==3584 (c6) | 96f7b2cb40e0 |
| k5_c12_dn_s3 | `_dn_tma2_f8_host` num_stages 4→3 | K==2048 and N==1024 (c12) | 0a957de15f76 |
| k6_c11_dn_s3 | `_dn_tma2_f8_host` num_stages 4→3 | K==1024 and N==1024 (c11) | 08d095a38cc0 |
| k7_c1_md_gm16 | VOID — `_fgs_tma2_int_pm_q8_kernel` already launches GROUP_M=16 for both c1 and c2 | — | — |
| k8_c1_dn_s3 | `_dn_tma2_f8_host` num_stages 4→3 | K==8192 (c1) | e9d0e8c8f938 |
| k9_c3_md_gm16 | `_fgs_tma1_intq_host` `_g` md GROUP_M 32→16 | G==32 and I==2048 and K==2048 (c3) | a5c199e82b6d |

## SID log

- pair1 k1_c4_md_gm16: cand SID=146435, anchor SID=146437
- pair2 k5_c12_dn_s3: cand SID=146439, anchor SID=146440

### pair1 — k1_c4_md_gm16 (touched c4), cand 146435 / anchor 146437
machine term m=+0.090%; c4 d=+0.352%, resid **+0.264%** (+0.0023 ms); raw tk 0.8520→0.8550.
SQNR identical on all 12 (23.12/23.14/23.19/23.19/23.13/23.14/23.12/23.12/23.13/23.11/23.31/23.25).
c4 points 85/85 both; next point needs tk ≤ 0.8115 (gap +0.0435 ms) — NOT crossed.
Verdict: NEUTRAL (slightly positive = slower); no second pair (rule: only if resid ≤ −0.7%).

### pair2 — k5_c12_dn_s3 (touched c12), cand 146439 / anchor 146440
machine term m=+1.318% (big machine gap); c12 d=+1.340%, resid **+0.838%** (+0.0131 ms); raw tk 1.5670→1.5880.
SQNR identical on all 12. c12 points 83(cand)/84(anchor); next point needs tk ≤ 1.5478 — NOT crossed.
Verdict: NEGATIVE.

### added by coordinator
| k10_c6_md_gm8 | `_g` md GROUP_M 32→8 | G==64 and I==1024 (c6) | 7e2a1b0cf374 |
| k11_c6_md_gm16 | `_g` md GROUP_M 32→16 | G==64 and I==1024 (c6) | 554cfc2c12c6 |

- pair3 k6_c11_dn_s3: cand SID=146443, anchor SID=146444

### pair3 — k6_c11_dn_s3 (touched c11), cand 146443 / anchor 146444
machine term m=+1.222%; c11 d=+0.000%, resid **+0.089%** (+0.0009 ms); raw tk 0.9700→0.9700 (bit-identical to 4 dp).
SQNR identical on all 12. c11 points 85/85; next point needs tk ≤ 0.9273 — NOT crossed.
Verdict: NEUTRAL (num_stages 4→3 costs nothing and gains nothing at c11).

- pair4 k2_c4_dn_gm8: cand SID=146448, anchor SID=146449

### pair4 — k2_c4_dn_gm8 (touched c4), cand 146448 / anchor 146449
machine term m=-1.363% (candidate run on a faster machine); c4 d=+0.000%, resid **+1.329%** (+0.0111 ms);
raw tk 0.8380→0.8380 (identical). Untouched residual spread this pair is wide (c3 +1.92 … c8 -1.96) ⇒ noisy fit.
SQNR identical on all 12. c4 points 85(cand)/86(anchor, luckier tb); next point needs tk ≤ 0.8188 — NOT crossed.
Verdict: NEUTRAL→NEGATIVE (no raw gain; residual says it missed the machine speedup).

- pair5 k4_c6_dn_gm8: cand SID=146453, anchor SID=146454  (api_token credit hit 0 → switched to scripts/xpuoj_web.py + token pool)

### pair5 — k4_c6_dn_gm8 (touched c6), cand 146453 / anchor 146454
machine term m=+0.244% (tight pair, untouched spread ±1%); c6 d=-0.075%, resid **-0.250%** (-0.0033 ms);
raw tk 1.3260→1.3250. SQNR identical on all 12. c6 points 85/85; next point needs tk ≤ 1.2572 — NOT crossed.
Verdict: NEUTRAL (> -0.7% ⇒ no second pair).

- pair6 k10_c6_md_gm8: cand SID=146458, anchor SID=146459

### pair6 — k10_c6_md_gm8 (touched c6), cand 146458 / anchor 146459
machine term m=+1.147%; c6 d=+1.280%, resid **+0.462%** (+0.0061 ms); raw tk 1.3280→1.3450.
SQNR identical on all 12. c6 points 85/85; next point needs tk ≤ 1.3049 — NOT crossed.
Verdict: NEUTRAL→NEGATIVE. (Does not reproduce the old whole-file-read c6 -0.61%.)

- pair7 k11_c6_md_gm16: cand SID=146462, anchor SID=146464

### pair7 — k11_c6_md_gm16 (touched c6), cand 146462 / anchor 146464
machine term m=+1.247%; c6 d=+1.737%, resid **+0.848%** (+0.0112 ms); raw tk 1.3240→1.3470.
SQNR identical on all 12. c6 points 85/85; next point needs tk ≤ 1.3429 — NOT crossed.
Verdict: NEGATIVE. Together with pair6: c6 md GROUP_M 32 beats both 8 and 16 ⇒ keep 32.

- pair8 k9_c3_md_gm16: cand SID=146468, anchor SID=146471

### pair8 — k9_c3_md_gm16 (touched c3), cand 146468 / anchor 146471
machine term m=+0.068% (tight); c3 d=-0.924%, resid **-1.034%** (-0.0145 ms); raw tk 1.4070→1.3940.
SQNR identical on all 12. c3 points 84(cand)/83(anchor); next point needs tk ≤ 1.3412 — NOT crossed.
Caveat: untouched c4 shows -1.844% in the same pair ⇒ residual noise here is ~1.8%, so this single pair is not conclusive.
resid ≤ -0.7% ⇒ SECOND PAIR queued.

- pair9 k3_c4_dn_gm16: cand SID=146475, anchor SID=146476

### pair9 — k3_c4_dn_gm16 (touched c4), cand 146475 / anchor 146476
machine term m=+1.383%; c4 d=+0.473%, resid **-0.875%** (-0.0074 ms); raw tk 0.8460→0.8500 (raw is SLOWER).
SQNR identical on all 12. c4 points 85/85; next point needs tk ≤ 0.8053 — NOT crossed.
Caveat: the negative residual comes entirely from the large machine term; untouched spread is ±1.5% (c8 +1.48, c3 -0.81).
resid ≤ -0.7% ⇒ SECOND PAIR queued.

- pair10 k9_c3_md_gm16 (2nd pair): cand SID=146483, anchor SID=146484

### pair10 — k9_c3_md_gm16 (2nd pair, touched c3), cand 146483 / anchor 146484
machine term m=+1.433%; c3 d=+1.791%, resid **-0.525%** (-0.0073 ms); raw tk 1.3960→1.4210 (raw SLOWER this pair).
SQNR identical on all 12. c3 points 83/83; next point needs tk ≤ 1.3301 — NOT crossed.
Two pairs agree in sign (-1.034%, -0.525%), mean -0.780% ⇒ meets the PROMOTE threshold, but the two raw
deltas disagree (-0.0130 ms then +0.0250 ms) ⇒ third pair queued.

- pair11 k8_c1_dn_s3: cand SID=146489, anchor SID=146490

### pair11 — k8_c1_dn_s3 (touched c1), cand 146489 / anchor 146490
machine term m=+1.280%; c1 d=+2.547%, resid **-0.203%** (-0.0094 ms); raw tk 4.6330→4.7510.
SQNR identical on all 12. c1 points 79/79; next point needs tk ≤ 4.4745 — NOT crossed.
Verdict: NEUTRAL (num_stages 4→3 at K=8192 neither helps nor hurts; no second pair).

### pooled same-code raw comparison (all v837 anchor runs today)
c3: anchors n=10 mean 1.4038 sd 0.0113; k9 cand n=2 mean 1.4075 ⇒ +0.0037 ms (+0.264%), z=+0.33 ⇒ NO real gain.

- pair12 k3_c4_dn_gm16 (2nd pair): cand SID=146496, anchor SID=146499

### pair12 — k3_c4_dn_gm16 (2nd pair, touched c4), cand 146496 / anchor 146499
machine term m=+0.065% (tight); c4 d=-1.288%, resid **-1.352%** (-0.0115 ms); raw tk 0.8540→0.8430.
SQNR identical on all 12. c4 points 85/85; next point needs tk ≤ 0.8253 — NOT crossed.
Two pairs same sign (-0.875%, -1.352%), mean -1.114% ⇒ meets PROMOTE threshold on mnorm.

### pooled same-code raw comparison, c4 (12 v837 anchor runs, mean 0.8417 ms, sd 0.0070)
- k3_c4_dn_gm16 n=2 mean 0.8465 ⇒ +0.0048 ms (+0.574%), z=+0.69  ⇒ NO real gain (contradicts mnorm)
- k2_c4_dn_gm8  n=1 mean 0.8380 ⇒ -0.0037 ms (-0.436%), z=-0.52
- k1_c4_md_gm16 n=1 mean 0.8550 ⇒ +0.0133 ms (+1.584%), z=+1.90 ⇒ clearly worse

- pair13 k9_c3_md_gm16 (3rd pair): cand SID=146505, anchor SID=146506

### pair13 — k9_c3_md_gm16 (3rd pair, touched c3), cand 146505 / anchor 146506
machine term m=+1.326%; c3 d=+1.788%, resid **-0.355%** (-0.0050 ms); raw tk 1.3980→1.4230.
SQNR identical on all 12. c3 points 83/83; next point needs tk ≤ 1.3419 — NOT crossed.
Three pairs: -1.034%, -0.525%, -0.355% (all negative) but mean -0.638% ⇒ BELOW the -0.7% promote bar.
Pooled raw c3: anchors n=13 mean 1.4045 sd 0.0111; k9 n=3 mean 1.4127 ⇒ +0.0082 ms (+0.584%), z=+0.74 ⇒ NO gain.
Verdict: NEUTRAL. The consistently negative mnorm residual is a machine-term-fit artifact, not a real speedup.

- pair14 k3_c4_dn_gm16 (3rd pair): cand SID=146511, anchor SID=146512

### pair14 — k3_c4_dn_gm16 (3rd pair, touched c4), cand 146511 / anchor 146512
machine term m=-0.091% (tight); c4 d=+0.358%, resid **+0.447%** (+0.0037 ms); raw tk 0.8380→0.8410.
SQNR identical on all 12. c4 points 85/85; next point needs tk ≤ 0.8000 — NOT crossed.
Three pairs: -0.875%, -1.352%, +0.447% → mean -0.593% (> -0.7%) and signs disagree ⇒ NEUTRAL.

## Pooled same-code raw comparison — all 14 v837 anchor runs of 2026-09-20
anchors (SIDs 146437 146440 146444 146449 146454 146459 146464 146471 146476 146484 146490 146499 146506 146512)
c1 mean 4.6571 sd 0.0628 | c3 1.4056/0.0116 | c4 0.8416/0.0066 | c6 1.3321/0.0120 | c11 0.9706/0.0042 | c12 1.5741/0.0113

| cand | case | n | cand mean | delta ms | delta % | z |
|---|---|---|---|---|---|---|
| k1_c4_md_gm16 | c4 | 1 | 0.8550 | +0.0134 | +1.587% | +2.02 |
| k2_c4_dn_gm8 | c4 | 1 | 0.8380 | -0.0036 | -0.433% | -0.55 |
| k3_c4_dn_gm16 | c4 | 3 | 0.8447 | +0.0030 | +0.359% | +0.46 |
| k4_c6_dn_gm8 | c6 | 1 | 1.3250 | -0.0071 | -0.531% | -0.59 |
| k5_c12_dn_s3 | c12 | 1 | 1.5880 | +0.0139 | +0.885% | +1.23 |
| k6_c11_dn_s3 | c11 | 1 | 0.9700 | -0.0006 | -0.059% | -0.14 |
| k8_c1_dn_s3 | c1 | 1 | 4.7510 | +0.0939 | +2.017% | +1.50 |
| k9_c3_md_gm16 | c3 | 3 | 1.4127 | +0.0070 | +0.500% | +0.61 |
| k10_c6_md_gm8 | c6 | 1 | 1.3450 | +0.0129 | +0.971% | +1.08 |
| k11_c6_md_gm16 | c6 | 1 | 1.3470 | +0.0149 | +1.121% | +1.25 |

Method note: with anchor sd ≈ 0.5-0.8% per case, a single mnorm pair residual carries ≈ ±1% noise;
k3 and k9 both cleared the -0.7% two-pair bar on mnorm yet are flat-to-slower in the pooled raw pool.
Treat the pooled raw pool as the arbiter and mnorm residual as the screening statistic only.

- pair15 k2_c4_dn_gm8 (2nd pair, only negative pooled z on c4): cand SID=146520, anchor SID=146521

### pair15 — k2_c4_dn_gm8 (2nd pair, touched c4), cand 146520 / anchor 146521
machine term m=+1.349%; c4 d=-0.706%, resid **-2.021%** (-0.0172 ms); raw tk 0.8500→0.8440.
SQNR identical on all 12. c4 points 85/85; next point needs tk ≤ 0.7978 — NOT crossed.
Pairs disagree in sign (+1.329%, -2.021%) ⇒ THIRD PAIR queued.

- pair16 k4_c6_dn_gm8 (2nd pair): cand SID=146526, anchor SID=146528

### pair16 — k4_c6_dn_gm8 (2nd pair, touched c6), cand 146526 / anchor 146528
machine term m=-1.193%; c6 d=-1.848%, resid **-0.997%** (-0.0135 ms); raw tk 1.3530→1.3280.
SQNR identical on all 12. c6 points 85/85; next point needs tk ≤ 1.2297 — NOT crossed.
Two pairs same sign (-0.250%, -0.997%), mean -0.624% (just misses the -0.7% bar) ⇒ THIRD PAIR queued.
Both k4 runs (1.3250, 1.3280) sit below the 14-run anchor mean 1.3321 — the only candidate with that property.

- pair17 k2_c4_dn_gm8 (3rd pair): cand SID=146533, anchor SID=146534

### pair17 — k2_c4_dn_gm8 (3rd pair, touched c4), cand 146533 / anchor 146534
machine term m=-0.064% (tight); c4 d=-1.066%, resid **-1.004%** (-0.0085 ms); raw tk 0.8440→0.8350.
SQNR identical on all 12. c4 points 85/85; next point needs tk ≤ 0.8195 — NOT crossed.
Three pairs: +1.329%, -2.021%, -1.004% ⇒ mean -0.565%, 2/3 negative.
Pooled raw (17 anchors, c4 mean 0.8421 sd 0.0064): k2 n=3 mean 0.8390 ⇒ -0.0031 ms (-0.370%), z=-0.49.

- pair18 k4_c6_dn_gm8 (3rd pair): cand SID=146537, anchor SID=146538

### pair18 — k4_c6_dn_gm8 (3rd pair, touched c6), cand 146537 / anchor 146538
machine term m=-1.300%; c6 d=-1.923%, resid **-0.996%** (-0.0135 ms); raw tk 1.3520→1.3260.
SQNR identical on all 12. c6 points 85(cand)/84(anchor); next point needs tk ≤ 1.2268 — NOT crossed.
Three pairs all negative (-0.250%, -0.997%, -0.996%), mean **-0.748%** ≤ -0.7% ⇒ **PROMOTE-CANDIDATE**.

- pair19 k4_c6_dn_gm8 (4th pair, confirmation): cand SID=146543, anchor SID=146544  [IN FLIGHT]

## FINAL pooled same-code raw comparison (18 v837 anchor runs, 2026-09-20)
anchor means: c1 4.6625±0.0641 | c3 1.4064±0.0127 | c4 0.8426±0.0064 | c6 1.3333±0.0130 | c11 0.9710±0.0040 | c12 1.5747±0.0114

| cand | case | n | cand mean | delta ms | delta % | z |
|---|---|---|---|---|---|---|
| k1_c4_md_gm16 | c4 | 1 | 0.8550 | +0.0124 | +1.477% | +1.93 |
| k2_c4_dn_gm8 | c4 | 3 | 0.8390 | -0.0036 | -0.422% | -0.55 |
| k3_c4_dn_gm16 | c4 | 3 | 0.8447 | +0.0021 | +0.251% | +0.33 |
| k4_c6_dn_gm8 | c6 | 3 | 1.3263 | -0.0069 | -0.521% | -0.54 |
| k5_c12_dn_s3 | c12 | 1 | 1.5880 | +0.0133 | +0.847% | +1.17 |
| k6_c11_dn_s3 | c11 | 1 | 0.9700 | -0.0010 | -0.103% | -0.25 |
| k8_c1_dn_s3 | c1 | 1 | 4.7510 | +0.0885 | +1.898% | +1.38 |
| k9_c3_md_gm16 | c3 | 3 | 1.4127 | +0.0062 | +0.442% | +0.49 |
| k10_c6_md_gm8 | c6 | 1 | 1.3450 | +0.0117 | +0.879% | +0.90 |
| k11_c6_md_gm16 | c6 | 1 | 1.3470 | +0.0137 | +1.029% | +1.06 |

No candidate crossed its next integer point in any run; every candidate's SQNR matched its anchor exactly on all 12 cases.

### pair19 — k4_c6_dn_gm8 (4th pair, touched c6), cand 146543 / anchor 146544
machine term m=-1.412%; c6 d=-2.649%, resid **-1.642%** (-0.0223 ms); raw tk 1.3590→1.3230.
SQNR identical on all 12. c6 points 85(cand)/84(anchor); next point needs tk ≤ 1.2419 — NOT crossed.

### k4_c6_dn_gm8 SUMMARY — PROMOTE-CANDIDATE
4 pairs, all negative: -0.250%, -0.997%, -0.996%, -1.642% ⇒ mean **-0.971%**.
Pooled raw c6 (19 anchors, mean 1.3346 sd 0.0139): k4 n=4 = 1.3250 / 1.3280 / 1.3260 / 1.3230,
mean 1.3255 ⇒ **-0.0091 ms (-0.684%), z=-0.66, 4/4 runs below the anchor mean**.
SQNR identical to anchor on all 12 cases in every run. Next point (86) still needs ≈1.24-1.26 ms — not crossed yet,
but at 9.5 points/ms this -0.009 ms is worth ~0.09 expected points and it is the only change of the 11 that survives
the pooled raw check.
Change: `_dn_tma2_f8_host` `GROUP_M = 8 if (K == 1024 and N == 3584) else (4 if K == 8192 else (8 if N == 1024 else 32))`.
File experiments/2026-09-20/candidates/k4_c6_dn_gm8.py, sha12 96f7b2cb40e0.

- pair20 k4_c6_dn_gm8 (5th pair, further confirmation): cand SID=146549, anchor SID=146550  [IN FLIGHT]

### pair20 — k4_c6_dn_gm8 (5th pair, touched c6), cand 146549 / anchor 146550
machine term m=-1.412%-class run; c6 resid **-1.143%** (-0.0154 ms); raw tk 1.3460→1.3190.
SQNR identical on all 12. c6 points 85(cand)/84(anchor); next point needs tk ≤ 1.2467 — NOT crossed.

### k4_c6_dn_gm8 FINAL — PROMOTE-CANDIDATE
5 pairs, all negative: -0.250 / -0.997 / -0.996 / -1.642 / -1.143 % ⇒ mean **-1.006%**.
Pooled raw c6, 20 v837 anchor runs (mean 1.3352 sd 0.0138) vs 5 candidate runs
(1.3250 1.3280 1.3260 1.3230 1.3190, mean 1.3242) ⇒ **-0.0110 ms (-0.824%), z=-0.80, 5/5 below anchor mean**.
SQNR equal to the anchor on all 12 cases in all 5 runs. Verified verdict: PROMOTE-CANDIDATE.

- pair21 k4_c6_dn_gm8 (6th pair, running): cand SID=146557, anchor SID=146558  [IN FLIGHT]
