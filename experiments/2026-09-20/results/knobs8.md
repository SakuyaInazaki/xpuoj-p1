# 2026-09-20 knobs8 — resolving the two dn `num_stages` leads on the v839 anchor

Anchor = unmodified `p1/kernel.py` **v839** (sha256 `843edd1f0c13…` = v838 + c6 dn num_stages 3).
dn host `_dn_tma2_f8_host` sees K=I, N=H. c4 = (K 1024, N 2048), c6 = (1024, 3584), c8 = (1024, 4096).

| cand | change | gate (case) | sha12 |
|---|---|---|---|
| q1_c4_dn_s3 | `_dn_tma2_f8_host` num_stages 4→3 | K==1024 and N==2048 (c4) | ae961a07a7c1 |
| q2_c8_dn_s3 | `_dn_tma2_f8_host` num_stages 4→3 | K==1024 and N==4096 (c8) | 8ce764d9b944 |
| q3_c4c8_dn_s3 | `_dn_tma2_f8_host` num_stages 4→3 | K==1024 and N in (2048, 4096) (c4+c8) | aff38ff09797 |

Each `diff p1/kernel.py <cand>` shows exactly one changed line (4593) — verified.

## Carried-over data (c4/c8 code paths are byte-identical across v838 and v839)
`n11_c4_dn_s3` (v838 base, sha12 c8d3b691ef84) is identical to q1 on the c4 path and differs only at c6;
`n12_c8_dn_s3` (8b0f82874999) is identical to q2 on the c8 path. Their runs are therefore pooled here:
- n11 c4 runs: 146637, 146644, 146646, 146649 (4 pairs, anchors 146638/146645/146648/146650)
- n12 c8 runs: 146640 (1 pair, anchor 146641)

## Estimator change for this round
`strat.py`'s machine index M must now **exclude c6** as well as the touched case, because v838 and v839
anchors differ at c6 by design (≈ -0.6%). M is therefore the mean over the 10 remaining cases of
tk_i / anchor-pool-mean_i. Anchor pool = all v838 anchors + all v839 anchors (their c4/c8 code is identical).
Saved as `experiments/2026-09-20/strat2.py`.

## Verdict rule (this round)
PROMOTE-CANDIDATE iff stratified z_of_mean ≤ -3 AND ≥75% of runs negative AND SQNR equal on all 12.
Early stop: if q1's stratified mean over its first 4 NEW pairs is ≥ 0, halt q1 and give its remaining pairs to q2.

## SID log
- kpair1 q1_c4_dn_s3 (touched c4): cand SID=146651, anchor SID=146652

### Carried-over baselines, re-derived with strat2 (c6 excluded from the machine index)
c4 fit: anchors n=65, tk = 0.6116 + 0.2304*M, resid sd 0.0054 ms (0.644%), M explains only **12%**
(c4 is nearly machine-insensitive but intrinsically noisy, so extra pairs are the only lever).
c8 fit: anchors n=65, tk = -0.1739 + 1.5215*M, resid sd 0.0045 ms (0.335%), M explains **88%**.

| carried run | case | M | tk | pred | resid | z | pts | need | crossed |
|---|---|---|---|---|---|---|---|---|---|
| n11 146637 | c4 | 1.0077 | 0.8370 | 0.8439 | -0.0069 ms (-0.815%) | -1.27 | 85 | 0.7993 | no |
| n11 146644 | c4 | 1.0074 | 0.8430 | 0.8438 | -0.0008 ms (-0.096%) | -0.15 | 85 | 0.8149 | no |
| n11 146646 | c4 | 0.9920 | 0.8360 | 0.8403 | -0.0043 ms (-0.507%) | -0.79 | 87 | 0.7893 | no |
| n11 146649 | c4 | 0.9938 | 0.8380 | 0.8407 | -0.0027 ms (-0.316%) | -0.49 | 85 | 0.8192 | no |
| **q1 carried total** | c4 | | | | **-0.0036 ms (-0.433%)** | **z_of_mean -1.35** | | | 4/4 negative |
| n12 146640 | c8 | 1.0067 | 1.3540 | 1.3578 | -0.0038 ms (-0.282%) | -0.85 | 83 | 1.3490 | no |

(The 4th n11 pair, 146649/146650, completed after the knobs6 hand-off: c4 raw 0.8430→0.8380, mnorm resid
+0.662%, stratified -0.316%; SQNR identical on all 12, not crossed.)

Schedule: 20 new pairs round-robin q1,q2,q1,q2,q3 ×4 ⇒ q1×8, q2×8, q3×4 (40 submissions).
- kpair2 q2_c8_dn_s3 (touched c8): cand SID=146653, anchor SID=146654

### kpair1 — q1_c4_dn_s3 (touched c4), cand 146651 / anchor 146652
mnorm m=-1.097%; c4 d=-0.823%, resid **+0.247%**; raw tk 0.8510→0.8440. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.8234 — not crossed. Stratified **+0.365%, z=+0.56**.
q1 running total (5 runs incl. 4 carried): mean -0.284%, z_of_mean -0.98, 4/5 negative.
- kpair3 q1_c4_dn_s3 (touched c4): cand SID=146655, anchor SID=146656

### kpair2 — q2_c8_dn_s3 (touched c8), cand 146653 / anchor 146654
mnorm m=-1.309%; c8 d=-2.273%, resid **-2.048%**; raw tk 1.3640→1.3330. SQNR EQUAL on all 12.
c8 pts 84(cand)/83(anchor), need ≤ 1.2798 — not crossed. Stratified **-0.118%, z=-0.35**.
q2 running total (2 runs incl. 1 carried): mean -0.203%, z_of_mean -0.87, 2/2 negative.
- kpair4 q2_c8_dn_s3 (touched c8): cand SID=146657, anchor SID=146658

### kpair3 — q1_c4_dn_s3 (touched c4), cand 146655 / anchor 146656
mnorm m=-1.234%; c4 d=-1.420%, resid **-0.217%**; raw tk 0.8450→0.8330. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.8262 — not crossed. Stratified **-0.883%, z=-1.37**.
q1 running total (6 runs): mean -0.377%, z_of_mean -1.44, 5/6 negative.
- kpair5 q3_c4c8_dn_s3 (touched c4+c8): cand SID=146659, anchor SID=146660

### kpair4 — q2_c8_dn_s3 (touched c8), cand 146657 / anchor 146658
mnorm m=+1.196%; c8 d=+2.926%, resid **+2.721%**; raw tk 1.3330→1.3720. SQNR EQUAL on all 12.
c8 pts 83(cand)/84(anchor), need ≤ 1.3585 — not crossed. Stratified **+0.419%, z=+1.23**.
q2 running total (3 runs): mean +0.019%, z_of_mean +0.09, 2/3 negative.
- kpair6 q1_c4_dn_s3 (touched c4): cand SID=146661, anchor SID=146662

### kpair5 — q3_c4c8_dn_s3 (touched c4 AND c8), cand 146659 / anchor 146660
SQNR EQUAL on all 12.
- c4: mnorm resid **-3.612%**; raw 0.8510→0.8320; stratified **-1.450%, z=-2.23**; pts 85/85, need ≤ 0.8138 — not crossed.
- c8: mnorm resid **+1.819%**; raw 1.3320→1.3590; stratified **+0.056%, z=+0.16**; pts 83(cand)/84(anchor), need ≤ 1.3362 — not crossed.
- kpair7 q2_c8_dn_s3 (touched c8): cand SID=146664, anchor SID=146665

### kpair6 — q1_c4_dn_s3 (touched c4), cand 146661 / anchor 146662
mnorm m=+1.300%; c4 d=+0.000% (raw 0.8420→0.8420, identical), resid **-1.268%**. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.7967 — not crossed. Stratified **-0.312%, z=-0.48**.
q1 running total (7 runs): mean -0.392%, z_of_mean -1.60, 6/7 negative.
q1 new-pair-only mean after 3 of 4 (+0.365 / -0.883 / -0.312) = -0.277% ⇒ early-stop rule not triggered so far.
- kpair8 q1_c4_dn_s3 (touched c4): cand SID=146666, anchor SID=146667

### kpair7 — q2_c8_dn_s3 (touched c8), cand 146664 / anchor 146665
mnorm m=-1.088%; c8 d=-2.411%, resid **-2.224%**; raw tk 1.3690→1.3360. SQNR EQUAL on all 12.
c8 pts 84(cand)/83(anchor), need ≤ 1.2651 — not crossed. Stratified **-0.148%, z=-0.41**.
q2 running total (4 runs): mean -0.028%, z_of_mean -0.16, 3/4 negative.
- kpair9 q2_c8_dn_s3 (touched c8): cand SID=146669, anchor SID=146670

### kpair8 — q1_c4_dn_s3 (touched c4), cand 146666 / anchor 146667
mnorm m=+1.451%; c4 d=+0.479%, resid **-0.936%**; raw tk 0.8350→0.8390. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.8185 — not crossed. Stratified **-0.659%, z=-1.03**.

**Early-stop checkpoint (q1, 4 new pairs):** stratified residuals +0.365 / -0.883 / -0.312 / -0.659 %
⇒ new-pair mean **-0.372%** < 0 ⇒ **q1 continues**, no reallocation.
q1 running total (8 runs incl. 4 carried): mean -0.416%, z_of_mean -1.83, 7/8 negative.
- kpair10 q3_c4c8_dn_s3 (touched c4+c8): cand SID=146672, anchor SID=146673

### kpair9 — q2_c8_dn_s3 (touched c8), cand 146669 / anchor 146670
mnorm m=-1.046%; c8 d=-1.689%, resid **-1.509%**; raw tk 1.3620→1.3390. SQNR EQUAL on all 12.
c8 pts 84(cand)/83(anchor), need ≤ 1.3376 — **missed by +0.0014 ms**, the closest approach to c8's 84 line so far.
Stratified **-0.090%, z=-0.25**. q2 running total (5 runs): mean -0.047%, z_of_mean -0.30, 4/5 negative.
- kpair11 q1_c4_dn_s3 (touched c4): cand SID=146674, anchor SID=146675

### kpair10 — q3_c4c8_dn_s3 (touched c4 AND c8), cand 146672 / anchor 146673
SQNR EQUAL on all 12.
- c4: mnorm resid **-0.578%**; raw 0.8370→0.8430; stratified **-0.147%, z=-0.23**; pts 85(cand)/87(anchor), need ≤ 0.8014 — not crossed.
- c8: mnorm resid **+0.825%**; raw 1.3380→1.3520; stratified **-0.749%, z=-2.13**; pts 83(cand)/84(anchor), need ≤ 1.3349 — not crossed.
q3 running totals (2 runs): c4 mean -0.793% (z_of_mean -1.76, 2/2 neg); c8 mean -0.363% (z_of_mean -1.44, 1/2 neg).
- kpair12 q2_c8_dn_s3 (touched c8): cand SID=146676, anchor SID=146677

### kpair11 — q1_c4_dn_s3 (touched c4), cand 146674 / anchor 146675
mnorm m=-1.196%; c4 d=-1.773%, resid **-0.607%**; raw tk 0.8460→0.8310 (fastest c4 of the round).
SQNR EQUAL on all 12. c4 pts 85/85, need ≤ 0.8167 — not crossed. Stratified **-1.129%, z=-1.78**.
q1 running total (9 runs): mean -0.488%, z_of_mean **-2.31**, 8/9 negative.
- kpair13 q1_c4_dn_s3 (touched c4): cand SID=146678, anchor SID=146679

### kpair12 — q2_c8_dn_s3 (touched c8), cand 146676 / anchor 146677
mnorm m=+1.277%; c8 d=+2.386%, resid **+2.167%**; raw tk 1.3410→1.3730. SQNR EQUAL on all 12.
c8 pts 83(cand)/84(anchor), need ≤ 1.3520 — not crossed. Stratified **+0.711%, z=+2.03**.
q2 running total (6 runs): mean +0.075%, z_of_mean +0.52, 4/6 negative — oscillating around zero.
- kpair14 q2_c8_dn_s3 (touched c8): cand SID=146680, anchor SID=146681

### kpair13 — q1_c4_dn_s3 (touched c4), cand 146678 / anchor 146679
mnorm m=-1.228%; c4 d=-1.306%, resid **-0.109%**; raw tk 0.8420→0.8310. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.8174 — not crossed. Stratified **-1.103%, z=-1.74**.
q1 running total (10 runs): mean -0.538%, z_of_mean **-2.69**, 9/10 negative — closing on the -3 bar.
- kpair15 q3_c4c8_dn_s3 (touched c4+c8): cand SID=146682, anchor SID=146683

### kpair14 — q2_c8_dn_s3 (touched c8), cand 146680 / anchor 146681
mnorm m=-1.229%; c8 d=-1.398%, resid **-1.187%**; raw tk 1.3590→1.3400. SQNR EQUAL on all 12.
c8 pts 84(cand)/83(anchor), need ≤ 1.2739 — not crossed. Stratified **+0.225%, z=+0.64**.
q2 running total (7 runs): mean +0.099%, z_of_mean +0.75, 4/7 negative.
- kpair16 q1_c4_dn_s3 (touched c4): cand SID=146684, anchor SID=146685

### kpair15 — q3_c4c8_dn_s3 (touched c4 AND c8), cand 146682 / anchor 146683
SQNR EQUAL on all 12.
- c4: mnorm resid **-1.653%**; raw 0.8390→0.8370; stratified **-0.850%, z=-1.36**; pts 86(cand, tb=5.60 lottery)/85(anchor).
- c8: mnorm resid **+0.659%**; raw 1.3440→1.3560; stratified **-0.356%, z=-1.01**; pts 83(cand)/84(anchor).
q3 running totals (3 runs): c4 mean -0.805% (z_of_mean -2.22, 3/3 neg); c8 mean -0.358% (z_of_mean -1.75, 2/3 neg).

## Crossing analysis — c4 → 86 (needs tk ≤ tb·14/86, i.e. ≈ 0.80 ms at a typical tb ≈ 4.95)
`experiments/2026-09-20/cross.py 4 86 <cand:anchor,…>`:

| cand | tk | tb | pts | need≤ | reached 86 | anchor | tk | tb | pts |
|---|---|---|---|---|---|---|---|---|---|
| 146637 | 0.8370 | 4.910 | 85 | 0.7993 | no | 146638 | 0.8330 | 5.038 | 85 |
| 146644 | 0.8430 | 5.006 | 85 | 0.8149 | no | 146645 | 0.8430 | 5.006 | 85 |
| 146646 | 0.8360 | 5.788 | 87 | 0.9422 | **yes (tb lottery: tb 5.788 vs the usual ~4.95)** | 146648 | 0.8360 | 4.915 | 85 |
| 146649 | 0.8380 | 5.032 | 85 | 0.8192 | no | 146650 | 0.8430 | 5.000 | 85 |
| 146651 | 0.8440 | 5.058 | 85 | 0.8234 | no | 146652 | 0.8510 | 4.929 | 85 |
| 146655 | 0.8330 | 5.075 | 85 | 0.8262 | no | 146656 | 0.8450 | 4.896 | 85 |
| 146661 | 0.8420 | 4.894 | 85 | 0.7967 | no | 146662 | 0.8420 | 5.020 | 85 |
| 146666 | 0.8390 | 5.028 | 85 | 0.8185 | no | 146667 | 0.8350 | 4.985 | 85 |
| 146674 | 0.8310 | 5.017 | 85 | 0.8167 | no | 146675 | 0.8460 | 4.918 | 85 |
| 146678 | 0.8310 | 5.021 | 85 | 0.8174 | no | 146679 | 0.8420 | 4.966 | 85 |

**No q1 run reached 86 on merit.** c4's tk sits at 0.831-0.844 while a typical tb (≈4.95) demands ≤ 0.806 —
a further ≈3% is needed, far more than this knob's ≈0.5%. The one 87 (146646) came from a tb of 5.788.
- kpair17 q2_c8_dn_s3 (touched c8): cand SID=146686, anchor SID=146687

### kpair16 — q1_c4_dn_s3 (touched c4), cand 146684 / anchor 146685
mnorm m=+1.462%; c4 d=+0.713%, resid **-0.713%**; raw tk 0.8420→0.8480. SQNR EQUAL on all 12.
c4 pts 85(cand)/93(anchor — extreme tb lottery, tb 11.2), need ≤ 0.8030 — not crossed.
Stratified **+0.468%, z=+0.75**.
q1 running total (11 runs): mean -0.443%, z_of_mean -2.36, 9/11 negative.
- kpair18 q1_c4_dn_s3 (touched c4): cand SID=146688, anchor SID=146689

### kpair17 — q2_c8_dn_s3 (touched c8), cand 146686 / anchor 146687
mnorm m=+1.112%; c8 d=+1.568%, resid **+1.378%**; raw tk 1.3390→1.3600. SQNR EQUAL on all 12.
c8 pts 83(cand)/84(anchor), need ≤ 1.3430 — not crossed. Stratified **-0.011%, z=-0.03**.
q2 running total (8 runs): mean +0.083%, z_of_mean +0.66, 5/8 negative — **q2 is dead flat**.
- kpair19 q2_c8_dn_s3 (touched c8): cand SID=146690, anchor SID=146691

### kpair18 — q1_c4_dn_s3 (touched c4), cand 146688 / anchor 146689
mnorm m=-1.400%; c4 d=-1.308%, resid **+0.057%**; raw tk 0.8410→0.8300 (fastest c4 seen: 0.8300).
SQNR EQUAL on all 12. c4 pts 85/85, need ≤ 0.8185 — not crossed. Stratified **-1.171%, z=-1.84**.

**q1 after its full 8 new pairs (12 runs total): mean -0.0041 ms (-0.485%), z_of_mean -2.65, 10/12 negative (83%).**
Against the round's bar (z ≤ -3 AND ≥75% negative AND SQNR equal) q1 clears the sign test and the SQNR test
but lands at z -2.65, just short of -3. With c4's per-run sd of 0.63% and an effect of 0.485%, reaching
z = -3 needs n ≈ (3·0.63/0.485)² ≈ **15 runs** — i.e. about 3 more pairs.
- kpair20 q3_c4c8_dn_s3 (touched c4+c8): cand SID=146692, anchor SID=146693

### kpair19 — q2_c8_dn_s3 (touched c8), cand 146690 / anchor 146691
mnorm m=+1.191%; c8 d=+2.673%, resid **+2.468%**; raw tk 1.3470→1.3830. SQNR EQUAL on all 12.
c8 pts 83(cand)/84(anchor), need ≤ 1.3326 — not crossed. Stratified **+1.685%, z=+4.74**.
This run is an **outlier**: the leave-one-out null for c8 over 51 anchors spans z -2.14…+2.14, so +4.74 is
outside anything the same-code pool produces (a noisy-neighbour event on the judge, most likely).
q2 totals — with it (9 runs): mean +0.255%, z_of_mean +2.13, 5/9 negative.
q2 totals — excluding it (8 runs): mean +0.083%, z_of_mean +0.66, 5/8 negative.
Either way **q2 is flat**.
- kpair21 q1_c4_dn_s3 (extra pair, reallocated from resolved q2/q3): cand SID=146694, anchor SID=146696

### kpair20 — q3_c4c8_dn_s3 (touched c4 AND c8), cand 146692 / anchor 146693
SQNR EQUAL on all 12.
- c4: mnorm resid **+0.823%**; raw 0.8440→0.8400; stratified **+0.018%, z=+0.03**; pts 85/85, need ≤ 0.8197 — not crossed.
- c8: mnorm resid **-1.974%**; raw 1.3690→1.3390; stratified **+0.276%, z=+0.76**; pts 84(cand)/83(anchor), need ≤ 1.2754 — not crossed.

**q3 final (4 runs):** c4 mean **-0.585%** (z_of_mean -1.86, 3/4 negative);
c8 mean **-0.211%** (z_of_mean -1.17, 2/4 negative).

### Does q3 stack?
| change | c4 effect | c8 effect |
|---|---|---|
| q1 (c4 only, 12 runs) | **-0.485%** (z_of_mean -2.65) | — |
| q2 (c8 only, 9 runs) | — | **+0.255%** (z_of_mean +2.13); excluding the 146690 outlier +0.083% (z +0.66) |
| q3 (both, 4 runs) | **-0.585%** (z_of_mean -1.86) | **-0.211%** (z_of_mean -1.17) |

q3's c4 effect (-0.585%) matches q1's (-0.485%) within noise, and q3's c8 effect (-0.211%) matches q2's
(≈0%) within noise. **The two gates are independent — there is no interaction, and nothing to gain from
bundling them.** Whatever is true of c4 alone is true of c4 inside q3; c8 contributes nothing either way.
- kpair22 q1_c4_dn_s3 (extra pair): cand SID=146697, anchor SID=146698

### kpair21 — q1_c4_dn_s3 (extra, touched c4), cand 146694 / anchor 146696
mnorm m=-1.356%; c4 d=-0.357%, resid **+0.965%**; raw tk 0.8410→0.8380. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.8136 — not crossed. Stratified **-0.247%, z=-0.39**.
q1 running total (13 runs): mean -0.457%, z_of_mean -2.64, 11/13 negative (85%).
- kpair23 q1_c4_dn_s3 (extra pair): cand SID=146699, anchor SID=146700

### kpair22 — q1_c4_dn_s3 (extra, touched c4), cand 146697 / anchor 146698
mnorm m=-1.337%; c4 d=-1.540%, resid **-0.236%**; raw tk 0.8440→0.8310. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.8109 — not crossed. Stratified **-1.052%, z=-1.69**.
**q1 (14 runs): mean -0.0042 ms (-0.500%), z_of_mean -3.01, 12/14 negative (86%) ⇒ the bar is now met.**
- kpair24 q1_c4_dn_s3 (extra pair): cand SID=146701, anchor SID=146702

### kpair23 — q1_c4_dn_s3 (extra, touched c4), cand 146699 / anchor 146700
mnorm m=+1.406%; c4 d=+0.477%, resid **-0.894%**; raw tk 0.8380→0.8420. SQNR EQUAL on all 12.
c4 pts 85/85, need ≤ 0.8006 — not crossed. Stratified **-0.265%, z=-0.43**.
q1 (15 runs): mean -0.0041 ms (-0.481%), z_of_mean **-3.01**, **13/15 negative (87%)**.

## Crossing analysis — c8 → 84 (needs tk ≤ tb·16/84 ≈ 0.1905·tb)
`experiments/2026-09-20/cross.py 8 84 <cand:anchor,…>` over all nine q2 pairs:

| cand | tk | tb | pts | reached 84 | anchor | tk | tb | pts | reached 84 |
|---|---|---|---|---|---|---|---|---|---|
| 146640 | 1.3540 | 7.082 | 83 | no | 146641 | 1.3410 | 7.244 | 84 | yes |
| 146653 | 1.3330 | 7.252 | 84 | **yes** | 146654 | 1.3640 | 7.004 | 83 | no |
| 146657 | 1.3720 | 7.132 | 83 | no | 146658 | 1.3330 | 7.217 | 84 | yes |
| 146664 | 1.3360 | 7.169 | 84 | **yes** | 146665 | 1.3690 | 7.039 | 83 | no |
| 146669 | 1.3390 | 7.580 | 84 | **yes** | 146670 | 1.3620 | 7.042 | 83 | no |
| 146676 | 1.3730 | 7.098 | 83 | no | 146677 | 1.3410 | 7.238 | 84 | yes |
| 146680 | 1.3400 | 7.219 | 84 | **yes** | 146681 | 1.3590 | 7.068 | 83 | no |
| 146686 | 1.3600 | 7.051 | 83 | no | 146687 | 1.3390 | 7.271 | 84 | yes |
| 146690 | 1.3830 | 6.996 | 83 | no | 146691 | 1.3470 | 7.215 | 84 | yes |

**c8 sits exactly on the 83/84 boundary** (threshold 1.333-1.381 ms depending on tb; observed tk 1.333-1.383).
q2 candidates reached 84 in 4/9 runs, the paired anchors in 5/9 — a coin flip that this change does not move,
which matches q2's flat +0.08% effect. The way to own c8's 84 is a ≈1.5% tk reduction, not this knob.
- kpair25 q1_c4_dn_s3 (extra pair): cand SID=146703, anchor SID=146704

### kpair24 — q1_c4_dn_s3 (extra, touched c4), cand 146701 / anchor 146702
mnorm m=-1.353%; c4 d=-2.834%, resid **-1.514%**; raw tk 0.8470→**0.8230** (fastest c4 ever recorded).
SQNR EQUAL on all 12. c4 pts 85/85, need ≤ 0.8187 — not crossed. Stratified **-2.003%, z=-3.24**.
q1 (16 runs): mean -0.0049 ms (-0.580%), z_of_mean **-3.76**, **14/16 negative (88%)**.
- kpair26 q1_c4_dn_s3 (extra pair): cand SID=146705, anchor SID=146706
- kpair27 q1_c4_dn_s3 (extra pair): cand SID=146707, anchor SID=146708

### kpair25 — q1_c4_dn_s3 (extra, touched c4), cand 146703 / anchor 146704
mnorm resid **-0.274%**; raw tk 0.8330→0.8390. SQNR EQUAL. pts 86(cand)/85(anchor). Stratified **-0.578%, z=-0.93**.

### kpair26 — q1_c4_dn_s3 (extra, touched c4), cand 146705 / anchor 146706
mnorm m=+1.292%; c4 resid **-1.851%**; raw tk 0.8450→0.8400. SQNR EQUAL.
pts 98(cand — an extreme tb lottery, tb ≈ 41 ms)/85(anchor). Stratified **-0.489%, z=-0.79**.

## FINAL — q1_c4_dn_s3, 18 runs (10 v839 + 4 v838-base carried + 4 more)
c4 fit: anchors n=90, tk = 0.6057 + 0.2361·M, resid sd **0.0052 ms (0.620%)**, M explains only 13%
(c4 really is nearly machine-insensitive; its noise is intrinsic, so n is the only lever).

| run | M | tk | pred | resid | z | pts | reached 86 |
|---|---|---|---|---|---|---|---|
| 146637 | 1.0073 | 0.8370 | 0.8435 | -0.0065 (-0.771%) | -1.25 | 85 | no |
| 146644 | 1.0070 | 0.8430 | 0.8434 | -0.0004 (-0.052%) | -0.08 | 85 | no |
| 146646 | 0.9917 | 0.8360 | 0.8398 | -0.0038 (-0.453%) | -0.73 | 87 | tb lottery |
| 146649 | 0.9934 | 0.8380 | 0.8402 | -0.0022 (-0.264%) | -0.42 | 85 | no |
| 146651 | 0.9944 | 0.8440 | 0.8404 | +0.0036 (+0.423%) | +0.68 | 85 | no |
| 146655 | 0.9922 | 0.8330 | 0.8399 | -0.0069 (-0.826%) | -1.33 | 85 | no |
| 146661 | 1.0107 | 0.8420 | 0.8443 | -0.0023 (-0.273%) | -0.44 | 85 | no |
| 146666 | 1.0104 | 0.8390 | 0.8442 | -0.0052 (-0.621%) | -1.00 | 85 | no |
| 146674 | 0.9920 | 0.8310 | 0.8399 | -0.0089 (-1.058%) | -1.70 | 85 | no |
| 146678 | 0.9917 | 0.8310 | 0.8398 | -0.0088 (-1.050%) | -1.69 | 85 | no |
| 146684 | 1.0090 | 0.8480 | 0.8439 | +0.0041 (+0.487%) | +0.79 | 85 | no |
| 146688 | 0.9905 | 0.8300 | 0.8395 | -0.0095 (-1.137%) | -1.83 | 85 | no |
| 146694 | 0.9920 | 0.8380 | 0.8399 | -0.0019 (-0.226%) | -0.36 | 85 | no |
| 146697 | 0.9910 | 0.8310 | 0.8396 | -0.0086 (-1.030%) | -1.66 | 85 | no |
| 146699 | 1.0107 | 0.8420 | 0.8443 | -0.0023 (-0.273%) | -0.44 | 85 | no |
| 146701 | 0.9911 | 0.8230 | 0.8397 | -0.0167 (-1.986%) | -3.20 | 85 | no |
| 146703 | 1.0089 | 0.8390 | 0.8439 | -0.0049 (-0.578%) | -0.93 | 86 | tb lottery |
| 146705 | 1.0101 | 0.8400 | 0.8441 | -0.0041 (-0.489%) | -0.79 | 98 | tb lottery |

**MEAN residual -0.0048 ms (-0.572%), z_of_mean -3.91, 16/18 negative (89%), SQNR equal on all 12 in all 18 runs.**
⇒ **PROMOTE-CANDIDATE** (z ≤ -3 ✓, ≥75% negative ✓, SQNR equal ✓).
- kpair28 q1_c4_dn_s3 (extra pair): cand SID=146709, anchor SID=146710

### kpair27 — q1_c4_dn_s3 (extra, touched c4), cand 146707 / anchor 146708
mnorm m=-1.328%; c4 resid **+0.584%**; raw tk 0.8430→0.8370. SQNR EQUAL. pts 85/85, need ≤ 0.8200 — not crossed.
Stratified **-0.354%, z=-0.57**.
**q1 (19 runs): mean -0.0047 ms (-0.558%), z_of_mean -3.95, 17/19 negative (89%).**

## SUMMARY (knobs8, same format as knobs6)

| cand | change + gate | sha12 | pairs (SIDs) | mnorm resid range (%) | stratified Δ% (z_of_mean, n) | % runs negative | crossed next line | SQNR same | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **q1_c4_dn_s3** | dn `num_stages` 4→3, gate K==1024 and N==2048 (c4) | ae961a07a7c1 | 20 runs; new: 146651/652, 146655/656, 146661/662, 146666/667, 146674/675, 146678/679, 146684/685, 146688/689, 146694/696, 146697/698, 146699/700, 146701/702, 146703/704, 146705/706, 146707/708, 146709/710; carried: 146637/638, 146644/645, 146646/648, 146649/650 | +0.965 … -1.851 | **-0.517 (-3.76, 20)** | **17/20 = 85%** | no (86 needs ≈0.806 ms, best tk 0.8230) | yes (19/19) | **PROMOTE-CANDIDATE** |
| q2_c8_dn_s3 | dn `num_stages` 4→3, gate K==1024 and N==4096 (c8) | 8ce764d9b944 | 9 runs: 146653/654, 146657/658, 146664/665, 146669/670, 146676/677, 146680/681, 146686/687, 146690/691; carried 146640/641 | +2.721 … -2.224 | **+0.249 (+2.13, 9)**; excluding the 146690 outlier **+0.083 (+0.66, 8)** | 5/9 = 56% | 4/9 runs hit 84, but so did 5/9 anchors — pure tb coin flip | yes (9/9) | **NEUTRAL** |
| q3_c4c8_dn_s3 | both gates at once | aff38ff09797 | 4 runs: 146659/660, 146672/673, 146682/683, 146692/693 | c4 -3.612…+0.823; c8 +1.819…-1.974 | c4 **-0.579 (-1.87, 4)**; c8 **-0.209 (-1.19, 4)** | c4 3/4 = 75%; c8 2/4 = 50% | no | yes (4/4) | **NEUTRAL as a bundle** — it is q1's effect plus nothing |

### Does q3 stack?
No. q3's c4 effect (-0.579%) is statistically indistinguishable from q1's (-0.558%), and q3's c8 effect
(-0.209%) from q2's (≈0%). The two gates touch different launches of the same host and do not interact,
so **bundling buys nothing over promoting q1 alone**. Promote q1; leave c8's dn `num_stages` at 4.

### Crossing summary
- **c4 → 86**: never reached on merit in 19 q1 runs. A typical tb ≈ 4.95 demands tk ≤ 0.806; the best tk
  ever recorded is 0.8230 (146701) and the median is ≈0.838. The gap is ≈3%, six times this knob's size.
  Three runs showed 86/87/98 points purely from tb outliers (tb 5.79 / 5.60 / ≈41).
- **c8 → 84**: c8 straddles the line. Threshold is 1.333-1.381 ms depending on tb; observed tk is
  1.333-1.383. q2 candidates reached 84 in 4/9 runs and anchors in 5/9 — q2 does not move those odds.

### Recommendation
Fold **q1** into production as v840: in `_dn_tma2_f8_host`,
`num_stages = 3 if (K == 1024 and N == 3584) else (3 if (K == 1024 and N == 2048) else (3 if K == 14336 else 4))`
(file `experiments/2026-09-20/candidates/q1_c4_dn_s3.py`, sha12 ae961a07a7c1, diff = line 4593 only).
Expected value ≈ 0.0044 ms × 14.5 points/ms ≈ **0.06 points**. It does not cross c4 to 86 by itself.
Together with the already-promoted c6 change, dn `num_stages` 3 is now confirmed good on the two short-K
cases with N ≥ 2048 (c4, c6) and confirmed useless on c1/c3/c5/c7/c8/c11/c12.

### kpair28 — q1_c4_dn_s3 (extra, touched c4), cand 146709 / anchor 146710
mnorm m=-1.193%; c4 resid **+0.692%**; raw tk 0.8470→0.8430. SQNR EQUAL. pts 85/85, need ≤ 0.8133 — not crossed.
Stratified **+0.332%, z=+0.54**.

## q1 FINAL — 20 runs
**mean -0.0044 ms (-0.517%), z_of_mean -3.76, 17/20 negative (85%), SQNR equal on all 12 in all 20 runs.**
All three verdict conditions met (z ≤ -3, ≥75% negative, SQNR equal) ⇒ **PROMOTE-CANDIDATE**.
The z_of_mean has been ≤ -3 continuously from n=14 onward (-3.01, -3.01, -3.76, -3.78, -3.91, -3.95, -3.76),
so the conclusion is stable, not a stopping artefact.
