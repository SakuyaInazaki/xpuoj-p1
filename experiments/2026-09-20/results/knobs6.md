# 2026-09-20 knobs6 — gated launch-parameter candidates on top of v838

Anchor = unmodified `p1/kernel.py` **v838** (sha256 `96f7b2cb40e0…` = v837 + c6 dn GROUP_M 8).
Case key = (T, H, E, I, topk) from `_KNOWN12`; dn host `_dn_tma2_f8_host` sees K=I, N=H;
md host `_fgs_tma1_intq_host` sees K=H, I=intermediate, G=E.

| cand | change | gate (case) | sha12 |
|---|---|---|---|
| n1_c6_dn_gm4  | `_dn_tma2_f8_host` GROUP_M 8→4  | K==1024 and N==3584 (c6) | 2101a3dcfcbd |
| n2_c6_dn_gm16 | `_dn_tma2_f8_host` GROUP_M 8→16 | K==1024 and N==3584 (c6) | 06586c010046 |
| n3_c6_dn_s3   | `_dn_tma2_f8_host` num_stages 4→3 | K==1024 and N==3584 (c6) | 843edd1f0c13 |
| n4_c8_dn_gm8  | `_dn_tma2_f8_host` GROUP_M 32→8  | K==1024 and N==4096 (c8) | cf4b28dff1ee |
| n5_c8_dn_gm16 | `_dn_tma2_f8_host` GROUP_M 32→16 | K==1024 and N==4096 (c8) | 8eb3011fafbb |
| n6_c4_md_reg232 | `_g` md `maxnreg=232` added (num_stages stays 3) | G==32 and I==1024 (c4) | 2a1ffe8c0e07 |
| n7_c12_md_gm16 | TMA md GROUP_M 8→16 | K==1024 and I==2048 (c12) | 342c8d3cd19e |
| n8_c11_md_gm16 | TMA md GROUP_M 8→16 | K==1024 and I==1024 (c11) | 8ed843433bf8 |
| n9_c8_md_gm16 | `_g` md GROUP_M 32→16 | `_gg` = G==96 and I==1024 (c8) | 141db2ed8436 |

Each `diff p1/kernel.py <cand>` shows exactly one changed line (verified).

## SID log

- pair1 n1_c6_dn_gm4 (touched c6): cand SID=146563, anchor SID=146564
- pair2 n4_c8_dn_gm8 (touched c8): cand SID=146567, anchor SID=146568

### pair1 — n1_c6_dn_gm4 (touched c6), cand 146563 / anchor 146564
machine term m=+1.330%; c6 d=+2.271%, resid **+1.323%** (+0.0175 ms); raw tk 1.3210→1.3510 (+0.0300 ms, SLOWER).
SQNR identical on all 12 (23.12/23.14/23.19/23.19/23.13/23.14/23.12/23.12/23.13/23.11/23.31/23.25).
c6 points 89(cand, lucky tb=11.27)/85(anchor, tb=7.55); with its own tb the cand needs tk ≤ 1.2526 — NOT crossed.
Verdict: **NEGATIVE** — GROUP_M 4 is clearly worse than the promoted 8; no second pair.

### pair2 — n4_c8_dn_gm8 (touched c8), cand 146567 / anchor 146568
machine term m=+1.335%; c8 d=+2.021%, resid **+1.792%** (+0.0239 ms); raw tk 1.3360→1.3630 (+0.0270 ms, SLOWER).
SQNR identical on all 12. c8 points 83(cand)/84(anchor); cand needs tk ≤ 1.3227 with its own tb — NOT crossed.
Verdict: **NEGATIVE** — dn GROUP_M 8 is worse than 32 at c8; no second pair.

- pair3 n6_c4_md_reg232 (touched c4): cand SID=146571, anchor SID=146572
- pair4 n7_c12_md_gm16 (touched c12): cand SID=146575, anchor SID=146576

### pair3 — n6_c4_md_reg232 (touched c4), cand 146571 / anchor 146572
machine term m=-1.253% (candidate ran on a fast machine); c4 d=-1.420%, resid **-0.198%** (-0.0017 ms);
raw tk 0.8450→0.8330 (-0.0120 ms). SQNR identical on all 12. c4 points 85/85; cand needs tk ≤ 0.8273
with its own tb (gap +0.0057 ms) — NOT crossed, but the closest any run has come today.
Pooled raw c4 (42 same-code anchors, mean 0.8425 sd 0.0057): cand 0.8330 ⇒ **-0.0095 ms (-1.128%), z=-1.68**
— the most negative single-run c4 pooled z of the day.
mnorm resid is above the -0.7% bar, but the pooled raw arbiter is strongly negative ⇒ SECOND PAIR queued.
- pair5 n8_c11_md_gm16 (touched c11): cand SID=146579, anchor SID=146580

### pair4 — n7_c12_md_gm16 (touched c12), cand 146575 / anchor 146576
machine term m=+1.207%; c12 d=+1.854%, resid **+1.395%** (+0.0218 ms); raw tk 1.5640→1.5930 (SLOWER).
Pooled raw c12 (43 anchors, mean 1.5764 sd 0.0121): +0.0166 ms (+1.052%), z=+1.37.
SQNR identical on all 12. c12 points 83/83; cand needs tk ≤ 1.5495 — NOT crossed.
Verdict: **NEGATIVE** — TMA md GROUP_M 16 is worse than 8 at c12; no second pair.
- pair6 n9_c8_md_gm16 (touched c8): cand SID=146585, anchor SID=146586

### pair5 — n8_c11_md_gm16 (touched c11), cand 146579 / anchor 146580
machine term m=-1.171%; c11 d=-0.615%, resid **-0.700%** (-0.0068 ms) — exactly on the promote bar;
raw tk 0.9760→0.9700. Pooled raw c11 (44 anchors, mean 0.9716 sd 0.0041): -0.0016 ms (-0.161%), z=-0.38.
Untouched residual spread this pair is wide (c9 -1.58 … c10 +1.38) ⇒ ±1.5% noise.
SQNR identical on all 12. c11 points 85/85; cand needs tk ≤ 0.9224 — NOT crossed.
resid ≤ -0.7% ⇒ SECOND PAIR queued.
- pair7 n2_c6_dn_gm16 (touched c6): cand SID=146589, anchor SID=146590

### pair6 — n9_c8_md_gm16 (touched c8), cand 146585 / anchor 146586
machine term m=+0.969%; c8 d=+1.119%, resid **+0.952%** (+0.0128 ms); raw tk 1.3410→1.3560 (SLOWER).
SQNR identical on all 12. c8 points 84/84; cand needs tk ≤ 1.3202 — NOT crossed.
Pooled raw c8 (45 anchors, mean 1.3470 sd 0.0133): +0.0090 ms (+0.665%), z=+0.68.
Verdict: **NEGATIVE** — `_g` md GROUP_M 16 is worse than 32 at c8 (GM8 was already neutral) ⇒ keep 32.
- pair8 n5_c8_dn_gm16 (touched c8): cand SID=146593, anchor SID=146594

### pair7 — n2_c6_dn_gm16 (touched c6), cand 146589 / anchor 146590
machine term m=-1.061%; c6 d=-2.148%, resid **-1.392%** (-0.0188 ms); raw tk 1.3500→1.3210.
Pooled raw c6 (13 v838 anchors, mean 1.3304 sd 0.0118): cand 1.3210 ⇒ **-0.0094 ms (-0.705%), z=-0.80**.
Caveat: this pair's anchor c6=1.3500 sits +1.7 sd above the v838 pool, and the untouched spread is wide
(c8 -2.01 … c4 +1.03).
SQNR identical on all 12. c6 points 85(cand)/84(anchor); cand needs tk ≤ 1.2357 — NOT crossed.
resid ≤ -0.7% ⇒ SECOND PAIR queued.
- pair9 n3_c6_dn_s3 (touched c6): cand SID=146598, anchor SID=146599

### pair8 — n5_c8_dn_gm16 (touched c8), cand 146593 / anchor 146594
machine term m=+1.256%; c8 d=+2.101%, resid **+1.885%** (+0.0251 ms); raw tk 1.3330→1.3610 (SLOWER).
Pooled raw c8 (47 anchors, mean 1.3472 sd 0.0136): +0.0138 ms (+1.022%), z=+1.02.
SQNR identical on all 12. c8 points 84/84; cand needs tk ≤ 1.2972 — NOT crossed.
Verdict: **NEGATIVE**. With pair2 (GM8, +1.79%) this closes c8 dn GROUP_M: 32 beats both 8 and 16.
- pair10 n6_c4_md_reg232 (2nd pair, touched c4): cand SID=146602, anchor SID=146603

### pair9 — n3_c6_dn_s3 (touched c6), cand 146598 / anchor 146599
machine term m=-1.158%; c6 d=-2.739%, resid **-1.913%** (-0.0258 ms); raw tk 1.3510→1.3140.
Pooled raw c6 (15 v838 anchors, mean 1.3315 sd 0.0122): cand 1.3140 ⇒ **-0.0175 ms (-1.312%), z=-1.43**;
1.3140 is below *every* v838 anchor observed (min 1.319).
SQNR identical on all 12. c6 points 85(cand)/84(anchor); cand needs tk ≤ 1.2455 — NOT crossed.
resid ≤ -0.7% ⇒ SECOND PAIR queued. Strongest candidate so far.

> Note on the c6 anchor pool: the 15 v838 anchors are bimodal — 11 runs in 1.319-1.328 and
> 4 runs at 1.350-1.351 (146572/146580/146590/146599), i.e. two machine populations.
> Pooled-raw z on c6 must be read with that in mind; mnorm's per-case machine term is the
> better corrector here, and both statistics agree in sign for n2 and n3.
- pair11 n3_c6_dn_s3 (2nd pair, touched c6): cand SID=146606, anchor SID=146607

### pair10 — n6_c4_md_reg232 (2nd pair, touched c4), cand 146602 / anchor 146603
machine term m=+1.448%; c4 d=+1.681%, resid **+0.269%** (+0.0022 ms); raw tk 0.8330→0.8470 (SLOWER).
SQNR identical on all 12. c4 points 85(cand)/86(anchor, luckier tb); cand needs tk ≤ 0.8131 — NOT crossed.
Two pairs disagree in sign (-0.198%, +0.269%), mean +0.036%.
Pooled raw c4 (49 anchors, mean 0.8419 sd 0.0058): n6 n=2 mean 0.8400 ⇒ -0.0019 ms (-0.230%), z=-0.33, 1/2 below mean.
Verdict: **NEUTRAL** — pair3's -1.1% pooled z was a single-run fluctuation; `maxnreg=232` is free but not a win at c4.
- pair12 n2_c6_dn_gm16 (2nd pair, touched c6): cand SID=146611, anchor SID=146612

### pair11 — n3_c6_dn_s3 (2nd pair, touched c6), cand 146606 / anchor 146607
machine term m=-1.079%; c6 d=-2.376%, resid **-1.606%** (-0.0216 ms); raw tk 1.3470→1.3150.
SQNR identical on all 12. c6 points 85/85; cand needs tk ≤ 1.2499 — NOT crossed.
Two pairs, both negative: -1.913%, -1.606% ⇒ mean **-1.760%** ≤ -0.7%.

## Machine-stratified pooled estimator (new, sharper than raw pooling)
The c6 anchor pool is bimodal because two machine populations are in it, and the split shows up in the
UNTOUCHED cases as well (c1 ≈ 4.74 ⟺ c6 ≈ 1.350; c1 ≈ 4.60-4.66 ⟺ c6 ≈ 1.319-1.328).
Estimator (`strat.py`): for each run compute a machine index M = mean over the 11 untouched cases of
tk_i / (anchor-pool mean of case i); fit the anchors' touched-case tk against M by OLS; report each
candidate's residual off that line, with z against the anchors' residual sd.
For c6 this index explains **91%** of the anchor variance — residual sd drops 0.0121 → **0.0037 ms (0.28%)**.

| c6 candidate | SID | M | tk | predicted | residual | z |
|---|---|---|---|---|---|---|
| n1 gm4  | 146563 | 1.0113 | 1.3510 | 1.3495 | +0.0015 ms (+0.108%) | +0.39 |
| n2 gm16 | 146589 | 0.9981 | 1.3210 | 1.3291 | **-0.0081 ms (-0.607%)** | -2.18 |
| n3 s3   | 146598 | 0.9946 | 1.3140 | 1.3237 | **-0.0097 ms (-0.734%)** | -2.63 |
| n3 s3   | 146606 | 0.9974 | 1.3150 | 1.3280 | **-0.0130 ms (-0.978%)** | -3.51 |

n3 mean residual **-0.0114 ms (-0.852%)**, z_of_mean **-4.35** (n=2). Both n3 runs are below *every*
observed v838 anchor. Correction to pair1: n1's naive pooled +2.1% was pure machine — stratified it is
+0.108% (z=+0.39), i.e. **NEUTRAL, not negative**.
- pair13 n3_c6_dn_s3 (3rd pair, touched c6): cand SID=146613, anchor SID=146615

### pair12 — n2_c6_dn_gm16 (2nd pair, touched c6), cand 146611 / anchor 146612
machine term m=-1.281%; c6 d=-2.219%, resid **-1.305%** (-0.0176 ms); raw tk 1.3520→1.3220.
SQNR identical on all 12. c6 points 85(cand)/84(anchor); cand needs tk ≤ 1.2473 — NOT crossed.
Two pairs, both negative: -1.392%, -1.305% ⇒ mean **-1.349%**.
Stratified (18 c6 anchors, 92% of variance explained, resid sd 0.0036 ms):
146589 -0.608% z=-2.25, 146611 -0.146% z=-0.54 ⇒ mean **-0.376%**, z_of_mean -1.97.

## Why mnorm over-states on c6/c8/c11/c12
`scripts/mnorm.py`'s machine signature comes from two old same-code pairs (SIDs 138154/138157,
138164/138166) and gives s = c6 +0.71, c8 +0.17, c11 -0.07, c12 +0.38 — i.e. it believes those cases are
nearly machine-insensitive. Today's 50-anchor regression says the opposite: the OLS slope of tk against
the untouched-case machine index is **c6 1.55, c8 1.54, c12 1.38, c11 0.43** (variance explained 92/88/94/82%).
So mnorm under-subtracts the machine term on exactly those four cases and the leftover machine effect lands
in the "residual". Symptom: c8's mnorm residual tracks the machine term almost 1:1 in every pair
(m +1.335→resid +1.792, m +0.969→+0.952, m +1.256→+1.885, m -1.253→-2.343, m -1.171→-1.125,
m -1.158→-1.640, m -1.079→-1.290). For c6/c8/c11/c12 the stratified estimator is the arbiter.
- pair14 n8_c11_md_gm16 (2nd pair, touched c11): cand SID=146617, anchor SID=146618

### pair13 — n3_c6_dn_s3 (3rd pair, touched c6), cand 146613 / anchor 146615
machine term m=+1.042%; c6 d=+0.600%, resid **-0.143%** (-0.0019 ms); raw tk 1.3340→1.3420.
(Both runs of this pair landed on a third, intermediate machine mode — c6 ≈ 1.334/1.342.)
SQNR identical on all 12. c6 points 85/85; cand needs tk ≤ 1.2388 — NOT crossed.
Three pairs, all negative: -1.913%, -1.606%, -0.143% ⇒ mnorm mean **-1.221%**.

### n3_c6_dn_s3 — **PROMOTE-CANDIDATE**
Stratified (19 c6 anchors, 90% of variance explained, resid sd 0.0038 ms = 0.286%):

| run | M | tk | predicted | residual | z |
|---|---|---|---|---|---|
| 146598 | 0.9941 | 1.3140 | 1.3242 | -0.0102 ms (-0.771%) | -2.68 |
| 146606 | 0.9969 | 1.3150 | 1.3284 | -0.0134 ms (-1.011%) | -3.52 |
| 146613 | 1.0105 | 1.3420 | 1.3493 | -0.0073 ms (-0.541%) | -1.91 |

Mean residual **-0.0103 ms (-0.773%)**, z_of_mean **-4.68**, 3/3 same sign.
mnorm 3 pairs all negative (mean -1.221%). SQNR equal to the anchor on all 12 cases in all 3 runs;
every run scored 100 on all 12. Not crossed to 86 yet (c6 needs ≈ 1.24-1.25 ms), but at 9.5 points/ms the
-0.0103 ms is worth ≈ 0.10 expected points and it stacks on top of the v838 c6 GROUP_M 8 promotion.
Change: `_dn_tma2_f8_host` `num_stages = 3 if (K == 1024 and N == 3584) else (3 if K == 14336 else 4)`.
File `experiments/2026-09-20/candidates/n3_c6_dn_s3.py`, sha12 843edd1f0c13.

### Estimator calibration (leave-one-out over the anchor pools)
Treating each anchor as a pseudo-candidate (refitting without it) gives z ~ N(0, ~1.1):
c6 mean -0.01 sd 1.15 (range -2.32…+1.42, n=18) | c8 -0.00/1.06 | c4 -0.00/1.06 | c11 -0.01/1.10 | c12 +0.01/1.06.
n3's three runs (-2.68, -3.52, -1.91) sit at or beyond the extreme of that null distribution.
- pair15 n4_c8_dn_gm8 (2nd pair, touched c8): cand SID=146620, anchor SID=146621

### pair14 — n8_c11_md_gm16 (2nd pair, touched c11), cand 146617 / anchor 146618
machine term m=+1.132%-class; c11 d=+0.412%, resid **+0.510%** (+0.0050 ms); raw tk 0.9710→0.9750.
SQNR identical on all 12. c11 points 85/85; cand needs tk ≤ 0.9092 — NOT crossed.
Two pairs disagree in sign (-0.700%, +0.510%), mean -0.095%.
Stratified (53 c11 anchors, 82% of variance explained, resid sd 0.0017 ms): 146579 +0.117% z=+0.65,
146617 -0.098% z=-0.55 ⇒ mean **+0.009%**, z_of_mean +0.07.
Verdict: **NEUTRAL** — TMA md GROUP_M 16 is free but worthless at c11.
- pair16 n3_c6_dn_s3 (4th pair, confirmation, touched c6): cand SID=146622, anchor SID=146623

### pair15 — n4_c8_dn_gm8 (2nd pair, touched c8), cand 146620 / anchor 146621
c8 d=+2.465%, resid **+2.264%** (+0.0303 ms); raw tk 1.3380→1.3710.
SQNR identical on all 12. c8 points 83(cand)/84(anchor); cand needs tk ≤ 1.3417 — NOT crossed.
Stratified (54 c8 anchors, 89% of variance explained, resid sd 0.0044 ms):
146567 -0.385% z=-1.19, 146620 +0.671% z=+2.07 ⇒ mean **+0.144%**, z_of_mean +0.62, signs disagree.
Verdict: **NEUTRAL** (mnorm reads it as clearly negative both times, but c8's mnorm residual is the
contaminated one — see the mnorm note above; the stratified arbiter says no effect either way).
- pair17 n3_c6_dn_s3 (5th pair, confirmation, touched c6): cand SID=146624, anchor SID=146625

### pair16 — n3_c6_dn_s3 (4th pair, touched c6), cand 146622 / anchor 146623
machine term m ≈ -1.3%-class; c6 d=-2.232%, resid **-1.269%** (-0.0171 ms); raw tk 1.3440→1.3140.
SQNR identical on all 12. c6 points 85(cand)/84(anchor); cand needs tk ≤ 1.2427 — NOT crossed.
Four pairs, all negative: -1.913 / -1.606 / -0.143 / -1.269 % ⇒ mnorm mean **-1.233%**.
Stratified (22 c6 anchors, 89% explained, resid sd 0.0041 ms): 146598 -0.735% z=-2.38,
146606 -0.965% z=-3.13, 146613 -0.448% z=-1.47, 146622 -0.643% z=-2.08 ⇒
mean **-0.0093 ms (-0.695%)**, z_of_mean **-4.53**, 4/4 same sign.

### pair17 — n3_c6_dn_s3 (5th pair, touched c6), cand 146624 / anchor 146625
c6 d=+1.737%, resid **+0.794%** (+0.0105 ms); raw tk 1.3240→1.3470. SQNR identical on all 12.
c6 points 85/85; cand needs tk ≤ 1.2514 — NOT crossed. (First pair of the five with a positive mnorm residual;
stratified for this run is -0.014% z=-0.05, i.e. exactly on the machine line.)

- pair18 n3_c6_dn_s3 (6th pair, touched c6): cand SID=146627, anchor SID=146628
- pair19 n3_c6_dn_s3 (7th pair, touched c6): cand SID=146630, anchor SID=146631

## FINAL — n3_c6_dn_s3, five completed pairs
mnorm residual per pair: -1.913 / -1.606 / -0.143 / -1.269 / +0.794 % ⇒ mean **-0.827%** (4/5 negative).
Stratified (23 c6 anchors, 89% of variance explained, resid sd 0.0040 ms = 0.302%):

| run | M | tk | predicted | residual | z |
|---|---|---|---|---|---|
| 146598 | 0.9943 | 1.3140 | 1.3239 | -0.0099 ms (-0.747%) | -2.46 |
| 146606 | 0.9970 | 1.3150 | 1.3279 | -0.0129 ms (-0.975%) | -3.21 |
| 146613 | 1.0107 | 1.3420 | 1.3480 | -0.0060 ms (-0.445%) | -1.49 |
| 146622 | 0.9935 | 1.3140 | 1.3227 | -0.0087 ms (-0.656%) | -2.16 |
| 146624 | 1.0101 | 1.3470 | 1.3472 | -0.0002 ms (-0.014%) | -0.05 |

Mean **-0.0075 ms (-0.566%)**, z_of_mean **-4.19**, 5/5 the same sign.
SQNR equal to the anchor on all 12 cases in every run; every run scored 100 on all 12.
**Verdict: PROMOTE-CANDIDATE** (real effect ≈ -0.006…-0.010 ms on c6, ≈ 0.06-0.10 expected points at 9.5 pt/ms).

## Summary table

| cand | change + gate | sha12 | pair SIDs | touched resid per pair (%) | stratified Δ% (z_of_mean, n) | crossed? | SQNR same | verdict |
|---|---|---|---|---|---|---|---|---|
| n1_c6_dn_gm4 | dn GROUP_M 8→4, K==1024 and N==3584 | 2101a3dcfcbd | 146563/146564 | +1.323 | +0.091 (+0.32, 1) | no | yes | NEUTRAL |
| n2_c6_dn_gm16 | dn GROUP_M 8→16, same gate | 06586c010046 | 146589/146590, 146611/146612 | -1.392, -1.305 | **-0.410 (-2.03, 2)** | no | yes | WEAK-POSITIVE (mnorm clears the bar; stratified size only -0.41%) |
| n3_c6_dn_s3 | dn num_stages 4→3, same gate | 843edd1f0c13 | 7 pairs: 146598/599, 146606/607, 146613/615, 146622/623, 146624/625, 146627/628, 146630/631 | -1.913, -1.606, -0.143, -1.269, +0.794, +0.214, -2.030 (mean -0.850) | **-0.607 (-5.02, 7)** | no | yes | **PROMOTE-CANDIDATE** |
| n4_c8_dn_gm8 | dn GROUP_M 32→8, K==1024 and N==4096 | cf4b28dff1ee | 146567/146568, 146620/146621 | +1.792, +2.264 | +0.144 (+0.62, 2) | no | yes | NEUTRAL |
| n5_c8_dn_gm16 | dn GROUP_M 32→16, same gate | 8eb3011fafbb | 146593/146594 | +1.885 | -0.005 (-0.02, 1) | no | yes | NEUTRAL |
| n6_c4_md_reg232 | `_g` md maxnreg=232 (s3 kept), G==32 and I==1024 | 2a1ffe8c0e07 | 146571/146572, 146602/146603 | -0.198, +0.269 | -0.296 (-0.64, 2) | no | yes | NEUTRAL |
| n7_c12_md_gm16 | TMA md GROUP_M 8→16, K==1024 and I==2048 | 342c8d3cd19e | 146575/146576 | +1.395 | +0.194 (+1.05, 1) | no | yes | NEUTRAL→NEGATIVE |
| n8_c11_md_gm16 | TMA md GROUP_M 8→16, K==1024 and I==1024 | 8ed843433bf8 | 146579/146580, 146617/146618 | -0.700, +0.510 | +0.009 (+0.07, 2) | no | yes | NEUTRAL |
| n9_c8_md_gm16 | `_g` md GROUP_M 32→16, `_gg` (G==96 and I==1024) | 141db2ed8436 | 146585/146586 | +0.952 | -0.176 (-0.53, 1) | no | yes | NEUTRAL |

No candidate crossed its next integer point in any run. SQNR matched the anchor on all 12 cases in every
one of the 18 candidate runs, and every run scored 100 on all 12 cases.

### pair18 — n3_c6_dn_s3 (6th pair, touched c6), cand 146627 / anchor 146628
c6 d=+1.135%, resid **+0.214%** (+0.0028 ms); raw tk 1.3210→1.3360. SQNR identical on all 12.
c6 points 85/85; cand needs tk ≤ 1.2961 — NOT crossed.

## FINAL (6 completed pairs) — n3_c6_dn_s3 = PROMOTE-CANDIDATE
mnorm residual per pair: -1.913 / -1.606 / -0.143 / -1.269 / +0.794 / +0.214 % ⇒ mean **-0.654%**, 4/6 negative.
Stratified (24 c6 anchors, 89% of variance explained, resid sd 0.0039 ms = 0.296%):
-0.743 / -0.971 / -0.446 / -0.651 / -0.015 / -0.720 % ⇒ mean **-0.0079 ms (-0.591%)**,
z_of_mean **-4.89**, **6/6 the same sign**.
SQNR equal to the anchor on all 12 cases in all 6 runs; every run scored 100 on all 12.
The two estimators agree in direction on every one of the 6 runs of this candidate; the stratified number
(-0.59%, ≈ -0.008 ms) is the one to believe, since it removes 89% of the machine variance whereas
mnorm's fixed signature under-corrects c6 (slope 1.48 measured vs 0.71 assumed).

> Tooling: the machine-stratified estimator is saved as `experiments/2026-09-20/strat.py`
> (`python3 experiments/2026-09-20/strat.py <touched_case> <cand_sids,comma>`); it reads a local tk cache,
> so re-point the `H` paths or re-fetch with `scripts/cases.py` before reuse.

### pair19 — n3_c6_dn_s3 (7th pair, touched c6), cand 146630 / anchor 146631
c6 d=-2.878%, resid **-2.030%** (-0.0275 ms); raw tk 1.3550→1.3160. SQNR identical on all 12.
c6 points 85(cand)/84(anchor); cand needs tk ≤ 1.2414 — NOT crossed.

## FINAL (7 completed pairs) — n3_c6_dn_s3 = **PROMOTE-CANDIDATE**
mnorm residual per pair: -1.913 / -1.606 / -0.143 / -1.269 / +0.794 / +0.214 / -2.030 % ⇒ mean **-0.850%**, 5/7 negative.
Stratified (25 c6 anchors, 88% of variance explained, resid sd 0.0043 ms = 0.320%):
-0.745 / -0.986 / -0.520 / -0.650 / -0.087 / -0.786 / -0.479 % ⇒
mean **-0.0081 ms (-0.607%)**, z_of_mean **-5.02**, **7/7 the same sign**.
SQNR equal to the anchor on all 12 cases in all 7 runs; every run scored 100 on all 12.

### Extension (beyond the 9-candidate brief) — n10_c6_dn_gm16_s3
Both single c6 dn knobs that survived (n2 GROUP_M 16 at -0.41%, n3 num_stages 3 at -0.61%) act on the
same kernel launch, so their combination is the obvious next test.
| n10_c6_dn_gm16_s3 | dn GROUP_M 8→16 **and** num_stages 4→3, gate K==1024 and N==3584 (c6) | sha12 e7f940bed4f1 |
`diff p1/kernel.py` shows exactly the two gated lines (4591 and 4593).
- pair20 n10_c6_dn_gm16_s3 (touched c6): cand SID=146632, anchor SID=146633
- pair21 n10_c6_dn_gm16_s3 (2nd pair, touched c6): cand SID=146635, anchor SID=146636

### pair20 — n10_c6_dn_gm16_s3 (touched c6), cand 146632 / anchor 146633
c6 d=-2.370%, resid **-1.485%** (-0.0201 ms); raw tk 1.3500→1.3180. SQNR identical on all 12.
c6 points 85(cand)/84(anchor); cand needs tk ≤ 1.2317 — NOT crossed.
Stratified (26 c6 anchors, 89% explained, resid sd 0.0042 ms): **-0.484%, z=-1.53** (1 run).
Read: the two knobs do **not** stack — the combination (-0.48%) is no better than n3 alone (-0.61% over 7 runs),
consistent with n2's GROUP_M 16 effect overlapping the same pipeline-drain mechanism that num_stages 3 addresses.
Second pair queued to confirm.

### pair21 — n10_c6_dn_gm16_s3 (2nd pair, touched c6), cand 146635 / anchor 146636
c6 d=-2.367%, resid **-1.439%** (-0.0195 ms); raw tk 1.3520→1.3200. SQNR identical on all 12.
c6 points 85(cand)/84(anchor); cand needs tk ≤ 1.2457 — NOT crossed.
Stratified over 2 runs: -0.484% / -0.422% ⇒ mean **-0.450%**, z_of_mean -2.05.
**Conclusion: the combination does NOT stack.** n3 alone is -0.607% (z -5.16, n=7); adding GROUP_M 16 on top
makes it *worse*, not better. Recommend promoting n3 (`num_stages` 3) alone and leaving c6 dn GROUP_M at 8.

### Second extension — `num_stages` 4→3 on the two dn cases where it was never tried
`dn num_stages 3` is the best knob found today, and it has now been measured on c1/c3/c5/c6/c7/c11/c12
(only c6 positive) but **never on c4 or c8**. Built and queued:
| n11_c4_dn_s3 | dn num_stages 4→3, gate K==1024 and N==2048 (c4) | sha12 c8d3b691ef84 |
| n12_c8_dn_s3 | dn num_stages 4→3, gate K==1024 and N==4096 (c8) | sha12 8b0f82874999 |
Each `diff p1/kernel.py <cand>` shows only line 4593.
- pair22 n11_c4_dn_s3 (touched c4): cand SID=146637, anchor SID=146638
- pair23 n12_c8_dn_s3 (touched c8): cand SID=146640, anchor SID=146641
- pair24 n11_c4_dn_s3 (2nd pair, touched c4): cand SID=146644, anchor SID=146645
- pair25 n11_c4_dn_s3 (3rd pair, touched c4): cand SID=146646, anchor SID=146648
- pair26 n11_c4_dn_s3 (4th pair, touched c4): cand SID=146649, anchor SID=146650  [IN FLIGHT]

### pair22 — n11_c4_dn_s3 (touched c4), cand 146637 / anchor 146638
c4 d=+0.480%, resid **-0.892%** (-0.0074 ms); raw tk 0.8330→0.8370.
Stratified (c4 anchors, resid sd ≈ 0.0055 ms): **-0.857%, z=-1.33**.
SQNR identical on all 12. c4 points 85/85; cand needs tk ≤ 0.7993 — NOT crossed.
Both estimators agree and clear the -0.7% bar on the first pair ⇒ SECOND PAIR needed.
c4 is the highest-value case on the board (14.5 points/ms) — this is the lead worth pushing next.

### pair23 — n12_c8_dn_s3 (touched c8), cand 146640 / anchor 146641
c8 d=+0.970%, resid **+0.778%** (+0.0104 ms); raw tk 1.3410→1.3540.
Stratified: **-0.322%, z=-1.00** (the two estimators disagree in sign, as they routinely do on c8).
SQNR identical on all 12. c8 points 83(cand)/84(anchor); cand needs tk ≤ 1.3490 — NOT crossed
(closest miss of the day on c8: +0.0050 ms).
Inconclusive on one pair; c8's mnorm residual is the contaminated one, so the -0.32% stratified reading
is the better estimate. Worth a second pair, below n11 in priority.



### pair24 — n11_c4_dn_s3 (2nd pair, touched c4), cand 146644 / anchor 146645
c4 d=+0.000% (raw tk 0.8430 → 0.8430, bit-identical), resid **-1.105%** (-0.0093 ms).
Stratified: **-0.135%, z=-0.21**.
SQNR identical on all 12. c4 points 85/85; needs tk ≤ 0.8149 — NOT crossed.
Two pairs, mnorm both negative: -0.892%, -1.105% ⇒ mean **-0.999%** ≤ -0.7%.
Stratified (63 c4 anchors, resid sd 0.0054 ms = 0.642%): -0.857% / -0.135% ⇒ mean **-0.497%**, z_of_mean -1.10.
Note c4's machine index only explains 14% of its variance (c4 is nearly machine-insensitive but intrinsically
noisy at 0.64% per run), so stratification buys little here and more pairs are the only way to sharpen it.
Status: meets the letter of PROMOTE-CANDIDATE on mnorm with the stratified arbiter agreeing in direction,
but the stratified z is weak ⇒ **3rd pair IN FLIGHT** before calling it.

### pair25 — n11_c4_dn_s3 (3rd pair, touched c4), cand 146646 / anchor 146648
c4 raw tk 0.8360 → 0.8360 (identical); mnorm resid **+1.316%**, stratified **-0.520% z=-0.80**.
SQNR identical on all 12. c4 points 87(cand, lucky tb=5.788)/85(anchor); cand needs tk ≤ 0.7893 — NOT crossed.
Three pairs — mnorm: -0.892 / -1.105 / +1.316 ⇒ mean -0.227%, signs disagree.
Stratified (64 c4 anchors, resid sd 0.0055 ms): -0.824 / -0.102 / -0.520 ⇒ mean **-0.483%**, z_of_mean -1.29,
3/3 same sign.
Status: **INCONCLUSIVE-POSITIVE** — every stratified run is negative but c4's per-run noise (0.65%) means
≈8-10 pairs are needed to resolve a -0.5% effect. 4th pair IN FLIGHT.

## Open queue at hand-off
1. **n3_c6_dn_s3** — **PROMOTE-CANDIDATE**, 7 pairs, stratified -0.607% (z_of_mean -5.02…-5.16), 7/7 same sign,
   mnorm mean -0.850%. Recommend folding into production as v839.
   (`experiments/2026-09-20/candidates/n3_c6_dn_s3.py`, sha12 843edd1f0c13.)
2. **n11_c4_dn_s3** — 3 pairs, stratified -0.483% with 3/3 same sign but z only -1.29. 4th pair in flight
   (146649/146650). c4 is 14.5 points/ms, so -0.005 ms ≈ 0.07 points; needs ~8-10 pairs to call.
3. **n12_c8_dn_s3** — 1 pair, stratified -0.32%, needs more pairs.
4. **n2_c6_dn_gm16** — stratified -0.41% (z -2.03, n=2); do NOT combine with n3 (n10 showed no stacking).
