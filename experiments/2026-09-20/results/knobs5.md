# 2026-09-20 knobs5 — geometry-gated launch-kwarg candidates (m1…m7)

Anchor = unmodified `p1/kernel.py` v837 (sha256 `81d994bea8e0a22e8cd95d7b5dfe7a519a95968d822be54bcddaa7d0423f7f91`).
Case key = (T, H, E, I, topk) from `_KNOWN12`. dn host `_dn_tma2_f8_host` sees G=E, K=I, N=H.
Every candidate is a **one-line** diff vs `p1/kernel.py` (`diff` verified, single hunk, launch kwargs only).

| cand | change | gate (as written) | touched | sha12 |
|---|---|---|---|---|
| m1_c5_dn_gm8 | dn `GROUP_M` 32→8 | `K == 2560 and N == 3584` (c5) | c5 | fb570cdbe7bb |
| m2_c5_dn_s3 | dn `num_stages` 4→3 | `K == 2560 and N == 3584` (c5) | c5 | 7d93e7dfeb4a |
| m3_c3_dn_s3 | dn `num_stages` 4→3 | `K == 2048 and N == 2048` (c3) | c3 | d2651cd45a6b |
| m4_c7_dn_s3 | dn `num_stages` 4→3 | `K == 2048 and N == 4096 and G == 96` (c7, excludes c9) | c7 | f949e8d6f9b8 |
| m5_c2_md_gm8 | q8 md `GROUP_M` 16→8 | `I == 14336` inside the `E == 8` branch (c2) | c2 | 181183a3986f |
| m6_c12_md_reg255 | q8 md `maxnreg` 232→255 | whole `E == 8` call site (c1+c2) | c1, c2 | d26d2b5f8d8c |
| m7_c7_md_s4 | `_g` md `num_stages` 3→4, **no** maxnreg | `_gg or (G == 96 and I == 2048)` (c8 keeps s4+reg232, c7 gains s4 only) | c7 | 69ef58c0f96d |

Anchor SQNR reference: c1 23.12, c2 23.14, c3 23.19, c5 23.13, c7 23.12.

## SID log

- pair1 m1_c5_dn_gm8: cand SID=146478, anchor SID=146481

### pair1 — m1_c5_dn_gm8 (touched c5), cand 146478 / anchor 146481
machine term m=+0.346%; c5 d=+0.570%, resid **-0.034%** (-0.0010 ms equivalent); raw tk 2.8250 vs anchor 2.8090.
Pooled v837 anchor c5 (n=9): mean 2.8337 sd 0.0291 → cand z=**-0.30** (-0.31% vs pool mean).
SQNR identical on all 12; all 12 scores 100/100. Verdict: **NEUTRAL** (|resid| < 0.7% ⇒ no second pair).

- pair2 m2_c5_dn_s3: cand SID=146487, anchor SID=146488

### pair2 — m2_c5_dn_s3 (touched c5), cand 146487 / anchor 146488
machine term m=-1.151% (anchor landed on a slow machine; untouched resid spread -1.85…+1.72 ⇒ noisy fit);
c5 d=-2.910%, resid **-0.905%** (-0.0261 ms); raw tk 2.8030 vs anchor 2.8870.
Pooled v837 anchor c5 (n=9): mean 2.8423 sd 0.0320 → cand z=**-1.23** (-1.38%); 2.8030 is below every
anchor run seen (min 2.8090). SQNR identical on all 12; scores 100×12.
Verdict: provisional (resid ≤ -0.7%) ⇒ **second pair required**.

- pair3 m3_c3_dn_s3: cand SID=146493, anchor SID=146495

### pair3 — m3_c3_dn_s3 (touched c3), cand 146493 / anchor 146495
machine term m=-1.287% (anchor on a slow machine again); c3 d=-2.032%, resid **+0.048%** (+0.0007 ms);
raw tk 1.3980 vs anchor 1.4270.
Pooled v837 anchor c3 (n=9): mean 1.4076 sd 0.0129 → cand z=**-0.74** (-0.68%).
SQNR identical on all 12; scores 100×12. Verdict: **NEUTRAL** (resid ≈ 0; the raw gain is the machine, not the knob).

- pair4 m4_c7_dn_s3: cand SID=146501, anchor SID=146502

### pair4 — m4_c7_dn_s3 (touched c7), cand 146501 / anchor 146502
machine term m=-0.029% (tightest pair so far: untouched resid spread -1.22…+0.98); c7 d=-0.219%,
resid **-0.182%** (-0.0042 ms); raw tk 2.2800 vs anchor 2.2850.
Pooled v837 anchor c7 (n=9): mean 2.2953 sd 0.0162 → cand z=**-0.95** (-0.67%), but the same-run anchor
was itself 2.2850 (-0.45% vs pool) ⇒ most of that is machine, not knob.
SQNR identical on all 12; scores 100×12. Verdict: **NEUTRAL** (resid > -0.7% ⇒ no second pair).

- pair5 m5_c2_md_gm8: cand SID=146507, anchor SID=146508

### pair5 — m5_c2_md_gm8 (touched c2), cand 146507 / anchor 146508
machine term m=-0.111% (tight pair, untouched resid spread -0.49…+0.64); c2 d=+1.030%,
resid **+1.281%** (+0.1032 ms); raw tk 8.1410 vs anchor 8.0580.
Pooled v837 anchor c2 (n=9): mean 7.9380 sd 0.1024 → cand z=**+1.98** (+2.56%).
SQNR identical on all 12; scores 100×12. Verdict: **NEGATIVE** — q8 md GROUP_M=16 beats 8 at c2; keep 16.

- pair6 m6_c12_md_reg255: cand SID=146516, anchor SID=146519

### pair6 — m6_c12_md_reg255 (touched c1+c2), cand 146516 / anchor 146519
maxnreg=255 compiles and runs (no TLE, no SQNR change). machine term m=+0.366%;
c1 d=-0.152% resid **-0.939%** (-0.0433 ms); c2 d=-0.356% resid **-1.184%** (-0.0930 ms).
raw tk c1 4.6080 vs 4.6150, c2 7.8270 vs 7.8550.
Pooled v837 anchors: c1 n=9 mean 4.6520 sd 0.0556 → z=**-0.79** (-0.95%); c2 n=9 mean 7.9154 sd 0.0957 → z=**-0.92** (-1.12%).
(c10 resid +1.76% is the one outlier inflating m.) SQNR identical on all 12; scores 100×12.
Verdict: provisional (both touched resid ≤ -0.7%) ⇒ **second pair required**.

- pair7 m7_c7_md_s4: cand SID=146524, anchor SID=146525

### pair7 — m7_c7_md_s4 (touched c7), cand 146524 / anchor 146525
machine term m=-1.303% (anchor on a slow machine); c7 d=-0.344%, resid **+1.353%** (+0.0314 ms);
raw tk 2.3190 vs anchor 2.3270 — c7 is the ONLY case that failed to pick up the machine-wide speedup.
Pooled v837 anchor c7 (n=9): mean 2.3000 sd 0.0184 → cand z=**+1.03** (+0.83%).
SQNR identical on all 12; scores 100×12.
Verdict: **NEGATIVE** — `_g` md s3→s4 without maxnreg costs c7; keep s3 (so v838's c7 gain, if real, came from maxnreg=232, not s4).

- pair8 m6_c12_md_reg255 (2nd): cand SID=146531, anchor SID=146532

### pair8 — m6_c12_md_reg255 (2nd pair, touched c1+c2), cand 146531 / anchor 146532
machine term m=-1.207%; c1 d=-1.691% resid **+0.903%** (+0.0427 ms); c2 d=-2.683% resid **+0.044%** (+0.0035 ms).
raw tk c1 4.6510 vs 4.7310, c2 7.8710 vs 8.0880.
Pooled v837 anchors: c1 n=9 mean 4.6649 sd 0.0589 → z=**-0.24** (-0.30%); c2 n=9 mean 7.9413 sd 0.1067 → z=**-0.66** (-0.89%).
SQNR identical on all 12; scores 100×12.
Pair8 **disagrees in sign with pair6** on c1 (+0.90 vs -0.94) and is ~0 on c2 (+0.04 vs -1.18) ⇒ **third pair required**.

- pair9 m2_c5_dn_s3 (2nd): cand SID=146535, anchor SID=146536

### pair9 — m2_c5_dn_s3 (2nd pair, touched c5), cand 146535 / anchor 146536
machine term m=-1.179%; c5 d=-2.632%, resid **-0.579%** (-0.0167 ms); raw tk 2.8110 vs anchor 2.8870.
Pooled v837 anchor c5 (n=9): mean 2.8423 sd 0.0320 → cand z=**-0.98** (-1.10%).
SQNR identical on all 12; scores 100×12.
Same sign as pair2 (-0.905%); two-pair mean = **-0.742%** — right on the -0.7% promotion line ⇒ third pair for confirmation.

- pair10 m6_c12_md_reg255 (3rd): cand SID=146541, anchor SID=146542

### pair10 — m6_c12_md_reg255 (3rd pair, touched c1+c2), cand 146541 / anchor 146542
machine term m=+1.590% (candidate on a slow machine); c1 d=+2.969% resid **-0.447%** (-0.0206 ms);
c2 d=+3.578% resid **-0.014%** (-0.0011 ms). raw tk c1 4.7510 vs 4.6140, c2 8.1050 vs 7.8250
(pooled z +1.78 / +1.96 — pure machine, the whole board moved +1.6…+3.6%).
SQNR identical on all 12; scores 100×12.

**m6 three-pair summary (residual %):** c1 -0.939 / +0.903 / -0.447 → mean **-0.161%**;
c2 -1.184 / +0.044 / -0.014 → mean **-0.406%**. Signs disagree, means > -0.7% ⇒ **NEUTRAL**
(maxnreg 232→255 is safe and free at c1/c2 but not a measurable win).

- pair11 m2_c5_dn_s3 (3rd): cand SID=146547, anchor SID=146548

### pair11 — m2_c5_dn_s3 (3rd pair, touched c5), cand 146547 / anchor 146548
machine term m=-1.278%; c5 d=-2.358%, resid **-0.132%** (-0.0038 ms); raw tk 2.8160 vs anchor 2.8840.
Pooled v837 anchor c5 (n=9): mean 2.8420 sd 0.0315 → cand z=**-0.83** (-0.91%).
SQNR identical on all 12; scores 100×12.
m2 three-pair residuals: -0.905 / -0.579 / -0.132 → mean **-0.539%**. All three the same sign but the
mean sits above the -0.7% promotion line ⇒ fourth pair run to settle it.

- pair12 m2_c5_dn_s3 (4th): cand SID=146554, anchor SID=146555

### pair12 — m2_c5_dn_s3 (4th pair, touched c5), cand 146554 / anchor 146555
machine term m=+1.294% (candidate on a slow machine); c5 d=+1.625%, resid **-0.628%** (-0.0178 ms);
raw tk 2.8760 vs anchor 2.8300. SQNR identical on all 12; scores 100×12.

**m2 four-pair summary (c5 residual %):** -0.905 / -0.579 / -0.132 / -0.628 → mean **-0.561%**,
sd 0.320, SE 0.160, t(3) = -3.5 (p≈0.02) — 4/4 the same sign, so the effect is real but its
size is below the -0.7% promotion bar. Verdict: **NEUTRAL (real but sub-threshold, ≈-0.016 ms on c5)**.

---

## Pooled same-code raw comparison (all v837 anchor runs, n=20)

v837 anchor pool raw tk (ms): c1 4.6763±0.0668 (4.5960…4.7700) | c2 7.9642±0.1201 (7.8150…8.1130) |
c3 1.4101±0.0132 (1.3900…1.4300) | c5 2.8479±0.0347 (2.8050…2.8910) | c7 2.3064±0.0226 (2.2820…2.3430).
(SIDs 146437/440/444/449/454/459/464/471 + 146481/488/495/502/508/519/525/532/536/542/548/555.)

| cand | touched | cand raw runs (ms) | mean | % vs pool | per-run z | mean z |
|---|---|---|---|---|---|---|
| m1_c5_dn_gm8 | c5 | 2.8250 | 2.8250 | -0.81% | -0.66 | -0.66 |
| m2_c5_dn_s3 | c5 | 2.8030 / 2.8110 / 2.8160 / 2.8760 | 2.8265 | -0.75% | -1.30 / -1.07 / -0.92 / +0.81 | **-1.24** |
| m3_c3_dn_s3 | c3 | 1.3980 | 1.3980 | -0.85% | -0.92 | -0.92 |
| m4_c7_dn_s3 | c7 | 2.2800 | 2.2800 | -1.14% | -1.17 | -1.17 |
| m5_c2_md_gm8 | c2 | 8.1410 | 8.1410 | +2.22% | +1.47 | +1.47 |
| m6_c12_md_reg255 | c1 | 4.6080 / 4.6510 / 4.7510 | 4.6700 | -0.13% | -1.02 / -0.38 / +1.12 | -0.16 |
| m6_c12_md_reg255 | c2 | 7.8270 / 7.8710 / 8.1050 | 7.9343 | -0.37% | -1.14 / -0.78 / +1.17 | -0.43 |
| m7_c7_md_s4 | c7 | 2.3190 | 2.3190 | +0.55% | +0.56 | +0.56 |

## Verdicts

| cand | pairs | touched residual per pair (%) | verdict |
|---|---|---|---|
| m1_c5_dn_gm8 | 1 | c5 -0.034 | NEUTRAL |
| m2_c5_dn_s3 | 4 | c5 -0.905 / -0.579 / -0.132 / -0.628 (mean -0.561) | NEUTRAL (real, sub-threshold; best remaining lead) |
| m3_c3_dn_s3 | 1 | c3 +0.048 | NEUTRAL |
| m4_c7_dn_s3 | 1 | c7 -0.182 | NEUTRAL |
| m5_c2_md_gm8 | 1 | c2 +1.281 | NEGATIVE |
| m6_c12_md_reg255 | 3 | c1 -0.939/+0.903/-0.447 (mean -0.161); c2 -1.184/+0.044/-0.014 (mean -0.406) | NEUTRAL |
| m7_c7_md_s4 | 1 | c7 +1.353 | NEGATIVE |

No candidate reaches PROMOTE-CANDIDATE. SQNR was identical to the anchor on every case of every run
(c1 23.12, c2 23.14, c3 23.19, c5 23.13, c7 23.12) and every run scored 100 on all 12 cases.
maxnreg=255 compiles and runs cleanly on the judge (no TLE, no compile-budget failure) — useful for future knobs.

> Coordinator note received mid-run: production `p1/kernel.py` moves to v838. All 12 anchors above were
> submitted from v837 (sha `81d994be…`), which is byte-identical to `p1/kernel_v837_c8_g_s4reg232.py`
> (verified). Any further anchor for these pairs must come from that explicit file.
