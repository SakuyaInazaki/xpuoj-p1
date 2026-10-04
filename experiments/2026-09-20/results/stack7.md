# 2026-09-20 stack7 — A (c5 dn num_stages 4→3) × B (c9/c10 tiled md maxnreg 232→200) on the v838 anchor

Production anchor = unmodified `p1/kernel.py` **v838**, SHA-256
`96f7b2cb40e0ddc2c8d6af53be43e51e32380538714fd6413b8b008823e324b0` (never modified this round).

## Candidates (v838 + change; `diff p1/kernel.py <cand>` shows only the gated line(s))

| id | file | sha256 (short) | change |
|---|---|---|---|
| s1 | `s1_c5dn_s3.py` | `3a9d6df7915b` | **A**: `_dn_tma2_f8_host` line 4593 `num_stages=3 if (K == 2560 and N == 3584) else (3 if K == 14336 else 4)` (c5 only) |
| s2 | `s2_c910md_reg200.py` | `4d15e7eda667` | **B**: `_fgs_tma1_host_tiled` line 1830 `maxnreg` 232→200 for `_fgs_tma1_kernel_gq_tiled` (c9, c10) |
| s3 | `s3_stack_A_B.py` | `9496b25754c4` | A + B |

`py_compile` + `scripts/check_jit_globals.py` pass on all three.
Touched cases: s1 → 5; s2 → 9, 10; s3 → 5, 9, 10.
Anchor SQNR reference: c5 23.13, c9 23.13, c10 23.11.

## SID log (appended immediately after each submission)

- pair1 s1 (A, touched c5): cand SID=**146565**, anchor SID=**146566**
- pair2 s2 (B, touched c9/c10): cand SID=**146569**, anchor SID=**146570**
- pair3 s3 (A+B, touched c5/c9/c10): cand SID=**146573**, anchor SID=**146574**
- pair4 s1 (A, touched c5): cand SID=**146577**, anchor SID=**146578**
- pair5 s2 (B, touched c9/c10): cand SID=**146582**, anchor SID=**146583**
- pair6 s3 (A+B, touched c5/c9/c10): cand SID=**146587**, anchor SID=**146588**
- pair7 s1 (A, touched c5): cand SID=**146591**, anchor SID=**146592**
- pair8 s2 (B, touched c9/c10): cand SID=**146595**, anchor SID=**146596**
- pair9 s3 (A+B, touched c5/c9/c10): cand SID=**146600**, anchor SID=**146601**
- pair10 s2 (B, touched c9/c10): cand SID=**146604**, anchor SID=**146605**
- pair11 s3 (A+B, touched c5/c9/c10): cand SID=**146608**, anchor SID=**146609**

All 11 pairs (22 submissions) Accepted, 100/100 on all 12 cases, no sample-slot TimeLimitExceeded,
no resubmission of identical candidate code.

## Per-pair results (`scripts/mnorm.py` convention: m = machine term fit on the untouched cases, + = slower)

| pair | cand | cand/anchor SID | m | c5 resid | c9 resid | c10 resid |
|---|---|---|---|---|---|---|
| 1 | s1 | 146565/146566 | +1.017% | **−0.429%** (raw 2.8690 vs 2.8310) | — | — |
| 2 | s2 | 146569/146570 | −1.173% | — | **−0.216%** (2.4930 vs 2.5030) | **+1.690%** (1.9640 vs 1.9450) |
| 3 | s3 | 146573/146574 | +1.309% | **−0.617%** (2.8730 vs 2.8260) | **+0.231%** (2.5350 vs 2.5240) | **+0.547%** (1.9610 vs 1.9350) |
| 4 | s1 | 146577/146578 | −1.189% | **−1.188%** (2.7910 vs 2.8850) | — | — |
| 5 | s2 | 146582/146583 | +1.248% | — | **+2.009%** (2.5500 vs 2.4950) | **−0.200%** (1.9770 vs 1.9660) |
| 6 | s3 | 146587/146588 | −1.188% | **+0.476%** (2.8390 vs 2.8850) | **+2.036%** (2.5330 vs 2.4870) | **+1.493%** (1.9640 vs 1.9490) |
| 7 | s1 | 146591/146592 | +1.375% | **−0.185%** (2.8690 vs 2.8070) | — | — |
| 8 | s2 | 146595/146596 | −1.142% | — | **+0.060%** (2.5230 vs 2.5260) | **+1.106%** (1.9540 vs 1.9460) |
| 9 | s3 | 146600/146601 | +1.296% | **+0.061%** (2.8690 vs 2.8040) | **+2.526%** (2.5600 vs 2.4920) | **+1.026%** (1.9640 vs 1.9290) |
| 10 | s2 | 146604/146605 | −1.295% | — | **−0.114%** (2.5160 vs 2.5240) | **−0.018%** (1.9690 vs 1.9850) |
| 11 | s3 | 146608/146609 | −1.393% | **−0.420%** (2.7990 vs 2.8810) | **+0.889%** (2.5520 vs 2.5350) | **−0.774%** (1.9420 vs 1.9740) |

### Pair-mean ± se (t = mean/se)

| set | n | mean | sd | se | t | same-sign |
|---|---|---|---|---|---|---|
| **A** c5, s1 only (v838) | 3 | **−0.601%** | 0.523 | 0.302 | −1.99 | 3/3 negative |
| **A** c5, s3 only (stacked, v838) | 4 | **−0.125%** | 0.492 | 0.246 | −0.51 | 2/4 negative |
| **A** c5, all v838 (s1+s3) | 7 | **−0.329%** | 0.526 | 0.199 | −1.65 | 5/7 negative |
| **A** c5, v838 + prior v837 m2 | 11 | **−0.413%** | 0.459 | 0.138 | −2.99 | 9/11 negative |
| **B** c9, s2 only (v838) | 4 | **+0.435%** | 1.056 | 0.528 | +0.82 | 2/4 negative |
| **B** c9, s3 only (stacked, v838) | 4 | **+1.420%** | 1.049 | 0.524 | +2.71 | 0/4 negative |
| **B** c9, all v838 (s2+s3) | 8 | **+0.928%** | 1.107 | 0.392 | +2.37 | 2/8 negative |
| **B** c9, v838 + prior v837 R4a | 16 | **+0.131%** | 1.398 | 0.350 | +0.38 | 9/16 negative |
| **B** c10, all v838 | 8 | **+0.609%** | 0.873 | 0.309 | +1.97 | 3/8 negative |
| **B** c10, v838 + prior v837 R4a | 16 | **+0.370%** | 0.854 | 0.213 | +1.73 | 5/16 negative |

## Pooled same-code raw comparison

Anchor pool = 22 v838 anchor runs (my 11 + knobs6's `n*_` line anchors
146564/568/572/576/580/586/590/594/599/603/607) **plus** the 54 v837 anchor runs of
`knobs5.md` / `c910_replicate.md` / `cheap_points.md` (c5/c9/c10 code is byte-identical in v837 and
v838 — v838 only adds the c6 dn `GROUP_M`). n = 76.

| case | anchor pool | candidate runs | mean | Δ% | z of cand mean | frac below pool mean |
|---|---|---|---|---|---|---|
| c5 | n=76, 2.8518 ± 0.0334 | **A**: s1 3 + s3 4 (v838) | 2.8441 | **−0.27%** | −0.63 | 3/7 = 43% |
| c5 | n=76 | **A**: + prior v837 m2 4 ⇒ n=11 | 2.8377 | **−0.49%** | **−1.40** | 6/11 = 55% |
| c9 | n=76, 2.5164 ± 0.0217 | **B**: s2 4 + s3 4 (v838) | 2.5328 | **+0.65%** | +2.43 | 2/8 = 25% |
| c9 | n=76 | **B**: + prior v837 R4a 8 ⇒ n=16 | 2.5207 | **+0.17%** | +0.79 | 7/16 = 44% |
| c10 | n=76, 1.9500 ± 0.0174 | **B**: v838 8 + v837 R4a 8 ⇒ n=16 | 1.9573 | **+0.37%** | +1.67 | 4/16 = 25% |

(v838-only anchor sub-pool, n=22: c5 2.8520 ± 0.0330, c9 2.5167 ± 0.0187, c10 1.9505 ± 0.0170 —
same conclusions, c9 candidate z=+0.86 per-run / +2.43 on the mean.)

## SQNR

All 22 runs: SQNR identical to its own anchor on **all 12** cases —
c1 23.12, c2 23.14, c3 23.19, c4 23.19, **c5 23.13**, c6 23.14, c7 23.12, c8 23.12,
**c9 23.13**, **c10 23.11**, c11 23.31, c12 23.25. Every case scored 100/100 in every run.

## Integer-point crossings (each run judged with its own tb)

`c9 crossed` = tk ≤ tb·23/77 (c9 reaches 77 pts); `c5 crossed` = tk ≤ tb·18/82 (c5 reaches 82 pts).

| cand | c9 crossed | c5 crossed |
|---|---|---|
| s1 (3 runs) | 1/3 (pair4) | 1/3 (pair4, tk 2.7910 — the only genuinely-fast crossing) |
| s2 (4 runs) | 4/4 (pairs 2, 5, 8, 10) | 0/4 |
| s3 (4 runs) | 1/4 (pair3) | 1/4 (pair9, on a lucky tb = 14.72) |
| anchor v838 (11 runs) | 5/11 | 1/11 |

s2's 4/4 on c9 is a **tb artifact**, not a tk gain: its four tb draws (8.391 / 8.828 / 8.507 / 8.483)
were all above its anchors' (8.314 / 8.406 / 8.341 / 8.347), while its raw tk mean is +0.16% *above*
the anchor pool.

## Verdicts

| change | pair-mean on its touched case | pooled frac-below | pooled z | verdict |
|---|---|---|---|---|
| **A** = c5 dn `num_stages` 4→3 | −0.413% (11 pairs) / −0.329% (7 v838 pairs) | 55% | −1.40 | **NEUTRAL** — real but sub-threshold (≈ −0.014 ms); fails frac-below ≥75% and z ≤ −1.5 |
| **B** = c9/c10 tiled md `maxnreg` 232→200 | c9 **+0.928%** (8 v838 pairs, t=+2.37) / +0.131% (16 pairs); c10 +0.370% | c9 44%, c10 25% | c9 +0.79, c10 +1.67 | **NEGATIVE on v838 / NEUTRAL pooled** — the v837 −0.5% lead did **not** replicate |
| **A+B stacked** (s3) | c5 −0.125%, c9 +1.420%, c10 +0.573% | — | — | **NEUTRAL/NEGATIVE** — no better than either alone |

**Did stacking change either?** Not significantly.
A's c5 residual is −0.601% alone (s1) vs −0.125% stacked (s3): difference +0.476% ± 0.39 (n.s.).
B's c9 residual is +0.435% alone (s2) vs +1.420% stacked (s3): difference +0.985% ± 0.74 (n.s.).
Both drifts are toward "worse when stacked", but each is inside the documented ±1.07% single-pair
c9 / ±0.5% c5 residual noise floor. There is no evidence of a super-additive gain, and no case where
the stack beats its best component.

**Recommendation: promote nothing.** `p1/kernel.py` v838 is unchanged
(SHA-256 `96f7b2cb40e0ddc2c8d6af53be43e51e32380538714fd6413b8b008823e324b0`).
