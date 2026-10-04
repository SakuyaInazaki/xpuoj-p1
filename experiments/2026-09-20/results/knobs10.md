# 2026-09-20 knobs10 — c11/c12 aux launch-config knobs on the v840 anchor

Anchor = unmodified `p1/kernel.py` **v840** (sha256 `ae961a07a7c1…` = v839 + c4 dn num_stages 3).
Case key = (T, H, E, I, topk) from `_KNOWN12`; c11 = (65536, 1024, 32, 1024, 2), c12 = (65536, 1024, 32, 2048, 2).
**Every knobs10 candidate touches c11 AND c12** (identical aux geometry), so each pair yields two readings.

## Launch values found in v840 before changing anything

| kernel | host | c11/c12 launch as shipped |
|---|---|---|
| `_route_full_kernel` | `_route_full` L996 | `e_pad=32` → `BLOCK_M=128`, `BLOCK_K=128`, `num_warps=8`, `num_stages=3`; grid = ceil(65536/128) = **512 CTAs** |
| `_gq1p_tm_kernel` (quantizer) | `_gq1p_tm` L871 | grid = `(T,)` = **65536 CTAs**, `BLOCK_H=next_pow2(1024)=1024`, `K_BRANCH=2`, `num_warps=4`, `num_stages=1` |
| `_sort_hist_kernel` | `_counting_sort_order` L750 | `BLOCK=256` (16384//32=512 clamped to 256), `E_PAD=32`, C = ceil(131072/256) = **512**, `num_warps=8, num_stages=1` |
| `_csort_colscan_kernel` / `_csort_offsets_kernel` | same | grid `(32,)`, `C_PAD=512`, `num_warps=4, num_stages=1` |
| `_sort_scatter_kernel` | same | grid `(512,)`, `BLOCK=256`, `E_PAD=32`, `num_warps=8, num_stages=1` |
| `_gather_branch_sum_f8_kernel` (fin) | `_gather_branch_sum_f8` L4715 | `H>=1024` branch → `BLOCK_T=32`, `BLOCK_H=256`, `num_warps=8`, `num_stages=1`; grid = (2048, 4) |

Gates are exclusive to c11/c12 among all 12 cases:
`H == 1024` (quantizer, fin — no other case has hidden 1024), `e_pad == 32 and K == 1024`
(route — c3/c4 also have E=32 but K=H=2048), `e_pad == 32 and n == 131072` (sort — c3/c4 give n=65536).

## Roofline note recorded before the first submission
At ~3.0 TB/s achievable HBM these three aux kernels are already near the streaming bound:
route reads x = 128 MB ⇒ ≈43 µs vs 41 µs measured; quantizer reads 128 MB + writes 134 MB + scales
⇒ ≈88 µs vs 80 µs measured; fin reads 134 MB FP8 + writes 134 MB BF16 ⇒ ≈89 µs vs 99 µs measured.
So the realistically addressable headroom in the whole aux chain is single-digit µs, well under the
0.063 ms c11 needs for 86. knobs10 is therefore a fine-grained sweep, not a line-crossing play.

## Candidates (each = v840 + exactly ONE gated line; `diff p1/kernel.py <cand>` verified one-line for all 10)

| cand | change | gate | sha12 |
|---|---|---|---|
| x1_gq_nw8 | `_gq1p_tm` `num_warps` 4→8 (L883) | `H == 1024` | e250115ddb0b |
| x2_gq_nw2 | `_gq1p_tm` `num_warps` 4→2 (L883) | `H == 1024` | 0f07305bfe3f |
| x3_route_bm64 | `_route_full` `BLOCK_M` 128→64 (L1015) | `e_pad == 32 and K == 1024` | 0aa21ccec395 |
| x4_route_bm256 | `_route_full` `BLOCK_M` 128→256 (L1015) | `e_pad == 32 and K == 1024` | 48aa2c2d59f6 |
| x5_route_s2 | `_route_full_kernel` `num_stages` 3→2 (L1030) | `e_pad == 32 and K == 1024` | 8eec66563be3 |
| x6_route_nw4 | `_route_full_kernel` `num_warps` 8→4 (L1030) | `e_pad == 32 and K == 1024` | 7fecd4220337 |
| x7_fin_nw4 | `_gather_branch_sum_f8` `num_warps` 8→4 (L4721) | `H == 1024` | 73ee9112ec66 |
| x8_fin_nw16 | `_gather_branch_sum_f8` `num_warps` 8→16 (L4721) | `H == 1024` | 45304b2e984d |
| x9_fin_bt64 | `_gather_branch_sum_f8` `BLOCK_T` 32→64 (L4719) | `H == 1024` | ec3ced7f97d0 |
| x10_sort_nw16 | `_sort_scatter_kernel` `num_warps` 8→16 (L805) | `e_pad == 32 and n == 131072` | f9d38c883625 |

**x9 deviation from the brief, deliberate:** the brief asked for the fin *column* BLOCK one step up.
That is unsafe here: `_gather_branch_sum_f8_kernel` dequantizes with one scale per 256-column chunk and
reads it as `DSCL[src_row*NCHUNK + (pid_h*BLOCK_H)//256]`, i.e. a CTA applies a single chunk scale to its
whole `BLOCK_H` span. `BLOCK_H=512` would apply chunk 0's scale to chunk 1 ⇒ wrong numerics, not a knob.
H==1024 is 4 column blocks, not one, so the brief's own fallback wording applies: x9 instead doubles the
*token* block `BLOCK_T` 32→64 (2× tokens per CTA), which is the host-exposed "more tokens per CTA" knob.

**x3 numerics caveat, recorded in advance:** the in-code comment at L1000–1011 documents that for
`e_pad==64` dropping `BLOCK_M` to 64 required `num_warps` to drop to 4 in lockstep, because 8 warps on a
64-row tile force warpsPerCTA `[4,2]`, which turns the `axis=1` softmax sum into a cross-warpgroup FP32
reduction. x3 changes `BLOCK_M` alone (as briefed, one change per candidate), so it may come back with a
different SQNR; if it does, the bitwise-safe variant is x3+x6 together and that is what should be retried.

## Estimator
`experiments/2026-09-20/strat3.py`. Machine index M = mean over cases {1,2,3,5,7,8,9,10} of
tk_i / anchor-pool-mean_i — c4 and c6 are excluded because the pooled anchors span v837/v838/v839/v840
which differ only at the c6 dn and the c4 dn; c11/c12 are the touched cases. c11/c12 code is
byte-identical across v837..v840, so all 93 pooled anchors are usable for the fit:
- c11: n=93, tk = 0.5442 + 0.4277*M, resid **sd 0.0020 ms = 0.205%**, M explains 80%.
- c12: n=93, tk = 0.3560 + 1.2218*M, resid **sd 0.0030 ms = 0.191%**, M explains 94%.

Verdict rule: PROMOTE-CANDIDATE iff stratified z_of_mean ≤ −3 AND ≥75% of runs negative AND SQNR equal
on all 12. Schedule: one pair each round-robin; ≤6 extra pairs to any candidate whose first residual on
c11 or c12 is ≤ −0.3%; stop after 2 pairs if both first residuals are ≥ +0.3%.
Crossing targets: c11 → 86 needs tk ≤ tb·14/86; c12 → 84 needs tk ≤ tb·16/84 (each run's own tb).

## SID log
- kpair1 x1_gq_nw8 (touches c11+c12): cand SID=146734, anchor SID=146735
- kpair2 x2_gq_nw2: cand SID=146739, anchor SID=146740
- kpair3 x7_fin_nw4: cand SID=146743, anchor SID=146744
- kpair4 x2_gq_nw2 (lead, pair 2): cand SID=146749, anchor SID=146750
- kpair5 x2_gq_nw2 (pair 3): cand SID=146751, anchor SID=146753
- kpair6 x2_gq_nw2 (pair 4): cand SID=146755, anchor SID=146756
- kpair7 x2_gq_nw2 (pair 5): cand SID=146761, anchor SID=146762
- kpair8 x2_gq_nw2 (pair 6): cand SID=146766, anchor SID=146767
- kpair9 x2_gq_nw2 (pair 7): cand SID=146773, anchor SID=146774
- kpair10 x2_gq_nw2 (pair 8): cand SID=146778, anchor SID=146779
- kpair11 x12_gq_nw1 (= "x2b", num_warps 1, sha12 08ef9b478867): cand SID=146782, anchor SID=146783
- kpair12 x12_gq_nw1 (pair 2): cand SID=146787, anchor SID=146788
- kpair13 x12_gq_nw1 (pair 3): cand SID=146793, anchor SID=146794
- kpair14 x9_fin_bt64: cand SID=146797, anchor SID=146798
- kpair15 x10_sort_nw16: cand SID=146803, anchor SID=146804
- kpair16 x13_sort_nw4 (sha12 e029243a4c90): cand SID=146807, anchor SID=146809
- kpair17 x13_sort_nw4 (pair 2): cand SID=146814, anchor SID=146815
- kpair18 x13_sort_nw4 (pair 3): cand SID=146820, anchor SID=146821
- kpair19 x13_sort_nw4 (pair 4): cand SID=146825, anchor SID=146826
- kpair20 x6_route_nw4: cand SID=146830, anchor SID=146831
- kpair21 x5_route_s2: cand SID=146834, anchor SID=146835
- kpair22 x8_fin_nw16: cand SID=146839, anchor SID=146840
- kpair23 x4_route_bm256: cand SID=146845, anchor SID=146846
- kpair24 x3_route_bm64: cand SID=146849, anchor SID=146850
- kpair25 x3_route_bm64 (pair 2): cand SID=146854, anchor SID=146855

### kpair1 — x1_gq_nw8, cand 146734 / anchor 146735
SQNR EQUAL on all 12 (c11 23.31, c12 23.25). Both Accepted, display 81.75 / 81.58.
- c11: raw 0.9680 → 1.0070 (+4.03%); M=1.0087; stratified resid **+0.0314 ms (+3.219%), z=+15.83**; pts 84, need ≤ 0.9090 for 86 — no.
- c12: raw 1.5670 → 1.6220 (+3.51%); stratified resid **+0.0337 ms (+2.121%), z=+11.21**; pts 83, need ≤ 1.5472 for 84 — no.
Verdict: **NEGATIVE, closed after 1 pair** (rule said 2, but z=+15.8 / +11.2 leaves nothing to resolve).
Mechanistic read: +0.031 ms is ~39% of the whole 0.080 ms quantizer. `BLOCK_H=1024`, so 4 warps =
128 threads = 8 BF16 = one 16-byte (128-bit) load per thread; 8 warps halves that to an 8-byte load and
loses the widest vector access. This makes `num_warps` a *first-order* knob on this kernel and promotes
x2 (num_warps 2, i.e. 2 × 128-bit loads per thread with twice the CTA residency) to the next slot.

### kpair2 — x2_gq_nw2, cand 146739 / anchor 146740
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9690 → 0.9710 (+0.21% raw, but the cand ran on a slower machine, M=1.0127);
  stratified resid **−0.0063 ms (−0.641%), z=−3.17**; pts 85, need ≤ 0.9118 for 86 — no.
- c12: raw 1.5660 → 1.5840; stratified resid **−0.0091 ms (−0.571%), z=−3.04**; pts 83, need ≤ 1.5488 for 84 — no.
Verdict so far: **LEAD** (both cases ≤ −0.3% on the first pair) ⇒ allocate extra pairs.
Confirms the x1 mechanism with the sign reversed: 2 warps = 64 threads × 16 BF16 = two 128-bit loads
per thread and 2× the CTA residency; 8 warps (x1) = one 64-bit load and it costs +3%.

### Anchor-pooling check (CPU-side, no submissions)
Splitting the 95 pooled anchors into the v837 group (n=33) and the v838/v839 group (n=62) and taking
each group's mean residual off the common fit: c11 +0.003% vs −0.002%, c12 −0.007% vs +0.004%
(group sds 0.18–0.21%). The pool is homogeneous at c11/c12, as the code-identity argument predicted.

### kpair3 — x7_fin_nw4, cand 146743 / anchor 146744
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9690 → 0.9770; M=1.0118; stratified resid **+0.0001 ms (+0.014%), z=+0.07**; pts 85, need ≤ 0.9134 — no.
- c12: raw 1.5740 → 1.5950; stratified resid **+0.0030 ms (+0.190%), z=+1.01**; pts 83, need ≤ 1.5480 — no.
Verdict: **NEUTRAL after 1 pair, no extra pairs allocated** (neither case reached the −0.3% lead
threshold, and c12 leans positive). Note the contrast with the quantizer: fin is much less warp-sensitive,
which fits its tile (`BLOCK_T=32 × BLOCK_H=256` = 8192 elements) already giving 32 elements/thread at 8 warps.

### kpair4 — x2_gq_nw2 (pair 2), cand 146749 / anchor 146750
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9670 → 0.9740; M=1.0073; stratified resid **−0.0009 ms (−0.089%), z=−0.44**; pts 85, need ≤ 0.9115 — no.
- c12: raw 1.5680 → 1.5910; stratified resid **+0.0047 ms (+0.298%), z=+1.59**; pts 83, need ≤ 1.5448 — no.
x2 running totals (n=2): c11 mean **−0.367%, z_of_mean=−2.57, 2/2 negative**;
c12 mean **−0.139%, z_of_mean=−1.04, 1/2 negative**. Lead still open, continuing to n=8.

### kpair5 — x2_gq_nw2 (pair 3), cand 146751 / anchor 146753
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9740 → 0.9730; M=1.0119; stratified resid −0.0010 ms (−0.106%), z=−0.52; pts 85 — not crossed.
- c12: raw 1.5990 → 1.5850; stratified resid **−0.0072 ms (−0.454%), z=−2.37**; pts 83, need ≤ 1.5480 — no.
x2 running totals (n=3): c11 mean **−0.371%, z_of_mean=−3.17, 3/3 negative**;
c12 mean **−0.253%, z_of_mean=−2.26, 2/3 negative**.

### kpair6 — x2_gq_nw2 (pair 4), cand 146755 / anchor 146756
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9680 → 0.9740; M=1.0121; stratified resid −0.0021 ms (−0.215%), z=−1.06; pts 85 — not crossed.
- c12: raw 1.5710 → 1.5840; stratified resid **−0.0084 ms (−0.528%), z=−2.75**; pts 83, need ≤ 1.5480 — no.
x2 running totals (n=4): c11 mean **−0.353%, z_of_mean=−3.49, 4/4 negative**;
c12 mean **−0.323%, z_of_mean=−3.33, 3/4 negative**. Both arms now clear the z ≤ −3 bar; going to n=8.

### kpair7 — x2_gq_nw2 (pair 5), cand 146761 / anchor 146762
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9790 → 0.9670; M=0.9915; stratified resid −0.0012 ms (−0.121%), z=−0.60; pts 85 — not crossed.
- c12: raw 1.5890 → 1.5630; stratified resid −0.0043 ms (−0.273%), z=−1.40; pts 83, need ≤ 1.5596 — no.
x2 running totals (n=5): c11 mean **−0.311%, z_of_mean=−3.44, 5/5 negative**;
c12 mean **−0.309%, z_of_mean=−3.56, 4/5 negative**.

### kpair8 — x2_gq_nw2 (pair 6), cand 146766 / anchor 146767
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9680 → 0.9710; M=1.0099; stratified resid **−0.0050 ms (−0.513%), z=−2.56**; pts 85, need ≤ 0.9108 — no.
- c12: raw 1.5690 → 1.5860; stratified resid −0.0037 ms (−0.230%), z=−1.20; pts 83, need ≤ 1.5497 — no.
x2 running totals (n=6): c11 mean **−0.344%, z_of_mean=−4.19, 6/6 negative**;
c12 mean **−0.296%, z_of_mean=−3.75, 5/6 negative**.

### kpair9 — x2_gq_nw2 (pair 7), cand 146773 / anchor 146774
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9780 → 0.9610; M=0.9917; stratified resid **−0.0072 ms (−0.747%), z=−3.71**; pts 85, need ≤ 0.9198 — no.
- c12: raw 1.5900 → 1.5670; stratified resid −0.0005 ms (−0.032%), z=−0.17; **pts 84 — CROSSED**
  (its own tb=8.246 ⇒ need ≤ 1.5726, tk 1.5670). First 84 on c12 in this round; the anchor of the
  same pair sat at 83, so it is a genuine line crossing on a favourable-tb draw rather than a free win.
x2 running totals (n=7): c11 mean **−0.403%, z_of_mean=−5.32, 7/7 negative**;
c12 mean **−0.255%, z_of_mean=−3.50, 6/7 negative**.

### kpair10 — x2_gq_nw2 (pair 8), cand 146778 / anchor 146779
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9690 → 0.9750; M=1.0094; stratified resid −0.0008 ms (−0.085%), z=−0.43; pts 85 — not crossed.
- c12: raw 1.5690 → 1.5850; stratified resid −0.0040 ms (−0.254%), z=−1.33; pts 83 — not crossed.

## x2_gq_nw2 — FINAL, n=8 pairs (fit on 103 pooled anchors)

| run | M | c11 resid | c11 z | c12 resid | c12 z |
|---|---|---|---|---|---|
| 146739 | 1.0127 | −0.642% | −3.23 | −0.574% | −3.01 |
| 146749 | 1.0071 | −0.089% | −0.45 | +0.295% | +1.54 |
| 146751 | 1.0119 | −0.399% | −2.01 | −0.444% | −2.33 |
| 146755 | 1.0120 | −0.303% | −1.52 | −0.519% | −2.72 |
| 146761 | 0.9916 | −0.122% | −0.61 | −0.276% | −1.42 |
| 146766 | 1.0098 | −0.515% | −2.58 | −0.226% | −1.18 |
| 146773 | 0.9917 | −0.750% | −3.73 | −0.036% | −0.18 (**pts 84, CROSSED**) |
| 146778 | 1.0094 | −0.085% | −0.43 | −0.254% | −1.33 |
| **mean** | | **−0.364% (−0.0035 ms)** | **z_of_mean −5.15** | **−0.256% (−0.0040 ms)** | **z_of_mean −3.76** |

- % negative: c11 **8/8 = 100%**, c12 **7/8 = 87.5%** — both ≥ 75%.
- SQNR: **EQUAL on all 12 cases in all 8 pairs** (c11 23.31, c12 23.25 throughout).
- Crossing: one run (146773) reached c12 = 84 pts; c11 never reached 86 (best run still needed
  tk ≤ 0.9198 against tk 0.9610, i.e. ~4.3% short — consistent with the roofline note above).

**VERDICT: PROMOTE-CANDIDATE `x2_gq_nw2`** — z_of_mean ≤ −3 on both touched cases, ≥75% of runs
negative on both, SQNR identical. Expected steady-state gain ≈ 0.0035 ms on c11 + 0.0040 ms on c12,
i.e. ~4–5% of the 0.080 ms quantizer, worth roughly 0.045 + 0.034 ≈ 0.08 raw points.

## x12_gq_nw1 ("x2b") — quantizer `num_warps` 4→1, gate `H == 1024` (L883, sha12 08ef9b478867)
BLOCK_H=1024 with 1 warp = 32 threads × 32 BF16 = four 128-bit loads per thread, 4× the CTA residency of x2.

### kpair11 — x12_gq_nw1 (pair 1), cand 146782 / anchor 146783
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9680 → 0.9720; M=1.0105; stratified resid **−0.0043 ms (−0.439%), z=−2.22**; pts 85, need ≤ 0.9102 — no.
- c12: raw 1.5710 → 1.5850; stratified resid **−0.0054 ms (−0.337%), z=−1.77**; pts 83, need ≤ 1.5617 — no.
Both arms ≤ −0.3% on the first pair ⇒ LEAD, extra pairs allocated. Same ballpark as x2's 8-pair mean,
so the open question is whether 1 warp beats 2, not whether it beats the shipped 4.

### kpair12 — x12_gq_nw1 (pair 2), cand 146787 / anchor 146788
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9770 → 0.9680; M=0.9917; stratified resid −0.0003 ms (−0.026%), z=−0.13;
  **pts 86 — CROSSED** (its own tb=5.946 ⇒ need ≤ 1.0215, tk 0.9680). First 86 on c11 in this round;
  the tb draw was high, so this is the target line being reached, not a tk breakthrough.
- c12: raw 1.5880 → 1.5640; stratified resid −0.0036 ms (−0.230%), z=−1.18; pts 83, need ≤ 1.5632 — no.
x12 running totals (n=2): c11 mean −0.233%, z_of_mean=−1.66, 2/2 negative;
c12 mean −0.280%, z_of_mean=−2.05, 2/2 negative.

### kpair13 — x12_gq_nw1 (pair 3), cand 146793 / anchor 146794
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9680 → 0.9750; M=1.0074; stratified resid +0.0000 ms (+0.003%), z=+0.02; pts 85 — not crossed.
- c12: raw 1.5660 → 1.5880; stratified resid +0.0015 ms (+0.093%), z=+0.48; pts 83 — not crossed.
x12 totals (n=3): c11 mean **−0.154%, z_of_mean=−1.35, 2/3 negative**;
c12 mean **−0.155%, z_of_mean=−1.39, 2/3 negative**.
Verdict: **POSITIVE BUT WEAKER THAN x2, not promotable at n=3** — 1 warp is about half the gain of
2 warps on both cases, so the quantizer optimum is `num_warps=2`: 2 warps still issues the widest
128-bit loads (two per thread) while 1 warp gives up thread-level parallelism inside the CTA for no
extra vector width. x12 parked in favour of x2; the remaining budget goes to the untested sweep.

### kpair14 — x9_fin_bt64, cand 146797 / anchor 146798
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9750 → 0.9720; M=0.9930; stratified resid **+0.0032 ms (+0.330%), z=+1.67**; pts 85, need ≤ 0.9414 — no.
- c12: raw 1.5910 → 1.5680; stratified resid −0.0012 ms (−0.075%), z=−0.38; pts 83, need ≤ 1.5533 — no.
Verdict: **NEUTRAL/MIXED, closed after 1 pair** — no arm reached the −0.3% lead bar and c11 leans positive.

### kpair15 — x10_sort_nw16, cand 146803 / anchor 146804
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9670 → 0.9810; M=1.0123; stratified resid **+0.0039 ms (+0.403%), z=+2.06**; pts 85, need ≤ 0.9108 — no.
- c12: raw 1.5650 → 1.5940; stratified resid +0.0015 ms (+0.094%), z=+0.48; pts 83 — no.
Verdict: **NEGATIVE, closed after 1 pair** — 16 warps on the counting-sort scatter is worse than 8.
Follow-up added: **x13_sort_nw4** (same line, 8→4), because the quantizer showed the opposite
direction winning and this kernel's `BLOCK=256` with 8 warps is already exactly 1 element/thread —
fewer warps is the only untested side of this knob.

## x13_sort_nw4 — `_sort_scatter_kernel` `num_warps` 8→4, gate `e_pad == 32 and n == 131072` (L805)

### kpair16 — x13_sort_nw4 (pair 1), cand 146807 / anchor 146809
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9740 → 0.9650; M=0.9948; stratified resid **−0.0045 ms (−0.468%), z=−2.37**; pts 85, need ≤ 0.9266 — no.
- c12: raw 1.5950 → 1.5630; stratified resid **−0.0083 ms (−0.528%), z=−2.64**; pts 83, need ≤ 1.5619 — no.
Both arms ≤ −0.3% ⇒ LEAD, extra pairs allocated. This is a different kernel from x2, so if it holds
the two gains stack.

### kpair17 — x13_sort_nw4 (pair 2), cand 146814 / anchor 146815
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9670 → 0.9750; M=1.0094; stratified resid −0.0008 ms (−0.079%), z=−0.40; pts 85 — not crossed.
- c12: raw 1.5650 → 1.5850; stratified resid −0.0041 ms (−0.257%), z=−1.30; pts 83 — not crossed.
x13 totals (n=2): c11 mean **−0.272%, z_of_mean=−1.96, 2/2 negative**;
c12 mean **−0.391%, z_of_mean=−2.78, 2/2 negative**.

### kpair18 — x13_sort_nw4 (pair 3), cand 146820 / anchor 146821
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9750 → 0.9720; M=0.9942; stratified resid +0.0027 ms (+0.279%), z=+1.42; pts 85 — not crossed.
- c12: raw 1.5940 → 1.5640; stratified resid **−0.0066 ms (−0.422%), z=−2.10**; pts 83 — not crossed.
x13 totals (n=3): c11 mean −0.088%, z_of_mean=−0.78, 2/3 negative;
c12 mean **−0.403%, z_of_mean=−3.48, 3/3 negative**. The two cases are separating: c12 strong, c11 not.

### kpair19 — x13_sort_nw4 (pair 4), cand 146825 / anchor 146826
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9800 → 0.9660; M=0.9923; stratified resid −0.0025 ms (−0.260%), z=−1.30; pts 85 — not crossed.
- c12: raw 1.5910 → 1.5640; stratified resid −0.0044 ms (−0.281%), z=−1.40; pts 83 — not crossed.

## x13_sort_nw4 — status at n=4

| run | M | c11 resid | c11 z | c12 resid | c12 z |
|---|---|---|---|---|---|
| 146807 | 0.9948 | −0.468% | −2.37 | −0.528% | −2.64 |
| 146814 | 1.0094 | −0.079% | −0.40 | −0.257% | −1.30 |
| 146820 | 0.9942 | +0.279% | +1.42 | −0.422% | −2.10 |
| 146825 | 0.9923 | −0.260% | −1.30 | −0.281% | −1.40 |
| **mean** | | **−0.134% (−0.0013 ms)** | **z_of_mean −1.34** | **−0.373% (−0.0059 ms)** | **z_of_mean −3.73** |

% negative: c11 3/4 = 75%, c12 **4/4 = 100%**. SQNR EQUAL on all 12 in all 4 pairs.
**Verdict: PROMOTE-CANDIDATE on c12 only** (z_of_mean −3.73, 100% negative, SQNR identical);
**c11 UNRESOLVED** (z −1.34 at n=4 — the sign is right but it does not clear the bar).
Since the change is a single gated launch line that costs nothing on c11 even in the worst reading,
it is safe to ship together with x2, but the honest claim is "c12 −0.37%, c11 not established".

### kpair20 — x6_route_nw4, cand 146830 / anchor 146831
SQNR EQUAL on all 12 (confirms the prediction that at `BLOCK_M=128` dropping to 4 warps keeps
warpsPerCTA `[4,1]`, so the axis=1 softmax sum stays an in-warp reduction — bitwise unchanged).
- c11: raw 0.9760 → 0.9700; M=0.9907; stratified resid +0.0022 ms (+0.223%), z=+1.12; pts 85 — not crossed.
- c12: raw 1.5980 → 1.5650; stratified resid −0.0015 ms (−0.093%), z=−0.46; pts 83 — not crossed.
Verdict: **NEUTRAL, closed after 1 pair.**

### kpair21 — x5_route_s2, cand 146834 / anchor 146835
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9680 → 0.9770; M=1.0082; stratified resid +0.0016 ms (+0.164%), z=+0.83; pts 85 — not crossed.
- c12: raw 1.5690 → 1.5910; stratified resid +0.0029 ms (+0.181%), z=+0.90; pts 83 — not crossed.
Verdict: **NEUTRAL/slightly negative, closed after 1 pair.** `num_stages=3` stays.

### kpair22 — x8_fin_nw16, cand 146839 / anchor 146840
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9660 → 0.9750; M=1.0088; stratified resid −0.0006 ms (−0.066%), z=−0.34; pts 85 — not crossed.
- c12: raw 1.5670 → 1.5940; stratified resid +0.0052 ms (+0.326%), z=+1.64; pts 83 — not crossed.
Verdict: **NEUTRAL/MIXED, closed after 1 pair.** Together with x7 (nw 4) and x9 (BLOCK_T 64),
all three fin knobs come back inside ±0.35%: the shipped `BLOCK_T=32 / BLOCK_H=256 / num_warps=8`
is already at a local optimum, and fin is not the place to spend more pairs.

### kpair23 — x4_route_bm256, cand 146845 / anchor 146846
Compiles and runs (the 256×128×2B×3-stage A tile plus the B tile fits H100 smem); SQNR EQUAL on all 12
(warpsPerCTA stays `[8,1]`, so the axis=1 softmax sum remains in-warpgroup). Both Accepted.
- c11: raw 0.9780 → 0.9690; M=0.9909; stratified resid +0.0011 ms (+0.114%), z=+0.57; pts 85 — not crossed.
- c12: raw 1.5870 → 1.5700; stratified resid +0.0032 ms (+0.206%), z=+1.02; pts 83 — not crossed.
Verdict: **NEUTRAL/slightly negative, closed after 1 pair.**

### kpair24 — x3_route_bm64, cand 146849 / anchor 146850
**SQNR EQUAL on all 12** — the advance-recorded numerics caveat did NOT materialize. At `E_PAD=32`
(not 64) a 64-row tile with 8 warps evidently does not get split `[4,2]`, so the axis=1 softmax sum
stayed an in-warp reduction and the result is bitwise identical. Both Accepted.
- c11: raw 0.9760 → 0.9630; M=0.9908; stratified resid **−0.0049 ms (−0.506%), z=−2.56**; pts 85, need ≤ 0.9243 — no.
- c12: raw 1.5940 → 1.5690; stratified resid +0.0023 ms (+0.144%), z=+0.71; pts 83 — not crossed.
c11 clears the −0.3% lead bar on its first pair ⇒ one replication pair funded from the remaining budget.

### kpair25 — x3_route_bm64 (pair 2), cand 146854 / anchor 146855
SQNR EQUAL on all 12. Both Accepted.
- c11: raw 0.9680 → 0.9770; M=1.0098; stratified resid +0.0009 ms (+0.088%), z=+0.45; pts 85 — not crossed.
- c12: raw 1.5680 → 1.5850; stratified resid −0.0052 ms (−0.324%), z=−1.63; pts 83 — not crossed.
x3 totals (n=2): c11 mean −0.207%, z_of_mean=−1.49, 1/2 negative; c12 mean −0.091%, z_of_mean=−0.64, 1/2.
Verdict: **UNRESOLVED / probably noise** — the first pair's −0.506% on c11 did not replicate.

---

# FINAL SUMMARY — knobs10, 50 submissions (25 pairs), anchors pooled to n=115

All 25 anchor runs were the byte-identical v840 `p1/kernel.py`; SQNR was EQUAL on all 12 cases in
every one of the 25 pairs (c11 23.31, c12 23.25 without exception). No candidate ever produced a
non-Accepted testcase. Fit quality on the final pool: c11 resid sd ≈ 0.196%, c12 resid sd ≈ 0.199%.

| cand | change (gate) | pairs | c11 mean / z_of_mean / %neg | c12 mean / z_of_mean / %neg | verdict |
|---|---|---:|---|---|---|
| x1_gq_nw8 | gq `num_warps` 4→8 (`H==1024`) | 1 | +3.221% / +16.49 / 0% | +2.105% / +10.57 / 0% | **NEGATIVE (large)** |
| **x2_gq_nw2** | **gq `num_warps` 4→2 (`H==1024`)** | **8** | **−0.361% / −5.20 / 100%** | **−0.265% / −3.74 / 87.5%** | **PROMOTE** |
| x3_route_bm64 | route `BLOCK_M` 128→64 | 2 | −0.207% / −1.49 / 50% | −0.091% / −0.64 / 50% | unresolved |
| x4_route_bm256 | route `BLOCK_M` 128→256 | 1 | +0.115% / +0.58 | +0.209% / +1.03 | neutral |
| x5_route_s2 | route `num_stages` 3→2 | 1 | +0.164% / +0.84 | +0.186% / +0.93 | neutral |
| x6_route_nw4 | route `num_warps` 8→4 | 1 | +0.228% / +1.16 | −0.094% / −0.46 | neutral |
| x7_fin_nw4 | fin `num_warps` 8→4 | 1 | +0.013% / +0.06 | +0.173% / +0.87 | neutral |
| x8_fin_nw16 | fin `num_warps` 8→16 | 1 | −0.065% / −0.34 | +0.331% / +1.66 | neutral |
| x9_fin_bt64 | fin `BLOCK_T` 32→64 | 1 | +0.340% / +1.73 | −0.069% / −0.34 | neutral |
| x10_sort_nw16 | sort-scatter `num_warps` 8→16 | 1 | +0.401% / +2.06 | +0.074% / +0.37 | negative |
| x12_gq_nw1 | gq `num_warps` 4→1 (`H==1024`) | 3 | −0.150% / −1.33 / 67% | −0.165% / −1.43 / 67% | positive but < x2 |
| **x13_sort_nw4** | **sort-scatter `num_warps` 8→4** | **4** | −0.130% / −1.32 / 75% | **−0.374% / −3.73 / 100%** | **PROMOTE on c12; c11 unresolved** |

## Recommendations
1. **Promote `x2_gq_nw2`** (`experiments/2026-09-20/candidates/x2_gq_nw2.py`, sha12 `0f07305bfe3f`):
   `_gq1p_tm` `num_warps = 2 if H == 1024 else 4` at L883. Meets all three arbiter conditions on both
   touched cases. Expected ≈ −0.0035 ms c11 and −0.0042 ms c12.
2. **Consider stacking `x13_sort_nw4`** (sha12 `e029243a4c90`, `_sort_scatter_kernel` `num_warps=4` at
   L805 under `e_pad==32 and n==131072`): promotable on c12 (z −3.73, 4/4), unresolved but non-harmful
   on c11. Different kernel from x2, so the two are additive. Worth 3–4 more pairs to settle c11.
3. Do **not** revisit the route or fin launch geometry. Seven of the eight route/fin knobs landed inside
   ±0.35% with no consistent sign, which is exactly what the roofline note at the top of this file
   predicted: route ≈ 41 µs against a ≈43 µs streaming bound and fin ≈ 99 µs against ≈89 µs are already
   bandwidth-limited, so launch shape cannot move them.
4. The quantizer is the one aux kernel with real launch sensitivity, and it is now bracketed:
   nw 8 = +3.2%, nw 4 = shipped, nw 2 = −0.36%, nw 1 = −0.15%. **2 is the optimum**; there is nothing
   left on this axis.
5. Line crossings observed: c12 reached 84 once (SID 146773, x2) and c11 reached 86 once
   (SID 146787, x12), both on favourable tb draws. Neither is reproducible from tk alone — the
   tk deficits are ≈4% on c11 and ≈1.5% on c12, far beyond the ≈0.4% the aux chain can still give.

---

# x13 on v841 — re-derivation after x2 was promoted

Production `p1/kernel.py` is now **v841** (sha256 `0f07305bfe3f551fa8700db8950f2e01d6821b74bc126fbd827eece0bbcd3543`),
byte-identical to the promoted `x2_gq_nw2.py` (= v840 + `_gq1p_tm` `num_warps=2` for `H==1024`).
Candidate `x13b_sort_nw4_v841.py` (sha12 **5cfd6a80d066**) = v841 + the same one-line sort change at
L805: `num_warps=4 if (e_pad == 32 and n == 131072) else 8`. `diff p1/kernel.py x13b…` is that one
line; AST parses. Bracket candidate `x14_sort_nw2.py` (sha12 **8a31ffaf1010**) is the same line with 2.

## Estimator for this section: `experiments/2026-09-20/strat4.py` (paired, version-agnostic)
The v841 anchor differs from the v840 anchor **at the quantizer, which is a c11/c12 kernel**, so the
anchor regression intercept moves between versions and the old single-sided residual is not
comparable across the boundary. The paired statistic cancels the intercept exactly:

    D = (tk_cand - a - b*M_cand) - (tk_anchor - a - b*M_anchor) = (tk_cand - tk_anchor) - b*(M_cand - M_anchor)

Only the machine slope b survives, and b is a bandwidth property of the case, not of the quantizer
launch, so it is shared across versions; b is taken from the large pre-v841 anchor regression
(c11 b=0.4358, c12 b=1.2367, n=118). M still uses cases {1,2,3,5,7,8,9,10}, none of which any of
these changes touch. The null sd is **empirical**: the same D formed from 59 disjoint anchor–anchor
pairs, giving 0.0026 ms (0.269%) on c11 and 0.0043 ms (0.271%) on c12 — i.e. √2 × the single-sided
0.19%, exactly as expected. v841 anchors are registered in a separate `v841.txt` pool so they never
contaminate the pre-v841 regression.

### The 4 v840 x13 pairs, re-reduced with the paired estimator (so they pool with the v841 pairs)
| pair (cand/anchor) | c11 D | c11 z | c12 D | c12 z |
|---|---|---|---|---|
| 146807/146809 | −0.269% | −1.00 | −0.870% | −3.25 |
| 146814/146815 | +0.060% | +0.22 | −0.067% | −0.24 |
| 146820/146821 | +0.367% | +1.37 | −0.711% | −2.65 |
| 146825/146826 | −0.680% | −2.55 | −0.389% | −1.45 |
| **mean (n=4)** | **−0.132%** | **z_of_mean −0.98** | **−0.514%** | **z_of_mean −3.80** |
2/4 negative on c11, 4/4 on c12 — same conclusion the single-sided strat3 reduction gave.

## SID log (v841 section)
- vpair1 x13b_sort_nw4_v841 (v841 base): cand SID=146862, anchor SID=146863
- vpair2 x13b_sort_nw4_v841: cand SID=146866, anchor SID=146867
- vpair3 x13b_sort_nw4_v841: cand SID=146868, anchor SID=146869
- vpair4 x13b_sort_nw4_v841: cand SID=146871, anchor SID=146872
- vpair5 x13b_sort_nw4_v841 (tie-break block): cand SID=146874, anchor SID=146875
- vpair6 x13b_sort_nw4_v841: cand SID=146877, anchor SID=146878
- vpair7 x13b_sort_nw4_v841: cand SID=146888, anchor SID=146889
- wpair1 x14_sort_nw2: cand SID=146893, anchor SID=146894
- wpair2 x14_sort_nw2: cand SID=146899, anchor SID=146900
- wpair3 x14_sort_nw2: cand SID=146902, anchor SID=146903

### vpair1 — x13b_sort_nw4_v841, cand 146862 / anchor 146863
SQNR EQUAL on all 12. Both Accepted. Anchor raw tk c11 0.9620 / c12 1.5550 — both below the whole
v840 anchor range (0.967–0.980 / 1.565–1.599), independent corroboration that the promoted x2 is real.
- c11: tk 0.9620 → 0.9710, dM=+0.0229; paired **D = −0.0010 ms (−0.101%), z=−0.37**; pts 85 — not crossed.
- c12: tk 1.5550 → 1.5920, dM=+0.0229; paired **D = +0.0087 ms (+0.560%), z=+2.04**; pts 83 — not crossed.
(Large machine drift within the pair, |dM|=0.023 vs a typical 0.015, so this is one of the noisier draws.)

### vpair2 — x13b_sort_nw4_v841, cand 146866 / anchor 146867
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9740 → 0.9610, dM=−0.0184; paired **D = −0.0050 ms (−0.511%), z=−1.89**; pts 85 — not crossed.
- c12: tk 1.5920 → 1.5600, dM=−0.0184; paired **D = −0.0092 ms (−0.580%), z=−2.02**; pts 83 — not crossed.
Pooled over all 6 x13 pairs (4 on v840 + 2 on v841): c11 mean −0.190%, z_of_mean=−1.71, 4/6 negative;
c12 mean **−0.348%, z_of_mean=−2.95**, 5/6 negative.

### vpair3 — x13b_sort_nw4_v841, cand 146868 / anchor 146869
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9720 → 0.9620, dM=−0.0213; paired D = −0.0007 ms (−0.074%), z=−0.27; pts 85 — not crossed.
- c12: tk 1.5900 → 1.5590, dM=−0.0213; paired D = −0.0047 ms (−0.293%), z=−1.02; pts 83 — not crossed.
Pooled (n=7): c11 mean −0.173%, z_of_mean=−1.69, 5/7 negative;
c12 mean **−0.341%, z_of_mean=−3.12**, 6/7 = 86% negative — c12 now clears both bars.

### vpair4 — x13b_sort_nw4_v841, cand 146871 / anchor 146872
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9630 → 0.9720, dM=+0.0167; paired D = +0.0017 ms (+0.177%), z=+0.65; pts 85 — not crossed.
- c12: tk 1.5610 → 1.5830, dM=+0.0167; paired D = +0.0013 ms (+0.083%), z=+0.28; pts 83 — not crossed.

## x13 across v840 + v841 — all reductions at n=8 (SQNR EQUAL on all 12 in all 8 pairs)

| reduction | c11 mean / z_of_mean / %neg | c12 mean / z_of_mean / %neg |
|---|---|---|
| paired `strat4`, n=8 (no intercept assumption) | −0.130% / **−1.36** / 5/8 | −0.288% / **−2.83** / 6/8 |
| single-sided `strat5`, n=8 (shared slope, per-version intercept) | −0.129% / **−1.87** / 6/8 | −0.218% / **−3.01** / 7/8 |
| v840 pairs only (`strat3`, n=4) | −0.130% / −1.32 / 3/4 | −0.374% / −3.73 / 4/4 |
| **v841 pairs only (paired, n=4)** | −0.128% / −0.95 / 3/4 | **−0.062% / −0.43 / 2/4** |

**Free by-product, and a good check on the whole method:** `strat5` estimates the c11/c12 intercept
shift between the pre-v841 and v841 anchor pools as **−0.440% (c11) / −0.252% (c12)**. That is an
independent re-measurement of the promoted x2 from anchor runs alone, and it lands on top of x2's own
8-pair estimate (−0.361% / −0.265%). The promotion is confirmed from a second direction.

**Verdict on x13 at n=8: DOES NOT CLEAR — and the reason is not just the z bar.**
- c11 fails outright in every reduction (z −1.36 to −1.87, 62–75% negative). The point estimate is
  stubbornly ≈ −0.13% in all four reductions, so there may be a small real gain, but it is not established.
- c12 **straddles** the bar (−3.01 single-sided vs −2.83 paired) and, more importantly, is
  **subgroup-unstable**: the c12 effect is −0.51% on the 4 v840 pairs and only −0.06% on the 4 fresh
  v841 pairs. Difference 0.45% ± 0.20%, z ≈ 2.2 — not decisive on its own (and post-hoc), but it means
  the pooled c12 number is not a trustworthy single estimate. Sort and quantizer are separate kernels
  with no plausible interaction, so the most likely reading is that the v840 c12 run of 4/4 negatives
  was a favourable fluctuation.

**Decision on the x14 branch.** The coordinator gated `x14_sort_nw2` on "x13 clears on at least c12".
That condition is exactly at its boundary, and taking it as met would mean bracketing an axis whose
centre point is not established. Spending the same 6 submissions on **3 more x13b pairs on v841** answers
the question that actually blocks the decision — does the c12 gain reproduce on the current production
base — so that is what the remaining budget goes to. `x14_sort_nw2.py` (sha12 8a31ffaf1010) is built,
diff-verified and ready if x13 firms up.

### vpair5 — x13b_sort_nw4_v841, cand 146874 / anchor 146875
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9600 → 0.9700, dM=+0.0198; paired D = +0.0014 ms (+0.145%), z=+0.53; pts 85 — not crossed.
- c12: tk 1.5640 → 1.5860, dM=+0.0198; paired D = −0.0024 ms (−0.156%), z=−0.54; **pts 85** on a high-tb
  draw (well past the 84 target line, but tb-driven, not a tk effect).
v841-only running total (n=5): c11 −0.074%, z_of_mean=−0.61, 3/5 negative;
c12 −0.080%, z_of_mean=−0.62, 3/5 negative. The fresh-base evidence keeps pointing at ~zero on c12.

### vpair6 — x13b_sort_nw4_v841, cand 146877 / anchor 146878
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9740 → 0.9610, dM=−0.0160; paired D = −0.0060 ms (−0.618%), z=−2.22; pts 85 — not crossed.
- c12: tk 1.5920 → 1.5580, dM=−0.0160; paired D = −0.0142 ms (−0.891%), z=−3.11; pts 83 — not crossed.

### A bias check that changed how the pooled number should be read
Across the 10 x13 pairs, the paired D correlated with the pair's own machine drift dM:
**r = +0.52 on c11 and +0.81 on c12.** If the shared slope b were right and dM balanced, that should be ~0.
Two things were checked:
1. **Is b attenuated?** Re-estimated b directly from 61 anchor–anchor null pairs, through the origin:
   c11 0.4289 vs 0.4358 from the anchor regression, c12 1.2567 vs 1.2367. Agreement to 2%, and the null
   residual sd is identical either way. **b is not the problem** — the null pairs show no dM correlation.
2. **Is dM imbalanced across the x13 pairs?** Yes, mildly: mean dM = −0.0026, i.e. the candidates sat on
   marginally faster draws than their anchors, which with a positive dM–D correlation biases the plain
   mean *downwards* (flatteringly).
So the plain mean of D is not the right headline. `experiments/2026-09-20/strat6.py` regresses D on dM
and takes the **intercept**, which is immune to both the residual slope error and the dM imbalance.

| estimator (n=10) | c11 | c12 |
|---|---|---|
| plain paired mean (`strat4`) | −0.151%, z −1.71, 6/10 | −0.336%, z −3.67, 8/10 |
| single-sided, per-version intercept (`strat5`) | −0.129%, z −1.87 | −0.218%, z −3.01 |
| **dM-adjusted intercept (`strat6`)** | **−0.126% ± 0.104%, t −1.20, 6/10** | **−0.285% ± 0.091%, t −3.12, 8/10** |

The adjustment moves c12 only from −0.336% to −0.285% and tightens the scatter, so the effect survives
the correction rather than being explained by it. All three estimators now agree: c11 ≈ −0.13% and not
significant; c12 ≈ −0.22% to −0.34% and significant at t/z ≈ −3.

### vpair7 — x13b_sort_nw4_v841, cand 146888 / anchor 146889
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9700 → 0.9710, dM=+0.0148; paired D = −0.0055 ms (−0.563%), z=−2.01; pts 85 — not crossed.
- c12: tk 1.5710 → 1.5830, dM=+0.0148; paired D = −0.0063 ms (−0.403%), z=−1.39; pts 83 — not crossed.

## x13 — FINAL, 11 pairs (4 on v840 + 7 on v841). SQNR EQUAL on all 12 in all 11 pairs.

| estimator | c11 | c12 |
|---|---|---|
| paired `strat4`, n=11 | −0.189%, z_of_mean **−2.24**, 7/11 (64%) | −0.342%, z_of_mean **−3.91**, 9/11 (82%) |
| single-sided per-version intercept `strat5`, n=11 | −0.188%, z **−3.07**, 9/11 (82%) | −0.291%, z **−4.67**, 11/11 (100%) |
| dM-adjusted intercept `strat6`, n=11 | −0.181% ± 0.107%, t **−1.69**, 7/11 | −0.324% ± 0.090%, t **−3.61**, 9/11 |
| v841 pairs only, paired, n=7 | −0.221%, z −2.10, 5/7 | −0.243%, z −2.22, 5/7 |

The n=4 subgroup worry recorded earlier has **resolved**: with 7 v841 pairs the fresh-base c12 estimate
is −0.243% (z −2.10), in line with the pooled −0.29% to −0.34%, so the earlier −0.06% was small-sample
noise and pooling v840 with v841 is sound. The dM–D correlation persists (c12 r=+0.76) but the
adjustment only moves c12 from −0.342% to −0.324%, so it is not what is producing the signal.

### VERDICT
- **c12: PROMOTE-CANDIDATE.** z/t between −3.61 and −4.67 on all three pooled estimators (≤ −3),
  82–100% of runs negative (≥ 75%), SQNR identical on all 12 in every pair. Effect ≈ **−0.005 ms**.
- **c11: NOT ESTABLISHED, and not harmful.** The point estimate is −0.18% in every reduction and the
  sign never flips, but the two conservative estimators give z −2.24 / t −1.69, short of the −3 bar.
  Treat c11 as neutral-to-slightly-positive; do not claim a c11 gain.
- Shipping `x13b_sort_nw4_v841.py` (sha12 **5cfd6a80d066**) is therefore justified on c12 alone, and
  costs nothing on c11. It is one gated launch line and does not touch any other case.

The coordinator's condition for the bracket ("clears on at least c12, with c11 not harmful") is met,
so `x14_sort_nw2` now runs for 3 pairs.

## x14_sort_nw2 — bracketing the sort-scatter warp axis (sha12 8a31ffaf1010, v841 + L805 `num_warps=2`)
Axis so far: 16 warps = +0.40%/+0.07% (x10, v840), 8 = shipped, 4 = −0.19%/−0.34%. If 2 is worse than 4
the optimum is 4 and the axis is closed; if 2 is better the sweep continues downward.

### wpair1 — x14_sort_nw2, cand 146893 / anchor 146894
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9760 → 0.9620, dM=−0.0164; paired D = **−0.0068 ms (−0.701%), z=−2.54**; pts 85 — not crossed.
- c12: tk 1.5900 → 1.5620, dM=−0.0164; paired D = **−0.0077 ms (−0.483%), z=−1.69**; **pts 84 — CROSSED**.

### wpair2 — x14_sort_nw2, cand 146899 / anchor 146900
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9700 → 0.9650, dM=−0.0161; paired D = +0.0020 ms (+0.209%), z=+0.75; pts 85 — not crossed.
- c12: tk 1.5850 → 1.5590, dM=−0.0161; paired D = −0.0061 ms (−0.383%), z=−1.34; **pts 84 — CROSSED** (2nd).
x14 running totals (n=2): c11 −0.248%, z_of_mean=−1.26, 1/2 negative;
c12 −0.435%, z_of_mean=−2.14, 2/2 negative.

### wpair3 — x14_sort_nw2, cand 146902 / anchor 146903
SQNR EQUAL on all 12. Both Accepted.
- c11: tk 0.9680 → 0.9730, dM=+0.0202; paired D = −0.0038 ms (−0.392%), z=−1.36; pts 85 — not crossed.
- c12: tk 1.5610 → 1.5890, dM=+0.0202; paired D = +0.0030 ms (+0.194%), z=+0.67; pts 83 — not crossed.

## x14_sort_nw2 — FINAL, n=3. SQNR EQUAL on all 12 in all 3 pairs.

| estimator | c11 | c12 |
|---|---|---|
| paired `strat4`, n=3 | −0.295%, z −1.78, 2/3 | −0.226%, z −1.37, 2/3 |
| single-sided `strat5`, n=3 | −0.157%, z −1.31, 3/3 | −0.259%, z −2.18, 2/3 |

**Verdict: does not clear at n=3 in its own right, and shows no advantage over 4 warps.**

## The sort-scatter warp axis, now fully bracketed
`_sort_scatter_kernel` at c11/c12 geometry (`BLOCK=256`, `E_PAD=32`, grid 512):

| num_warps | c11 | c12 | pairs |
|---|---|---|---|
| 16 | +0.40% | +0.07% | 1 (x10, on v840) |
| 8 (shipped) | — baseline — | — baseline — | — |
| **4** | **−0.19%** | **−0.34%** | **11 (x13/x13b)** |
| 2 | −0.30% / −0.16% | −0.23% / −0.26% | 3 (x14) |

The axis is a **plateau between 2 and 4 warps**, both clearly better than the shipped 8, with 16 worse.
2 shows no measured advantage over 4 on either case, and 4 has 11 pairs behind it against 3 — so
**4 is the value to ship**, and the axis is closed. Mechanistically 8 warps = 256 threads on a
`BLOCK=256` tile is exactly 1 element per thread, which leaves the `tl.cumsum` over the 256×32 `eq`
matrix with no per-thread work to amortise the scan; halving the warps at least doubles it.

## What to ship
`experiments/2026-09-20/candidates/x13b_sort_nw4_v841.py` (sha12 **5cfd6a80d066**) = v841 + L805
`num_warps=4 if (e_pad == 32 and n == 131072) else 8`. Promotable on c12 (≈ −0.005 ms, z −3.6 to −4.7,
82–100% negative, SQNR identical), neutral-to-slightly-positive on c11 (−0.18%, not established).
Do **not** also take x14; it is the same plateau with a quarter of the evidence.
