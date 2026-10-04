# 2026-09-20 knobs11 — counting-sort aux launch knobs per geometry, on the v842 anchor

Anchor = unmodified `p1/kernel.py` **v842** (sha256 `5cfd6a80d066d8a729ff1b6741541c1c70df7df75eeffe40c67440d4fe962a3c`).
Each candidate = v842 + exactly one gated launch-kwarg change; `diff p1/kernel.py <cand>` verified.

## Geometry produced by `_counting_sort_order` (read from the host, L750-762)

`n = T*k`; `e_pad = next_pow2(E)`; `block = clamp(16384//e_pad, 64, 256)`; `C = ceil(n/block)`;
`C_PAD = next_pow2(C)`. `E` from `_KNOWN12`.

| cases | T | E | k | n | e_pad | BLOCK | C (= scatter/hist grid) | C_PAD | `eq` tile |
|---|---|---|---|---|---|---|---|---|---|
| c1/c2 | 16384 | 8 | 2 | 32768 | 8 | 256 | 128 | 128 | 256x8 |
| c3/c4 | 16384 | 32 | 4 | 65536 | 32 | 256 | 256 | 256 | 256x32 |
| c5/c6 | 8192 | 64 | 8 | 65536 | 64 | 256 | 256 | 256 | 256x64 |
| c7/c8 | 16384 | 96 | 3 | 49152 | **128** | **128** | 384 | 512 | 128x128 |
| c9/c10 | 4096 | 256 | 8 | 32768 | 256 | **64** | 512 | 512 | 64x256 |
| c11/c12 | 65536 | 32 | 2 | 131072 | 32 | 256 | 512 | 512 | 256x32 |

All six `(e_pad, n)` keys are distinct, so each gate hits exactly one case pair. Note c7/c8 pad E=96
up to **128** (the brief's open question), and that c9/c10 share `n=32768` with c1/c2 — the gate must
therefore test `e_pad` as well as `n`, which every candidate below does.
`_sort_scatter_full_kernel` (the `e_pad == 16` branch, L769) is **dead** for all 12 judged cases.
`_counting_sort_order` is reached by all 12 on timed calls (c1/c2 via the `_CALLN[0] >= 2` arm at L6086).

## Candidates

| cand | change | gate | sha12 |
|---|---|---|---|
| s11_a_scatter_nw4_c34 | `_sort_scatter_kernel` nw 8->4 (L805) | `e_pad==32 and n==65536` | 1c15c135be0a |
| s11_b_scatter_nw4_c56 | same | `e_pad==64 and n==65536` | 649ecb1b8b1f |
| s11_c_scatter_nw4_c78 | same | `e_pad==128 and n==49152` | fc47830ed8c3 |
| s11_d_scatter_nw4_c910 | same | `e_pad==256 and n==32768` | 5992a0dfd170 |
| s11_e_scatter_nw4_c12 | same | `e_pad==8 and n==32768` | e0630371f679 |
| s11_f_hist_nw4_c1112 | `_sort_hist_kernel` nw 8->4 (L767) | `e_pad==32 and n==131072` | f8f38f857a48 |
| s11_g_hist_nw4_all | `_sort_hist_kernel` nw 8->4 (L767) | none (all 12) | 892282d7207c |
| s11_h_colscan_nw8_c1112 | `_csort_colscan`+`_csort_offsets` nw 4->8 (L788,L793) | `e_pad==32 and n==131072` | 958b06acfa10 |
| s11_i_colscan_nw2_c1112 | same, nw 4->2 | `e_pad==32 and n==131072` | f99b3c295428 |
| s11_j_meta_nw | **VOID** | — | — |

**s11_j is VOID as briefed.** `build_block_row_idx_info_kernel` is launched at L301 and L4141 with
positional args only and **no** `num_warps`/`BLOCK` kwarg on the host (it inherits the Triton default
num_warps=4), so there is no host-exposed knob to step down; the brief's `else VOID` applies.

## Estimator

`scratchpad/knobs11/strat11.py` — the strat4 paired estimator generalised to any touched case set.
`D = (tk_c - tk_a) - b*(M_c - M_a)`, M = mean over control cases `{1,2,3,5,7,8,9,10} \ touched` of
`tk / anchor-pool-mean`. `b` is fitted through the origin on all within-pool anchor-anchor pairs
(pool = v837..v840 / v841 / v842; the old pool is homogeneous for every case **except c4/c6**, whose
dn changed at v839/v840, so c4/c6 use only the v841+v842 pools). Null sd = the same D over those
anchor-anchor pairs. Validated: it reproduces knobs10's x13 v841-only row exactly
(c11 -0.221% z -2.11 5/7; c12 -0.243% z -2.27 5/7 vs the recorded -0.221%/-2.10 and -0.243%/-2.22).

### Measured single-pair noise floor (paired null sd, % of mean tk)

| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.73 | 0.70 | 0.62 | 0.98* | 0.53 | 0.28* | 0.39 | 0.50 | 1.06 | 1.00 | 0.28 | 0.28 |

(* c4/c6 from the thin v841 pool only, 45 anchor-pairs; all others from 6948.)
**Consequence recorded before spending the budget:** clearing `z_of_mean <= -3` on a **-0.3%** effect
needs n >= 10*sd%^2... concretely n >= (3*sd/0.3)^2 pairs = 38 for c3, 96 for c9/c10, 9 for c11/c12.
Within a ~30-pair budget spread over 9 candidates, the -3 bar is only reachable on c11/c12 at -0.3%,
or on the wider cases if the effect is >= 1% (c9/c10) / >= 0.6% (c7/c8). Everything else can only be
screened, not promoted.

## Pair log (SIDs recorded at submission time)

| # | candidate | cand SID | anchor SID |
|---|---|---|---|
| p1 | s11_a_scatter_nw4_c34 | 146912 | 146913 |
