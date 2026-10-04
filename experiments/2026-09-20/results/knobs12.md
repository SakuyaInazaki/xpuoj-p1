# 2026-09-20 knobs12 — token-quantizer `num_warps` sweep, per hidden size, on the v842 anchor

Anchor = unmodified `p1/kernel.py` **v842** (sha256 `5cfd6a80d066d8a729ff1b6741541c1c70df7df75eeffe40c67440d4fe962a3c`,
= v841 + `_sort_scatter_kernel` `num_warps` 8→4 gated on c11/c12).

## What is being swept
Two quantizer hosts, `num_warps` only (`num_stages` stays 1, `BLOCK_H` stays `next_pow2(H)`):

| host | line | kernel | shipped launch | which cases |
|---|---|---|---|---|
| `_gq1p_tok` | L867 | `_gq1p_tok_kernel`, grid `(T,)`, one token per CTA | `BLOCK_H=next_pow2(H)`, `num_warps=4`, `num_stages=1` | c3–c10 (gated by `_GA[0]`, i.e. only calls 3–5 = the timed calls) |
| `_gq1p_tm` | L883 | `_gq1p_tm_kernel`, grid `(T,)`, k sorted copies per token | `num_warps = 2 if H==1024 else 4`, `num_stages=1` | c1/c2 (H=4096, k=2) and c11/c12 (H=1024, k=2); also the *untimed* calls 1–2 of c3–c10 |

Gate exclusivity (verified against `_KNOWN12` and `_GASET` at L18/L46):
`_GASET` = exactly c3…c10, so `_gq1p_tok` is reached only by those eight cases, and only while
`3 <= _CALLN[0] <= 5`. Hidden sizes: c3/c4 H=2048, c5/c6 H=3584 (`BLOCK_H`=4096), c7/c8 H=4096,
c9/c10 H=4096. So `H==4096` on the **tok** host touches c7–c10 only (c1/c2 never reach that host),
`H==3584` touches c5/c6, `H==2048` touches c3/c4. On the **tm** host, `H==4096` touches c1/c2 for
timing; it also changes the untimed warm-up calls of c7–c10, which cannot move their timed numbers
and adds no extra specialisation (one `num_warps` value per H either way, so c1's 500 s compile
budget is unchanged).

## Candidates (each = v842 + exactly ONE line; `diff` verified one line for all ten)

| cand | change | gate | sha12 |
|---|---|---|---|
| s12_a_tok_nw8_h4096 | `_gq1p_tok` `num_warps` 4→8 | `H == 4096` (c7–c10) | 900e72232c4b |
| s12_b_tok_nw16_h4096 | 4→16 | `H == 4096` | 3e5f2f71415c |
| s12_c_tok_nw2_h4096 | 4→2 | `H == 4096` | deb3d36972c2 |
| s12_d_tok_nw8_h3584 | 4→8 | `H == 3584` (c5/c6) | 661d6b73533a |
| s12_e_tok_nw2_h3584 | 4→2 | `H == 3584` | 7e57257173a9 |
| s12_f_tok_nw8_h2048 | 4→8 | `H == 2048` (c3/c4) | 3bf0733e8e6f |
| s12_g_tok_nw2_h2048 | 4→2 | `H == 2048` | b9b3012fba52 |
| s12_h_tm_nw8_h4096 | `_gq1p_tm` `num_warps` 4→8 | `H == 4096` (c1/c2) | 422d5e59900c |
| s12_i_tm_nw16_h4096 | 4→16 | `H == 4096` | 6684af39dc23 |
| s12_j_tm_nw2_h4096 | 4→2 | `H == 4096` | df6505c8d45e |

## Estimator
`experiments/2026-09-20/strat_knobs12.py` (working copy in the session scratchpad `knobs12/strat12.py`).
Machine index M = mean over {1,2,3,5,7,8,9,10} **minus the candidate's touched cases** of
tk_i / pooled-anchor-mean_i. Anchor pools respect the per-case code-identity rule: c4 and c6 regress
only on v840+/v839+ anchors (`knobs6_data/v841.txt` + the v842 anchors logged below); every other
target case pools all v837/v838/v841/v842 anchors, since their code is byte-identical from v837 on.
c11/c12 are never a target here and are always outside M.

Pre-registered pooled residual sd (before any knobs12 run, n=128 / n=10):

| case | resid sd | quantizer share of tk (roofline) |
|---|---|---|
| c1 | 0.521% | ~1.8% |
| c2 | 0.493% | ~1.0% |
| c3 | 0.432% | ~3.0% |
| c4 | 0.693% (n=10 pool) | ~5.0% |
| c5 | 0.386% | ~1.0% |
| c6 | 0.276% (n=10 pool) | ~2.2% |
| c7 | 0.233% | ~2.8% |
| c8 | 0.345% | ~4.7% |
| c9 | 0.747% | ~0.6% |
| c10 | 0.706% | ~0.8% |

So c8, c4, c3, c7 and c6 are the cases that can actually resolve a quantizer-sized effect; c9/c10/c2
cannot at any affordable n and are reported for completeness only.

Verdict rule (per touched case): z_of_mean ≤ −3 AND ≥75% of runs negative AND SQNR equal on all 12.
Next-integer targets used for the crossing column, each run against its own tb:
c1→80, c2→79, c3→84, c4→86, c5→82, c6→86, c7→82, c8→84, c9→77, c10→78.

## SID log
- pair1 s12_a_tok_nw8_h4096 (touches c7-c10): cand SID=146910, anchor SID=146911
- pair2 s12_b_tok_nw16_h4096 (touches c7-c10): cand SID=146914, anchor SID=146915
