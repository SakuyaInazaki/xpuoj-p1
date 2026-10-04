# Route-weight branch pruning closeout (2026-09-13)

Scope: public contract, current implementation, and the archived v313-v315/v318 evidence only. No new implementation or submission was made.

## Contract and current path

The public statement defines BF16 router logits, FP32 softmax, exact top-k selection, and renormalization of all selected probabilities (`1-full.md:105-121`). It says only that inputs and weights use distributions typical of real models (`1-full.md:219`); it gives no initialization scale, logit bound, minimum selected weight, or threshold under which a branch may be removed. Timed inputs are regenerated (`1-full.md:223-225`), and acceptance requires SQNR >= 22 dB (`1-full.md:239-245`).

The active replicated path materializes `T*k` route ids and weights (`p1/kernel.py:951-975,1003-1004,5556-5575`), sorts all of them, builds metadata for `T*k`, and sums exactly `k` branches per token (`p1/kernel.py:5577-5585,5600-5610,5941-5946`). It contains no route-weight threshold or branch pruning.

## Exact historical artifacts

| artifact | SHA-256 | threshold | recorded result |
|---|---|---|---|
| `p1/kernel_v313_drop.py` | `ca265f392d462f99602fd877bc068d7837f8e6a4eebafc678cdca9b5edca9eb2` | `(E,k)=(8,2),(96,3): 0.01` | SID 131341; 11/12 passed and all executed SQNR checks passed; c1 TLE. WDIST later showed threshold 0.01 dropped zero branches, so this run establishes mechanism/no-op behavior, not pruning speed. Per-case SQNR/tk is not retained in the bounded record. |
| `p1/kernel_v314_drop.py` | `d664e883af8d1e69391225b5626ee346c2a5fa94a258baa1ac61c995778bb93c` | `(8,2): 0.15`, `(96,3): 0.12` | SID 131367; archived decision says 12/12 TLE after call-2 dummy warmup. No SQNR or timing evidence. |
| `p1/kernel_v315_drop.py` | `e774297faf07241cf38f8913e50ca6805a6d5b8f8b8261a54768ea33612a2daa` | `(8,2):.15,(32,4):.1,(64,8):.055,(96,3):.12,(256,8):.06,(32,2):.2` | Built but no matching SID or terminal result was found. SQNR, speed, and failure signature are unknown. |

## Decisive follow-up

WDIST SID 131332 measured nearly uniform selected-route weights: medians clustered around `1/k`; at threshold 0.01 the dropped fraction was zero. The archived error-budget estimate allowed only about 2-3% of rows to be removed.

The same six-threshold table was ultimately exercised by v318, SID 131448. It falsified the proxy error model by 2.5-3.5x: c1 lost 0.87 dB, c3 lost 0.96 dB, and c5 lost 2.43 dB to **21.37 dB**, below the required 22 dB. Its `.item()+zero_` machinery added about **0.19 ms per call** on c2 (`8.96` versus `8.77 ms`), while the estimated saving from dropping 2-3% of rows was at most **0.05 ms**. c1 also TLE'd across v313/v314/v317/v318; the archive did not isolate that failure to pruning math, and c3/c4/c6 silent failures were not autopsied.

## Status

Route-weight branch pruning is closed on the archived uniformly routed workload and implementation family. This is an evidence-bounded project decision, not a theorem for other models, route distributions, compiler versions, or implementations. Reopening requires materially different measured route distributions or a new cost/error mechanism; the public phrase “typical numerical distribution” is not such evidence.

Primary source: `archive/handoffs/handoff-session-20260821/session-notes.md:515-562`.
