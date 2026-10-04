# P1 c9/c10 bulk expert-parallel feasibility audit

Date: 2026-09-05  
Scope: bounded, read-only source/history audit. No kernel was changed and no platform job was submitted.

## Decision

The old V602 result does **not** disprove four-way expert parallelism as a whole. It disproves its particular branch-dispatch implementation: the hot path sorted `T*k` branch rows, materialized a 128 MiB FP8 send buffer, issued many per-expert puts, synchronized, returned branch rows, and gathered them again. Its 3–4× regression was dominated by those mechanics.

The proposed bulk layout has not been tested on c9/c10 in the evidence reviewed. `kernel_v602.py` contains a close BF16 prototype—gather all ranks' input/routes, compute only local experts, materialize `[world,T,H]` partials, then `reduce_scatter_tensor`—but c9/c10 return through `_run_ep_moe` first, so that generic block is unreachable for them.

However, V702's measured link rate makes the opportunity narrow. A BF16 bulk output is already too expensive. An FP8 input all-gather plus FP8 owner-output exchange has a physical communication floor near 0.51 ms/rank at 200 GB/s and about 0.57 ms using V702's measured directional rates, before route, synchronization, partial aggregation, and quantization. That leaves c9 a small, falsifiable window. c10 should remain disabled unless c9 first demonstrates a clear win.

Recommended action: run exactly one c9 communication-only probe. Stop if the paired median added time for the two bulk legs and their signals is `>= 0.65 ms`, or if any call is nondeterministic/deadlocks. Only build the full c9 path if it is below that threshold.

## What V602 actually tested

The c9/c10 shape is `(T,H,E,k)=(4096,4096,256,8)`, with `I=2048` for c9 and `I=1536` for c10. Each rank holds `Ep=E/4=64` experts.

The active V602 path is explicit in [kernel_v602.py](/Users/sakimi/Desktop/xpuoj-p1/p1/kernel_v602.py):

- Lines 5952–5957 send every `E=256,T=4096,H=4096,k=8` case to `_run_ep_moe` and return.
- Lines 5399–5435 route, counting-sort, quantize, gather `M=T*k=32768` FP8 branch rows, dispatch them by expert, then call a global NVSHMEM barrier.
- Lines 5335–5351 show the old `E*CHUNKS` grid and three puts per nonempty chunk: row payload, activation scale, and routing weight.
- Lines 5451–5463 reorder local results, return BF16 branch rows through NCCL `all_to_all_single`, and sum them at the source.

The retained history records a fully correct V602 run with c9/c10 SQNR 23.80/23.79, but no unambiguous submission ID was retained. Its timings were c9 7.9–9.9 ms and c10 5.0–8.4 ms versus then-current replicated timings near 2.65/2.08 ms. Doubling selected sections from a V602 base of 8.908/5.187 ms gave:

| Doubled section | c9 | c10 |
|---|---:|---:|
| dispatch + barrier | 20.941 ms (`+12.0`) | 8.584 ms (`+3.4`) |
| md GEMM | 8.124 ms | 7.185 ms (`+2.0`) |
| gather kernel | 6.645 ms | 6.478 ms (`+1.3`) |

The dispatch grid nominally permitted `256*8*3=6144` small puts. Reducing chunks and coalescing scalar metadata cut that count to about 512 and improved one c9 observation from 8.908 to 7.914 ms, while c10 remained noisy and slow. This is evidence against fragmented dispatch, not a lower bound for bulk input exchange.

## The bulk prototype is present but unreachable for c9/c10

Lines 6007–6149 of the same file already implement the broad structure now proposed:

1. all-gather `hidden_states` and routing records;
2. select routes owned by the local 64 experts across all `4T` input rows;
3. grouped GEMM with local expert weights;
4. accumulate into a BF16 `[world,T,H]` partial tensor;
5. BF16 reduce-scatter to each source rank.

For `E=256`, this block also selects an int8 local-weight path (lines 6072–6086), and accumulation uses `index_add_` (line 6103). It therefore is not a numerically equivalent ready-made implementation. More decisively, the c9/c10 early return at lines 5952–5957 prevents these cases from reaching it. No reviewed SID establishes c9/c10 performance or correctness for this bulk block.

## Strongest platform evidence

V701 replaced V602's expert-fragmented dispatch with larger peer-contiguous segments, signals instead of a global barrier, local FP8 expert weights, and FP8 return rows. The mechanics are visible in [kernel_v701_ep_works.py](/Users/sakimi/Desktop/xpuoj-p1/p1/kernel_v701_ep_works.py): peer-contiguous dispatch at lines 5591–5630, acquire waits at lines 5633–5662, FP8 expert result return at lines 5665–5694, and local-expert compute at lines 5866–5877.

SID 138845 was Accepted on all 12 cases: `Σtk=33.765 ms`, c9 `3.990 ms` with SQNR 23.13, and c10 `4.511 ms` with SQNR 23.11. The paired replicated baseline was about c9 2.630 ms and c10 1.976 ms. This is the exact accepted EP evidence; it should replace the noisier V602 result when estimating compute and link cost.

SID 138869 instrumented the same path. Its c9/c10 phases in milliseconds were:

| phase | c9 | c10 |
|---|---:|---:|
| route | .385 | .317 |
| count exchange | .244 | .183 |
| host/device metadata | .067 | .149 |
| input quantization | .099 | .109 |
| branch packing | .126 | .125 |
| dispatch | .805 | .769 |
| gate/up | .913 | .730 |
| down | .447 | .376 |
| return | .693 | .718 |
| second wait/metadata | .370 | .009 |
| final combine | .070 | .068 |

Thus V701's measured local-expert GEMM time was 1.360 ms for c9 and 1.106 ms for c10. Its dispatch and return legs were 1.498/1.487 ms. The recorded effective rates were about 167 GB/s outbound and 193 GB/s return, already close to the machine's observed one-direction ceiling. Bulk exchange must win mainly by reducing bytes, not by expecting a much faster link.

## Byte and compute model for the proposed layout

Per rank, bulk expert parallelism does not reduce MoE FLOPs:

- replicated: local `T*k=32768` branches spread over all 256 experts, averaging 128 rows/expert;
- bulk EP: `4*T*k` global branches, of which one quarter belong to the local 64 experts, again 32768 branches/rank, averaging 512 rows/local expert.

The possible compute gain is therefore better GEMM occupancy and weight locality, not fewer arithmetic operations. A local expert's FP8 gate/up/down weights occupy about 24 MiB for c9 and 18 MiB for c10; one expert can fit in the roughly 50 MiB L2, but all local experts occupy about 1.5/1.125 GiB. A fourfold reduction in total expert-weight traffic is an upper-bound intuition, not a guaranteed realized gain. The measured V701 `md+dn` times are the safer estimate.

For `T=H=4096, world=4`:

| item per rank | bytes | transfer floor at 200 GB/s |
|---|---:|---:|
| local FP8 `x` | 16 MiB | — |
| remote FP8 `x` received/sent for all-gather | 48 MiB | .252 ms |
| remote route ids + weights | .75 MiB | .004 ms |
| BF16 `[4T,H]` partial materialization | 128 MiB | local HBM |
| BF16 reduce-scatter remote fraction | 96 MiB | .503 ms |
| FP8 `[4T,H]` partial materialization | 64 MiB | local HBM |
| FP8 owner-exchange remote fraction | 48 MiB | .252 ms |

Consequences:

- BF16 bulk needs about 144.75 MiB remote traffic/rank, a .759 ms physical floor and about .83 ms at V702's measured directional rates. This leaves too little time for routing, quantization, metadata, local aggregation, and GEMMs.
- FP8 bulk needs about 96.75 MiB remote traffic/rank, a .507 ms physical floor and about .57 ms at measured rates. It halves V701's branch-row communication because each input row travels once per peer instead of each of eight routed branches.
- The FP8 layout still has to materialize or stream 32768 local branch outputs (128 MiB at FP8), combine them into a 64 MiB global-token partial, write scales, then read/transfer owner blocks. Those local passes and kernel launches are outside the .57 ms link estimate.

Using the measured V701 GEMMs, even the optimistic lower sums are c9 `1.360 + .57 = 1.93 ms` and c10 `1.106 + .57 = 1.68 ms`, before all auxiliary work. Against 2.630/1.976 ms baselines, the entire remaining budgets are only .70/.30 ms. This does not prove c10 impossible, but makes it a poor implementation target before c9 wins.

## Correctness and determinism constraints

The input may use the already Accepted `_gq1p_tok` FP8 plus per-row scale representation. Routes and scales are dynamic and must be exchanged on every call; they cannot be cached. Static local weight preprocessing may be cached only by a key that includes tensor identity and layout/device/rank invariants and validates the live source references on hits.

The fastest output proposal adds a numerical operation absent from V701: sum this rank's local-expert branches per global token, then requantize that partial to FP8 before owner exchange. V701 instead returns FP8 branch rows and lets the source combine them. The new requantization creates an extra FP8 rounding, and regrouping by owner changes floating-point association. SID 138845's c9/c10 margins around 1.1 dB are evidence only for V701, not permission to assume the new rounding passes.

For deterministic behavior:

- process top-k slots in a fixed slot order within each owner contribution;
- send one contiguous owner block per peer with payload-before-signal ordering;
- have the owner combine rank contributions in fixed rank order;
- avoid `index_add_`/atomics for the numerical reduction and do not rely on an unspecified NCCL reduction order;
- make every rank take the same collective/control path on every invocation, including empty route sets and fallback decisions.

A BF16 partial can avoid the second FP8 quantization but its communication floor is too high. Preserving the exact original top-k summation order across owners would require carrying slot-level contributions or equivalent metadata and would erode the byte saving; the practical candidate should instead pass both official oracles and repeated determinism checks.

## Interface and rule evidence

The accepted V701 SID 138845 called `libshmem_device.putmem_signal_nbi_block`, `fence`, and `signal_wait_until` on the c9/c10 hot path. That is direct evaluator evidence that these NVSHMEM interfaces are accepted in this P1 environment. It is stronger than a historical claim of compliance, though it does not authorize output caching, call-count behavior, validation/timing phase detection, or rank-divergent control.

P1's current official environment is distributed Triton 3.4.0 and official custom tests are unavailable. Any eventual minimal test therefore has to be a normal platform submission handled by the designated platform agent. Historical references to a separate sandbox are not evidence that official custom testing is available now.

## Minimal falsification sequence

1. **Communication-only c9 probe.** Preserve the Accepted baseline result. Add FP8 row quantization and three peer-contiguous `x+scale+route` puts/signals, then exchange three contiguous FP8 owner-sized output blocks plus scales/signals using scratch data. All four ranks execute identical stages on every call. Compare paired c9 timing against an unchanged anchor. This measures the two required network legs without claiming numerical speedup.
2. **Stop rule.** Stop this route if the paired median added time is `>= .65 ms`, if spread is too wide to distinguish .1 ms, or if any repeated call times out or diverges. At .65 ms, c9 has at most `.62 ms = 2.630 - 1.360 - .650` left for route, input quantization, local partial aggregation/requantization, waits, and final combine. c10's corresponding budget is only `.22 ms`, so do not enable it.
3. **One full c9 candidate only if the probe passes.** All-gather FP8 `x` and routes; compute 64 local experts over all `4T` routes; deterministically combine local branches; FP8-quantize the `[4T,H]` partial; send one owner block per peer; combine owners in fixed rank order. Exact shape guard `(T,H,E,I,k,world,Ep)=(4096,4096,256,2048,8,4,64)`; all unknown shapes and capacity failures use the existing correct path. Validate every invocation, both numerical oracles with fresh SQNR margin, and repeat determinism before timing.

This leaves the bulk EP idea as a bounded c9 experiment rather than a general replacement. V602's small-put result is not the reason to stop it; the .57 ms link floor plus measured 1.36 ms local GEMMs is the controlling evidence.
