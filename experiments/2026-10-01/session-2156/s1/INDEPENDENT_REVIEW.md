# Expanded S1 independent read-only review

Reviewed frozen `p1_s1_c3567_vector.py`, SHA `0328320d02ba2425fe9904b0817cebea28ebfbbd780efc0729e9e642ca7a987b`. Compared full AST with frozen v12 `08dd08…` and c6 vector parent `d11bf076…`; reused existing CPU contract result, did not rerun its batch or submit/import GPU code. No candidate or production edits.

**Conclusion: no identified blocking source/address/dispatch defect for the four selected shapes.** Online compiler mask lowering, SQNR, determinism, actual timed-path coverage and benefit remain unverified. The inherited static-cache/call-phase concerns are not new regressions and are not evidence that this OJ run fails.

## Independent scope and masks

Against v12, the only changed original final function is `_run_replicated`;185 other originals remain identical. Against c6 vector parent, exactly three functions differ: `_run_replicated`, GQ parameter kernel and its host. MD gather consumer+host are full AST-identical to c6 v2.

Selection6453–6458 uses full `(T,H,E,I,k)` shapes for c3 `(16384,2048,32,2048,4)`, c5 `(8192,3584,64,2560,8)`, c6 `(8192,3584,64,1024,8)`, c7 `(16384,4096,96,2048,3)`. `_GA` must be true. c4/c8/E256 and other producers retain parent behavior.

Kernel7732 receives separate real K_BRANCH and power-of-two K_BRANCH_PAD. Every branch-dependent read has the actual `j<k` mask: INV7745, FLAT_IDS7746, FLAT_W7747, BNORM7748. Every scatter has that mask: AH7755,WI7756,ACT_SCALE7757. For c7 j3 uses masked-other values and performs no memory read/write; even though dest defaults0, it cannot overwrite row0. Pointer arithmetic uses `t*real_k+j`, not padded4 stride. Each valid branch's INV destination comes from the current sorting permutation; all three arrays share that destination.

Q and token SCALE stores7741–7742 occur once outside the branch vector. Host7759–7776 allocates Q[T,H],token SCALE[T],AH/WI/ACT_SCALE[T*k]; `K_BRANCH=k`,KPAD=next_power_of_2(k). GridT eliminates a partial token tile. H2048/3584/4096 is bounded by BLOCK_H next power of two; X/Q memory is masked by offs_h<H. Tail padding cannot alter the max because missing values are0 and reduction is abs max.

Existing `expanded_checks.json` binds to the exact SHA: four scenarios total245760 unique branch destinations and737280 FP32 scalar comparisons. It supports the address/order arithmetic, not GPU FMA lowering or extreme-scale guarantees. No redundant full batch was run during this review.

## Producer/consumer pairing and real dimensions

Current sorting produces ORDER and INV; new producer receives current X/INV/ids/weights/bnorm at6460. `_hybrid` and its arrays reset per call6349–6352, and MD selects the new consumer solely after the new producer sets this flag6597–6602. New `_hybrid` is reached under existing direct-GQ token branch and _GA>=globalcall3, so selected shapes enter the existing q8 MD branch; fallback old producers never pass None/new arrays to the consumer.

MD host7840 derives M from ORDER.shape[0]=T*real_k and K from Q.shape[1]=H; consumer `KTOP=k`7854 uses real3 for c7, notKPAD4. ORDER/real_k at7810 recovers source token. AH/WI loads use compact offs_m with row_mask7825/7831; ACT remains compact[M,I]. Consumer scales returned to DN are ACT_SCALE[M], not token SCALE[T]. Bound/WI association order and original f16x2/math remain parent-identical. I2048/2560 and H2048/3584/4096 are tile128 divisible; B descriptor/gather accesses do not require new column masks.

All four shapes have original `_gg=(G96 and I1024)` false. Thus MD launch stage3,warps8,BM/BN/BK128,GROUP_M32 and outerstage2 agree with their original MDg branch. c7 G96/I2048 must not accidentally use its neighboring c8 stage4/maxnreg232 branch; current code does not.

## Stream and TMA lifetime

No stream API, barrier or NVSHMEM lifetime change is introduced. GQ allocation/producer, MD and existing DN/fin are launched in the same existing stream context. Current-call references retain Q/parameter arrays/ACT until their downstream host submission. MD's B tensor is a view of cached interleaved GU weights and remains alive through its owner; ACT descriptor uses a live new ACT tensor. The consumer keeps original full-tile ACT TMA store and masked pointer tail store, so compact neighboring experts are not overwritten. No descriptor-only/drop-contract rewrite occurs.

This is the existing same-stream allocation/launch assumption; no cross-stream allocator hazard was added. Actual judge stream behavior and TMA completion ordering still require online checks; CPU inspection does not prove hardware synchronization.

## Required online observations

- Whole12-caseAccepted, two SQNR≥22 and two bitwise determinism passes each, with attention to newc3/c5/c7; c3 baseline minimum22.70dB has relatively little margin.
- c7 k3 remains deterministic and finite; mask code generation must not introduce invalid-lane side effects. Error details must be tied to case/rank rather than interpreting any failed sample as performance.
- Fulltk/tb/q vector, sourceSHA and cases beyond selected shapes. Compare c3/c5/c6/c7 normal times to paired normal baseline; do not credit tb inflation or low/zero samples as actual speed.
- `_GA` only globalcalls3–5; missing visible improvement cannot prove the new chain was inactive, and source eligibility cannot prove it was measured. Do not invent timing probes/new launches to diagnose this.
- First-compile500-second batch risk now includes new shapes and KPAD4 specialization. Collect terminal result before deciding another run; no same-SHA duplicate while pending. No new stage/warp sweep follows from this review.
