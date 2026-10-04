# S2b flatten minimum delta — proposal only

Base is the frozen layout-only S2 `ac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8`. No candidate implemented, no GPU verification or claimed acceleration.

## Exact two-expression change

1. `_fgs_t1i_mdq_kernel_g_s2_pad`, line7754: change outer `tl.range(pid,total_tiles*num_block_n,num_pid,num_stages=2)` to the same range with `flatten=True` and no range-level num_stages.
2. `_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad`, line7844: make the identical range-keyword change.

Only these two *new padded ACT producers* change. Leave the paired DN outer loop7959 unchanged: it already uses `flatten=FLAT`, and its host passes FLAT=True. Leave the inner `range(0,tl.cdiv(K,BLOCK_K))` dot loops, arguments, q conversion/FP32 expression order, swizzle, row masks, SCL compact writes, padded ACT stores and pre_nf DROP untouched. No added launch or phase gate.

Keep host controls exactly as layout-only: MDg c4 launch **num_stages3**, num_warps8/BM128/BN128/BK128/GROUP_M32; pre_nf **num_stages4,maxnreg232**, same tile/warps/group; paired static DN **num_stages3**, num_warps8/BM128/BN256/BK128/GROUP_M32/FLAT=True. Do not confuse range-level stage2 with host launch stage3/4, and do not broaden stage/warp tuning. Preserve c4 gating, actual-producer bool and explicit logical_m=T*k; all other producers remain compact.

Static review can assert all AST differences are those two outer range keyword lists, and rerun layout/argument checks. CPU address correctness is unchanged; GPU scheduling/async behavior, precision, deterministic repeated output, resource use and full-batch compilation must still be verified. The v12 compact `_fgs_t1i_mdq_tma_pre_kernel` already provides an existing flatten=True implementation family, but this does not establish speed or acceptance of the new padded variants.

## Relation to failed R3

R3 changed **Down output scale contract**: `_dn_tma2_f8_pad_static_nosc_kernel` omitted CSCL, and `_gather_branch_sum_f8_rebuild_kernel` rebuilt dequantization scale using A_SCALE/C_SCALE/INV/INV_PAD/flat expert ids. Earlier R3 invocation omitted A_SCALE; its corrected version then had rank2 exit255 without PTXAS stderr. R3 was not a test of this MDg/pre_nf outer flatten delta.

S2b retains SCL/CSCL and the existing fin path; it does not enter either `_nosc` DN or rebuilt fin. pre_nf's DROP allocation/argument/store are preserved, unlike a descriptor-only simplification. Thus the known missing-A_SCALE/new-fin contract defect does not recur. Missing diagnostics mean one cannot prove no common compiler failure mechanism, so retain whole-batch AC verification.

## Is one more test justified if layout-only has no normal signal?

Flatten is a distinct scheduling hypothesis: it may let loop scheduling overlap the persistent outer tile traversal with the16 inner K blocks at c4 H2048, reducing repeated pipeline drain/fill. Layout-only could be neutral because tails are only≤32 of roughly512 row tiles, while pipeline overhead could affect every tile. This is an independent rationale, **not evidence that the compiler actually produces such overlap or that it is net faster**. Flatten may instead extend live ranges/increase registers or inhibit profitable staging; the retained DROP/SCL stores may constrain optimization.

Given the deadline, existing guide stop condition remains sensible: first require complete AC and normal layout-only signal around≥2%, then optionally take one isolated S2b test with no further sweep. If layout-only has no signal, it is insufficient evidence to spend another late first-compile submission by default; prefer already validated routes and final freeze. A single no-signal-layout→flatten test would be a root-directed exploratory exception supported only by the independent scheduling question, and should have an explicit one-shot budget/compilation-time allowance. No resources/PTX trace in inspected evidence currently strengthens that exception.
