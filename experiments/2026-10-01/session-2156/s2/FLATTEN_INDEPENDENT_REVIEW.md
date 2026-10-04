# S2b independent read-only review

Candidate `p1_s2_c4_flatten.py` SHA `c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b` and parentlayout-only `ac338895…` independently rehashed. **No identified additional source-contract blocker.** This is not GPU correctness or performance approval, and no submission is authorized by this review.

Independent full-AST reversible proof: the only differences are outer `tl.range` in `_fgs_t1i_mdq_kernel_g_s2_pad` and `_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad`. Each replaces sole keyword `num_stages=2` with sole `flatten=True`; reverting both restores the entire parent AST, including all top-level statements and duplicate definitions. Inner K loop, numerical bodies and all other functions are identical.

Consequently c4MDg hoststage3, pre_nf hoststage4/maxnreg232, pairedDNstage3/FLATTrue are unchanged; all tiles/warps/grid, DROP store, compactSCL/AH/WI, paddedACT, CSCL/Down/INV_PAD/fin, logicalM/P and actual-producer flag/phase gating match the parent. No new stream, synchronization, cache orlaunch. Parent4157tile/524288row CPU address contract is directly reusable; no batch rerun was needed.

Flatten is an unmeasured scheduling change; compiler resource pressure, async TMA scheduling, fullSQNR/determinism and500-second first-compile behavior remain open. It does not enter R3 `_nosc` Down/rebuilt-fin contracts or repeat the missing-A_SCALE defect. Missing R3 diagnostics still cannot exclude a compiler-level issue by analogy alone.

Selection condition remains **parentS2 complete12AC first, preferably with normal target-case signal≥2%**, then root may allocate one isolatedS2b test. No automatic extra budget or stage/warp sweep. Parentno-signal leaves only the speculative persistent-loop scheduling rationale in FLATTEN_PROPOSAL.md; this review does not promote that rationale to measured benefit. Candidate remains unsubmitted/root-directed alternative.
