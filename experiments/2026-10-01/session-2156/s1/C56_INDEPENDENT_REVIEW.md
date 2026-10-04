# c5+c6 unified independent read-only review

FrozenSHA `3f95092adf1225f59b5e29a5bc7ad56000ca65dfeda02dd882ab6ded17c9db9a` independently verified; parentbefd19 independently verified. **No identified blocking dispatch/layout/argument defect.** No submission, GPU import, candidate/production edit or repeated CPU batch.

Independent full-AST proof: only original `_run_replicated` differs from parent. Exactly two complete `(T,H,E,I,k)` comparisons at6453/6461 expand c6 to c5+c6; reverting both restores the entire parent AST including all helper bodies/top-level statements. No KPAD/k3/mask/helper/stream/stage change is present.

c5 is precisely `(8192,3584,64,2560,8)`. Existing_GA producer and existing_dir_a producer both return tokenQ plus all hybrid parameters; consumer6598 prioritizes_hybrid, so tokenQ cannot enter sorted-input pre_far. Coldcall1/2, all other10known shapes and parentc6 are unchanged. Flags/arrays reset each call. RealM=8192*8=65536, QK=H3584, consumerKTOP8; parameter INV scatter uses realk8. Both GU producer paths return compactACT_SCALE[M], and MD output is compactACT[M,2560]. Existing expanded_checks.json explicitly covers c5T8192/E64/k8 with65536unique writes; because all helpers are identical, its branch/FP32 arithmetic proof is reusable, while GPU validity remains open.

c5 uses the unchanged **dynamic padded Down** branch: `_use_static_pad` is false for E64/I2560/H3584, so DN is `_dn_tma2_f8_pad_host` with compact ACT input and compact A_SCALE; its output/DSCL remain padded and fin uses INV_PAD. The new hybrid selection does not replace DN or reinterpret its input scale. This must not be confused with S2's padded-ACT input contract or c6's static Down path.

Originalc5MDg `_gg=(G96,I1024)` is false; launchstage3/outerstage2,warps8,grid132,BM/BN/BK128,GROUP_M32 match new hybrid consumer. Oldc5pre_far also launchstage3/outerstage2/warps8. No num_ctas keyword or maxnreg override is added by the new candidate; any default/compiler behavior remains inherited. H3584/I2560 are tile128 divisible. DescriptorB remains a view of cachedGU; ACT lifetime and same-stream launch order are parent-identical.

Newly enabled c5_dir_a is **not bitwise-identical to oldpre_far**: sortedQ→tokenQ+gather; consumerI becomes constexpr; tanh f32→f16x2 with input/outputf16 rounding; silu h*(1+th)→h*th+h possiblyFMA; fulltileACT pointer→TMA store, maskedtails retained. C56_UNIFIED.md correctly records these risks. These mechanisms already existed in the parenthybrid consumer, but c5 wholeSQNR/determinism/TMA behavior and500-second new specialization budget need formal verification.

Required online evidence:12casecompleteAC, twoSQNR and determinism passes each; c5 normaltk/tb/q and minimumSQNR versus baseline, c6 regression check, all untouchedcase vector; zero/low anomalies excluded from normal benefit. A c6unified result alone cannot establish c5 correctness or gain. Candidate remains unsubmitted/root-directed alternative; no SID assigned here.
