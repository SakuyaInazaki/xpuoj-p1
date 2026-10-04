# c3+c5+c6 unified independent review

Frozen SHA `75c702478afa8c7d282064fad23d6a542556f30593cae8cb41a201c38da0ea7a` and parent c56 `3f95092a…` independently rehashed. **No identified blocking address/dispatch defect.** No submission, GPU execution, candidate edits or repeated CPU batch.

Independent reverse proof: remove exactly c3 `(16384,2048,32,2048,4)` from the two full-shape hybrid sets; the whole candidate AST becomes identical to parent c56. All helpers, numerical bodies, other shapes, early paths, DN/fin and phase rules are therefore unchanged. c5/c6 inherit their prior reviews; this candidate is not a second automatic test alongside c56.

For c3, true k=4, T=16384, M=T*k=65536, Q[T,H2048], AH/WI/ACT_SCALE[M]. Producer branch addressing uses t*4+j with four lanes, INV uniquely maps each branch to compact position. Consumer KTOP=4 obtains token via ORDER//4; host derives M from ORDER length and K=H2048, not expert intermediate I2048. Both original GA branch and original dir_a branch return fresh hybrid parameters; MD prioritizes hybrid so token Q cannot enter sorted-Q pre_nf. Existing expanded CPU evidence explicitly covers c3 T16384/E32/k4/M65536 with unique mappings; unchanged helpers permit reuse, but not inference of GPU SQNR.

ACT and scale stay compact. Existing c3 INV_PAD/static padded Down/output DSCL and fin remain unchanged; static DN stage4/group32 continue to consume compact ACT input. No S2 padded-input reinterpretation or new stream/cache lifecycle.

**Resource/producer path change needs explicit attribution:** c3 original dir_a producer was TMA-A pre_nf with outer stage2, host stage4/maxnreg232. Unified consumer is gather input, outer stage2, host stage3 with no maxnreg override. Original GA c3 MDg already used gather/stage3; unified coverage now also replaces non-GA dir_a calls. This is a resource/scheduling change in addition to sorted-Q→token-Q traffic, not merely reusing identical execution. All inherited helper launch controls stay unchanged, but which existing helper executes changes.

Original c3 pre_nf math already uses f16x2 tanh and silu h*th+h, with full-tile TMA ACT store plus masked pointer tails. New gather hybrid retains this numerical family and store layout; unlike c5/c6 old pre_far transition, c3 does not newly change f32 tanh to f16x2. Loads/scales are moved/precomputed and consumer I specialization/resource use still can affect GPU compilation/rounding/synchronization; CPU equivalence cannot establish complete AC.

Online acceptance must inspect complete12AC, both SQNR/determinism groups especially c3 minimum historically22.70dB, c3 normal tk/tb and c5/c6 regressions, unaffected cases, and500-second first-compile cost. Anomalous timings are separate from normal gain. Root will choose **c56 or c356** after c6 unified evidence; do not automatically run both or expand stage/warp candidates. Review complete; wait for online results.
