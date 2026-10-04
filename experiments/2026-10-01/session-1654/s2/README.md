# S2 c4 layout-only candidate

Base: production v12 `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`.
Candidate: `p1_s2_c4_layout_only.py`, SHA-256 `ac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8`.

Only existing `_run_replicated` changes; all other existing definitions are AST-identical. Six uniquely named helpers append MDg/pre_nf padded ACT producer pairs and the matching static DN input reader/host. The exact c4 shape and existing INV_PAD availability gate candidate selection; cold calls and every other producer stay compact. Only actually executed padded producers return `act_is_padded=True`; DN selection uses that return flag.

Producer ACT has capacity `P=128*((logical_m+127*E+127)//128)`, while scale/AH/WI remain logical compact rows. `logical_m=T*k` is passed explicitly to both producers and the paired DN host. For c4 ACT grows from 65536 to 69632 FP8 rows (64 to 68 MiB). All ACT tiles use padded TMA store; MDg retains compact scale writes and pre_nf retains DROP. DN changes only ACT descriptor read row, preserving compact A_SCALE, padded Down/DSCL, INV_PAD and final gather. MDg outer stage2/launch stage3, pre_nf outer stage2/launch stage4/maxnreg232 and DN stage3 remain unchanged; numerical expressions and operation order are AST-verified.

Reproduce with `python3 build_s2.py` then `python3 check_contract.py` from this directory. `candidate.diff` and `contract_results.json` hold review evidence. CPU checks cover nine histograms, empty experts, all-empty, first/last concentration, many 127/128/129 boundaries and random dense/skew distributions: 4157 full tiles do not overlap, 524288 effective row mappings are unique and covered, capacity is sufficient, compact scales pair with padded ACT, and existing INV_PAD semantics recover each original branch. Swizzle and 132-program tile enumeration are bijective. Python compilation, final definitions, launch argument/keyword identity and helper call arities pass.

GPU TMA, sandbox acceptance, descriptor lifecycle, SQNR, determinism and performance remain unverified. This candidate is not a measured improvement and has not been submitted by its author. Production and existing candidates were not changed. Submit only through the designated platform executor.
