# dn two-CTA short-K candidate

- Baseline: `p1/kernel.py`, SHA-256 `dd46bdebb7be2eed2f1ebe1106be258789bde35e4ee6c9421d5756b163f426b9`.
- Candidate: `candidates/dn_two_cta_shortk.py`, SHA-256 `38424c0e98a5ebd0a0b0bb40cb5045fcc9bdf2958cc3905e6b3ecf629a30a851`.
- Targets: c4 `(16384,2048,32,1024,4)`, c6 `(8192,3584,64,1024,8)`, c8 `(16384,4096,96,1024,3)`, c11 `(65536,1024,32,1024,2)`.
- Change: the existing FP8-output down kernel uses BM64/BN256/BK128, 8 warps, 2 stages, grid 264 and `maxnreg=128`; all other paths stay at baseline.
- Resource model: the FP32 accumulator is 16,384 registers per CTA (64/thread). Staged A+B storage is about `(64*128 + 256*128)*2 = 80 KiB/CTA`, or 160 KiB for two CTAs. Actual occupancy and spill behavior require platform evidence.
- Metadata: a separate BM64 builder allocates `ceil(M/64)+E` entries. Its first return is the per-expert prefix; down consumes the third return, the block-row prefix indexed by `pid_m`.
- Validation: `python3 experiments/2026-09-08/notes/verify_dn_two_cta.py` passes target/config checks, AST removal back to baseline, boundary counts `0/1/63/64/65`, one fixed random count vector per target, capacity, tail rows and unique logical output coverage. `py_compile` also passes. No local GPU/JIT result is claimed.
