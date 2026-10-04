# c1/c2 MD BM256 closeout

Date: 2026-09-12. Static/source audit only; no platform action.

- The current call-3+ c1/c2 path launches `_fgs_tma2_int_pm_q8_kernel` at `p1/kernel.py:5680`: persistent grid 132, BM128, logical BN128 for each of G and U (two `[128,128]` B-descriptor loads, combined physical GU width 256), BK128, w8/s4, `maxnreg=168`, and default `num_ctas=1`. GROUP_M is 16 for c1 (`I=8192`) and 8 for c2 (`I=14336`).
- The current metadata builder passes `GROUP_GEMM_BLOCK_SIZE_M=128`; MD and DN consume the same six metadata tensors. A BM256 MD launch therefore needs a second metadata set built with block size 256 while DN keeps the BM128 set.
- BM256/w16 raises the per-stage TMA footprint to approximately 64 KiB (A 32 KiB plus G/U 16 KiB each). The current s4 would require 256 KiB, above H800's 232,448-byte shared-memory limit; s3 is the viable historical setting. Its two FP32 accumulators contain 65,536 values, or 128 registers per thread across 512 threads before epilogue and address state, so the current `maxnreg=168` cannot be retained.
- `p1/kernel_v363.py` is the closest executed predecessor: BM256, logical G/U BN128, BK128, w16/s3, grid132, A descriptor `[256,128]`, separate BM256 metadata, and no `maxnreg`. The archived verdict records c1+c2 **+11.15%** versus its baseline and closes the large-tile plane (`archive/handoffs/handoff-session-20260821/session-notes.md:1249-1264`). Its epilogue produced BF16 plus per-N-tile AMAX, so it is not byte-identical to today's direct-q8 rowscale kernel; the GEMM geometry is the same.
- No current-kernel c1/c2 MD repeat probe exists. The accepted repeat probe SID 140251 covers c5/c7 and a different `_fgs_tma1_intq_host` kernel; SID 133030 is an older wall-clock ledger and is not reusable as a current pure-MD measurement.

Decision: close the c1/c2 BM256/w16/BK128 same-geometry route; the executed negative result and current resource constraints are sufficient.
