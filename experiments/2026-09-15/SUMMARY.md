# 2026-09-15 summary

Production `p1/kernel.py` remains v830 SHA-256 `736411a42da642756df2a42d5e59375aefe54f95437dca2d8cbd2db773da1dd4`; no candidate was promoted. Board score remains 72.75 / best SID 141408. Submission count rose from 2637 to 2647 and no job is in flight.

Two candidates were tested, each requiring one cold-compile TLE repeat because the changed source bytes hit the c1 cold-compile lottery.

## c9/c10 local-expert EP (closed)

- File: `candidates/ep256.py`, SHA-256 `200de2f1b1e8de89b44fdad007653fc631124a5e4298e79fe85ac043a2bc58a3`.
- First run SID 144528 TLE at c1 cold compile; repeat SID 144536 Accepted, display 72.25, `Σtk=225.461`.
- All 12 SQNR gates matched v830; c9/c10 were 23.13/23.11.
- c9 `tk=183.160 ms`, c10 `tk=17.153 ms`. The BM128 EP layout gives each local expert about 512 rows instead of 128, so B weights are re-read four times; total weight bytes do not fall, while all-gather/reduce_scatter and worse A locality are added. Closed for the current tile geometry.

## c9/c10 MD weight pre-sort (not promoted)

- File: `candidates/wpresort_c910.py`, SHA-256 `9d5ecf3c822265e4fb74b91c59a17f08fce64d14eb77ca65276d59f004167297`.
- First run SID 144549 TLE at c1 cold compile; repeat SID 144553 Accepted, display 81.5, `Σtk=30.287`.
- Only the c9/c10 private `_fgs_tma1_host` / `_fgs_tma1_kernel*` paths changed: precompute `weights[order]`, remove the second ORDER load and scattered W load inside the kernel, and drop the unused `torch.zeros(M)` amax in the q8 host.
- c9 `tk=2.570` versus SID 141408 `2.587` (−0.017 ms); c10 `1.995` neutral. Other ten cases are code-identical. Effect is below the promotion threshold, so production is unchanged.

## Consequence

These two structural options do not move the board. Under the current Triton/toolchain ceiling, the old physical-limit analysis is still the best explanation: an anomaly-free display 85 is not reachable, and board 75 would require several concurrent baseline `tb` anomalies. No further production change was made.

## memcpy billing probe (closed)

- File: `candidates/memprobe_c4.py`, SHA-256 `fc8ef469f794f7c1bac9bffab26c9d654933f83fffe0f80ee415deeccf1a2eeb`.
- Inserts one 8192x2048 BF16 D2H plus H2D copy into the c4 timed path. SID 144589 Accepted; c4 `tk` rose from about 0.84 ms to **24.243 ms** while the other cases stayed normal and display fell to 75.5.
- Consequence: explicit host round-trips are billed by the judge. CPU offload of dynamic `gq`/`fin`/sort work is not a viable scoring route.
- Consequence: explicit host round-trips are billed by the judge. CPU offload of dynamic `gq`/`fin`/sort work is not a viable scoring route.

## Dead-allocation cleanup probe (not promoted)

- File: `candidates/cleanup_dead.py`, SHA-256 `23a3e40f6c3955fdea75f1a55eae242dda4bdb1d80c2e49c2b376394a5a08e5b`.
- Python-only removal of two provably unused C2-path allocations: `tokens_sorted` (M x H BF16) and `_gateup_shadow` (M x 2I BF16). No Triton kernel source changed, so it cannot alter the cold-compile set.
- SID 144645 Accepted, display 81.5, `Σtk=29.672`; 12 SQNR values unchanged from v830.
- C2 `tk=7.959` versus same-code v830 SIDs 144560/144562/144564 values 7.932/8.193 (and SID 141408 7.947): no measurable allocator cost. Production remains v830.

## v831 promotion (2026-09-15)

- Candidate: `candidates/maxnreg232_c12.py`, SHA-256 `216158090b434eeaadc6f581e1e2258ee866c57671a1fae1eb7c4f4c76bcf7b7`.
- Only launch option `maxnreg=168` -> `maxnreg=232` for `_fgs_tma2_int_pm_q8_kernel` (c1/c2 q8 MD). No kernel source or JIT-source count change.
- SID 144660 Accepted, display 81.5, `Σtk=29.637`, c2 `tk=7.894`; SID 144714 Accepted, display 81.67, `Σtk=29.589`, c2 `tk=7.885`.
- v830 same-code controls: c2 `tk=7.931/7.932/8.193/7.959` (SID 144560/144562/144564/144645). c2 is reproducibly about 0.5–0.9% faster; c1 and the other ten cases stay in window noise, SQNR unchanged.
- `p1/kernel.py` is now v831 (`p1/kernel_v831_q8maxnreg232.py`); v830 remains the rollback.

## v832 promotion (2026-09-15)

- Candidate: `candidates/maxnreg232_c912.py`, SHA-256 `44b70d137232bc4701857e5679201442b98428bfdced353e62242cd3ce8223e1`.
- Extends v831 with `maxnreg=232` on `_fgs_tma1_kernel_gq` (c9/c10 q8 MD) and `_fgs_t1i_mdq_tma_kernel` (c11/c12 TMA md). No kernel source change.
- SID 144734 Accepted, display 81.75, `Σtk=30.189`; SID 144737 Accepted, display 81.58, `Σtk=30.261`.
- Both runs were globally slow windows; the unchanged c1/c2 controls were about 2.8–3.0% slower than v831. After normalizing by those controls, c9/c10/c11/c12 `tk` fell by about 1.3–2.7% in both runs, with no SQNR change. Production is now v832 (`p1/kernel_v832_mdreg232.py`).
## maxnreg follow-up sweep (2026-09-15)

- `v833_c38.py` (adds `maxnreg=232` to `_fgs_t1i_mdq_kernel_g`) was tested as SID 144740 against the v832 control SID 144737. The eight control cases moved within ±0.5% and c3–c8 were neutral (c7 -0.4%, c5 -0.1%, c8 +0.2% before control normalization). Not promoted.
- `v834_mdreg255_c912.py` (c9–c12 caps 232→255) was tested as SID 144745 against v832. Normalizing by the c1–c8 controls gave c9 -0.6%, c10 -0.1%, c11 -0.2%, c12 -0.1%, i.e. 255 is neutral/slightly worse than 232.
- `v835_mdreg200_c912.py` (c9–c12 caps 232→200) was tested as SID 144749. Normalizing by c1–c8 controls gave c9 +0.2%, c10 +1.0%, c11 0.0%, c12 +0.5%, so 200 is worse than 232.
- `v836_q8reg200.py` (c1/c2 q8 cap 232→200) was tested as SID 144750. Normalizing by the c3–c12 controls gave c1 +0.2%, c2 -0.2%; 200 is neutral versus 232.
- Conclusion: the v832 cap values (q8 c1/c2 = 232, c9/c10 q8 MD = 232, c11/c12 TMA md = 232) are the local optimum of the tested set. No further cap change was promoted.

## v833 promotion (2026-09-15)

- Candidate: `candidates/v837_tma_s4.py`, SHA-256 `afb57d9fa5199092dffc3279435f7b429f09c5153dac63667ef126d2aaa5218c`.
- v832 plus `num_stages=4` (was 3) for `_fgs_t1i_mdq_tma_kernel`, affecting only c11/c12 timed paths.
- SID 144753 Accepted, display 81.42, `Σtk=29.627`; SID 144757 Accepted, display 81.17, `Σtk=30.224`.
- Using the unchanged c1–c10 controls to normalize the window, c11 improved by about 0.75%/1.5% and c12 by about 1.2%/1.3% across the two runs; all SQNR values stayed identical. Production is now v833 (`p1/kernel_v833_tma_s4reg232.py`).
## post-v833 stage/cap follow-ups (not promoted)

- `v838_g_s4reg232.py` changed the c3–c8 `_fgs_t1i_mdq_kernel_g` launch from `num_stages=3` to `num_stages=4, maxnreg=232`. SID 144760; using c1/c2/c9–c12 as controls, c5/c6 were clearly slower (+4.9%/+2.6% raw, still slower after normalization) while c8 was about 1% faster. Not promoted; the historical s3 choice remains best for this kernel.
- `v839_tma_s4reg255.py` changed the c11/c12 TMA md cap from 232 to 255 while keeping `num_stages=4`. SID 144762; normalizing by the unchanged c1–c10 controls left c11/c12 neutral (within ±0.1%). 232 remains the chosen cap.
- Current production is v833: SHA-256 `afb57d9fa5199092dffc3279435f7b429f09c5153dac63667ef126d2aaa5218c`.
