# c1/c2 per-rank compute and communication audit

Scope: the active call-3+ replicated path in `p1/kernel.py`, with no proposed
change.  SID 141390 uses four ranks.  c1 is `(T,H,E,I,k) =
(16384,4096,8,8192,2)` and c2 is `(16384,4096,8,14336,2)`.

- The E8 dispatch passes each rank's complete local `hidden_states` directly
  to `_run_replicated` (`p1/kernel.py:5517-5525,5983-5988`).  Each rank therefore
  processes `T=16384` private tokens and `M=T*k=32768` routed branch rows; it
  does not process `T/4` and performs no activation all-gather, all-to-all, or
  reduction.  The final result on each rank is BF16 `[T,H]`.
- The steady MD launch produces FP8 `[M,I]` plus one FP32 row scale `[M]`
  (`p1/kernel.py:5670-5688`).  DN produces FP8 `[M,H]` plus FP32 scales
  `[M,H/256]`, then the gather writes `[T,H]` (`p1/kernel.py:4280-4297,
  5814-5823,4338-4357`).  Useful FLOPs per rank are `4*M*H*I` for MD and
  `2*M*H*I` for DN: c1 is 4.398/2.199 TF, total 6.597 TF; c2 is
  7.697/3.848 TF, total 11.545 TF.  Physical GEMM work uses the routed expert
  counts padded independently to BM128, so its exact row count is dynamic and
  bounded by `M <= M_padded <= M + E*127`.
- A full-weight cache miss performs three BF16 `dist.all_gather` calls for the
  local gate, up, and down shards (`p1/kernel.py:4194-4218`).  FP8 weight
  derivation reuses those full tensors and adds no collective
  (`p1/kernel.py:3491-3508`).  With world size 4 and `Ep=2`, the three local
  contributions total 384 MiB for c1 and 672 MiB for c2.  Each rank logically
  receives 1.125 GiB / 1.96875 GiB of remote payload, and the three gathered
  results including its own shard contain 1.5 GiB / 2.625 GiB.  A consecutive
  same-shape cache hit communicates 0 bytes.  Both caches clear to one shape,
  so alternating c1 and c2 evicts the preceding entry and causes another miss.
- SID 141390 saved only end-to-end kernel times, c1 4.746 ms and c2 8.193 ms;
  it has no MD/DN split.  SID 131644 retained only the aggregate c2
  `fgs+dnq ~= 8.98 ms` cross-check.  SID 135553/135554 and 135570/135571 tried
  duplicate/twin GEMM probes, but the fast/slow-call minimum saturated; the
  archive explicitly marks their individual c1/c2 MD and DN values invalid.
- There is no current-code c1/c2 expert-parallel A/B.  Old A2A SID 114465
  measured c1 gateup/down at 7.1/2.9 ms and c2 at 12.9/5.1 ms.  Accepted
  SID 114627 reported c1/c2 totals 16.859/22.996 ms; SID 114629 reported c2
  24.15 ms.  These are old A2A internal comparisons, not contemporaneous
  controls for the current replicated kernels.  The later v285/v286 EP tests
  cover only c9/c10.

The ranks own different tokens.  Replication duplicates full expert weights,
not computation of the same token.  A balanced four-way EP receives routed
branches from all ranks and still processes about `T*k` rows per rank, so the
useful FLOPs above do not fall to one quarter.  No saved c1/c2 result supports
such a compute reduction.
