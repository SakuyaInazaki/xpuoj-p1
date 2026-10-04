# Mechanism facts and bounded pseudocode

The public computation is route -> dispatch/group -> expert gate/up matmuls -> FP32 SwiGLU and route weighting -> down matmul -> FP32 branch sum -> BF16 output. Main matmuls must use Triton or Triton-distributed; simple routing operations may use torch. Source: `1-full.md:101-177,249`.

Per rank in the active E8 steady path:

```text
private X[T,H]
route into M=T*k branch rows
MD: two projections, useful work 4*M*H*I -> FP8 [M,I] + row scales
DN: useful work 2*M*H*I -> FP8 [M,H] + grouped scales
gather/sum k branches -> BF16 [T,H]
```

Source: `experiments/2026-09-12/notes/c1_c2_rank_compute_comm_audit.md:3-18`.

For c2 `(T,H,E,I,k)=(16384,4096,8,14336,2)`, useful per-rank work is 7.697 TF for MD and 3.848 TF for DN. Stable stage-only diagnostics are about 6.14 ms MD and 2.75 ms DN, but these omit surrounding stages and are not end-to-end evidence. Sources: `experiments/2026-09-12/notes/c1_c2_rank_compute_comm_audit.md:3-18`; `docs/STATE.md:40`.

Weights may be preprocessed during untimed warmup because they remain static inside a test point. A correct cache must distinguish a new test point even when tensor shapes repeat. Source: `1-full.md:225`; `docs/STATE.md:69`.
