# P1 bounded brainstorm evidence map

Core question: Which genuinely structural, non-repeated kernel or execution-graph changes could plausibly raise the P1 board score from 72.75 to at least 75, while preserving the exact public contract and avoiding closed configurations?

## Verified facts

- F1: Current board score is 72.75; target is 75. `docs/STATE.md:7`.
- F2: The score includes a 10-point penalty. Board 75 requires displayScore >=85 and the 12 integer case scores to sum to >=1020. `docs/SUBMISSION.md:58`.
- F3: The main comparable baseline SID 141390 is 12/12 Accepted with sum kernel time 30.436 ms. `docs/STATE.md:9`.
- F4: Hardware is one node with four H800 GPUs; each rank owns private local tokens and one quarter of experts. `1-full.md:13,211`.
- F5: For E=8 c1/c2, the active steady path performs no activation collective; each rank processes its own T=16384 tokens and M=T*k=32768 routed rows. `experiments/2026-09-12/notes/c1_c2_rank_compute_comm_audit.md:3-10`.
- F6: On a consecutive same-shape weight-cache hit, c1/c2 communicate zero bytes. `experiments/2026-09-12/notes/c1_c2_rank_compute_comm_audit.md:19-28`.
- F7: The c2 single-GPU stage diagnostic measured stable MD 6.1391/6.1739 ms and DN 2.7455/2.7476 ms; it excludes routing, allocation, quantization, cache/communication and final gather, and is not end-to-end P1 evidence. `docs/STATE.md:40`.
- F8: The established full-density path uses FP8/BF16 mixed precision; prior FP8 core measurements are the mature control, not an untested novelty claim. `1-full.md:227`; `docs/STATE.md:40`.
- F9: Route weights were near-uniform in the historical workload; threshold pruning failed accuracy and cost more overhead than its estimated saved work. `docs/HISTORICAL_NO_REPEAT.md:13`; `docs/STATE.md:50`.
- F10: Inputs are read-only, output must be finite and fully written, SQNR must be >=22 dB, and identical inputs must produce byte-identical outputs. `1-full.md:193-199,239-243`.
- F11: Weights and topk are static only within one test point; after switching test point they may change, so derived caches must update. `1-full.md:225`.
- F12: P1 remote Triton version is unknown. The recorded P1 3.4.0 is historical; a Triton 3.6 single-GPU auxiliary environment is not P1 evidence. `docs/STATE.md:8,21`.

## Gaps

- G1: No current c1/c2 expert-parallel A/B exists; old A2A timings are not contemporaneous controls. `experiments/2026-09-12/notes/c1_c2_rank_compute_comm_audit.md:31-38`.
- G2: Current shape-only derived-weight caches can be stale when successive test points reuse a shape with changed weights. `docs/STATE.md:69`.
- G3: Exact remote compiler/runtime capabilities remain unknown. `docs/STATE.md:8,21`.

## Constraints

- C1: Do not repeat any configuration closed in `docs/STATE.md:47-61` or mechanism/configuration measured negative in `docs/HISTORICAL_NO_REPEAT.md:6-18` unless the idea states a materially changed premise and contemporaneous control.
- C2: Do not request hidden test data or answers. Do not exploit result caches, evaluation timing, data dependence, or platform state. Work must be a legitimate kernel/execution change.
- C3: Do not modify the repository kernel or submit to XPUOJ in this panel. The chair chooses later; implementation is assigned separately to 5.6-sol. Complexity-optimizer is forbidden.
- C4: Every idea needs a minimal local/custom diagnostic, dependency list, and quantitative kill criterion. Parameter sweeps of closed families are invalid.

## Unknowns

- U1: Per-case headroom and bottleneck mix beyond saved public timings may vary; do not invent hidden-case data.
- U2: Whether a proposed compiler feature works on P1 cannot be assumed from local 3.4 or auxiliary 3.6.
- U3: Academic or external novelty is not verified in this local-material-only panel.
