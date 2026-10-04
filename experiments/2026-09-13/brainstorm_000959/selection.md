# Chair screening decision

Date: 2026-09-13

This file records the root chair's decision after reading the four generated Markdown outputs. It does not select an implementation from the panel, modify `p1/kernel.py`, or authorize an XPUOJ submission.

## Run outcome

- Process success: 4/4 lanes.
- Format contract: 4/4 outputs had the required section markers.
- Format compliance is not evidence acceptance. Several claims failed semantic checks against the current implementation or supplied records.
- Total provider usage: 57,997 tokens (54,887 input, 3,110 output, 0 reasoning), reported cost approximately USD 0.017519.
- Result: no panel candidate is directly implementable after evidence gating.

## Candidate screening

### Deferred or retained only as an unverified direction

- `AT-1` and `cross-domain-translator B-1` form one DN/final-reduction fusion cluster. Earlier k=2 BF16 atomic work has a negative record, and these proposals do not supply a deterministic reduction mechanism that changes that failed premise. Do not select them now.
- `AT-2` identifies a real correctness gap: shape-only derived-weight caches can reuse stale data across test points with identical shapes and changed weights. Retain this as required correctness work, not as a score-improvement candidate. A fix must not infer test-point identity from evaluation or warmup timing.
- `constraint-opportunist L-2` and `cross-domain-translator A-1` form an offline quantization/layout-balancing cluster. They do not explain which current scale instructions would disappear or how the transformation preserves the required computation while reducing timed work. Retain only as an unverified hypothesis.

### Rejected by evidence or implementation semantics

- `constraint-opportunist L-1`: rejected because the current route already groups rows by expert; the claimed missing local expert bucketing premise is false.
- `mechanism-builder A-1`: rejected because the active implementation already applies row scale outside/fused into DN use, and the proposal gives no executable mechanism that removes the intermediate tensor while preserving the mature FP8 path.
- `mechanism-builder A-2`: rejected because activation/output scale varies with input data; treating it as a rank-local static value is a false premise.

## Score gate

P1 scores 12 cases equally through integer per-case scores. Candidates must show plausible improvement across multiple cases or a sufficiently large percentage improvement in affected cases. Large absolute milliseconds in c2 alone do not justify selection; a 10% c2 MD improvement is only about one integer case-score point and is insufficient to reach board score 75.

## Separate audit decision

After a separate official-format audit, the root chair selected one bounded legacy INT4 MMA upper-bound probe. This probe is not a panel winner and is not evidence that INT4 will improve P1. Its throughput on the actual P1 environment is unknown. The probe is only a cheap prerequisite check of whether the relevant legacy format/instruction path exists and has a useful upper bound. Do not extend it into another swarm, critic round, literature search, kernel integration, or XPUOJ submission on the strength of this note.

## Boundary

The raw JSONL files are local runner diagnostics and are not reproduced here. This decision contains no source payload, credentials, hidden-test information, authentication material, or internal reasoning trace.
