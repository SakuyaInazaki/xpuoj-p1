# Minimal static generation design — proposal only

No implementation or submission. Base v12 SHA `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`. This proposal fixes new-object/same-shape transitions; it does **not** prove support for every permitted mutation scheme.

## Entry contract and hook

`1-full.md:223–225` explicitly makes **gate_weight, expert_gate_proj, expert_up_proj, expert_down_proj, topk** static within a test point. X is regenerated after warmup and output must be overwritten. The text says static values remain unchanged, not that Python tensor identities remain unchanged. It also does not promise fresh objects at a case transition. The supplied addinfo P2 question asks about this identity guarantee, but has no answer establishing it for P1.

I found no local copy of the official P1 judge/driver exposing its tensor construction or per-call argument reuse. Local triton-dist operator tests and our candidate scripts are not evidence of the deployed judge's identity policy. Runtime logs reveal warmup/iters/groups but cannot establish object identity.

At the single public `run_kernel` entry, before existing dispatch, maintain one state containing strong references to the four static source tensor objects, snapshots of their shape/dtype/device/stride, current integer topk, and an integer generation. Compare source references using `is`, scalar topk by value, metadata snapshots by value. This uses no tensor equality/content reduction and no X/output identity. Same signature: do nothing. First call or changed signature: clear the affected derived caches once, release the prior signature references after invalidation, and install new strong references/metadata/topk. Increment generation only as cache ownership metadata; it must not decide a computational path.

Use all four static sources even though individual caches depend on subsets. This trades an occasional unnecessary rebuild on router-only/topk change for a small auditable hook and complete invalidation of the present dependency graph. Internal shape keys can remain unchanged, because their cache is scoped to one static generation. A→B→A always rebuilds on each identity transition rather than retaining old generations.

Do **not** reset `_CALLN`, `_GA`, `_FL`, `_EPP`, `_MDL`, or change `_KNOWN12/_GASET`. Existing `run_kernel:6887–6892` continues to update `_CALLN/_FL/_GA` exactly as before. No new warmup/timing inference is introduced. This deliberately leaves phase-coverage concerns as a separate issue.

## Invalidation set (14 dictionaries)

| Cache | Dependency / derived material | Why generation invalidation is necessary |
|---|---|---|
| `_FULL_WEIGHT_CACHE` | all-gathered gate/up/down; concatenated GU BF16 | shape-only key, final helper4546 |
| `_FULL_INT8_CACHE` | full GU int8+scale | full GU source, helper3748 |
| `_FULL_FP8_CACHE` | full GU and Down FP8+scale | shape-only key3769 |
| `_FULL_FP8_LOWMEM_CACHE` | lowmem full GU/Down FP8 | three local expert weights3792 |
| `_FULL_FP8_LOWMEM_TILED_CACHE` | tiled lowmem full GU/Down FP8 | three local expert weights3845 |
| `_FULL_DOWN_INT8_CACHE` | full Down int8+scale | expert_down / full-weight source3919 |
| `_STATIC_INT8_GATEUP_CACHE` | local/concatenated GU int8+scale | derived gate_up3940 |
| `_STATIC_INT8_DOWN_CACHE` | local Down int8+scale | expert_down/topk3954 |
| `_STATIC_FP8_CACHE` | local GU/Down FP8+scale | derived gate_up/expert_down/topk3968 |
| `_STATIC_CACHE` | local concatenated gate_up | expert_gate/up; existing signature4375 includes other static sources |
| `_BNORM_CACHE` | per-expert GU row norm bound | GU FP8+scale6291; shape-only |
| `_INT_GU_CACHE` | interleaved GU FP8 and scales | GU FP8+scale, gran6313; shape-only |
| `_STATIC_DN_SCALE_CACHE` | static Down bound and folded scale | DN FP8+scale, chunk7468; has identities but clear conservatively |
| `_STATIC_DN_TILED_CACHE` | static tiled Down bound and folded scale | tiled DN FP8+scale7650; clear conservatively |

The explicit list should be validated against final top-level cache definitions; earlier duplicate `_STATIC_CACHE/_FULL_WEIGHT_CACHE` assignments are not separate runtime caches. Clearing at entry means no active host helper is holding stale derived values for the current call.

`_AMAX_CACHE` and `_KQ_FP` are **not static-weight caches**. AMAX has dynamic scale writes and only literal-False guarded reads in v12; KQ_FP has no active references beyond its declaration. Optional entry-transition `.clear()` would drop old dynamic residues without changing current reachable behavior, but they need not be included in the minimal 14-cache fix. Do not activate their reuse. No new dynamic cache is added.

## Keep these caches/lifecycles intact

| Cache | Why its present key is safe for static-weight generation changes |
|---|---|
| `_TOKEN_IDX_CACHE` | deterministic arange/repeat tensor determined solely by `(T,k,device)` at580; no input values |
| `_DIRECT_BUF_CACHE` | symmetric communication buffer determined by `(total_rows,H,device)` at57; contents rewritten by communication |
| `_DIRECT_AG_BUF_CACHE` | symmetric buffer determined by shape/device at132 |
| `_ROUTE_AG_BUF_CACHE` | symmetric route scratch determined by n/dtype/device at158 |
| `_SORTED_BUF_CACHE` | symmetric sorted scratch determined by n/dtype/device at207 |

For communication buffers, shape-safety assumes the existing producer fully writes consumed regions; this proposal does not audit or modify that protocol. Do not free/reinitialize NVSHMEM or add distributed initialization/finalization. Metadata/sort/route/ACT/Down helpers already rebuild dynamic values each call, so add no generation caching to them.

## Limits that must remain explicit

1. **New view/object every call:** Python `is` changes even if storage/content are the same, so this design rebuilds all 14 caches every call. That can put all-gathers/quantization in timed calls, increase wall time and exceed the 500-second budget. A storage-pointer key might reduce rebuilds but is not the requested design and introduces alias/mutation/sandbox questions; no documented stable storage guarantee exists here. Retaining a view strongly may keep its base storage alive but does not establish content immutability.
2. **Same object in-place update:** identity and unchanged metadata do not detect new contents. `1-full225` permits static parameters to change across cases; it does not rule this out. Complete handling requires an allowed version or test-boundary signal, or content validation/recomputation. Tensor `_version` is explicitly rejected in the supplied discussion. Identity-only generation is a partial but concrete fix.
3. **Rank coherence:** cache rebuild contains `dist.all_gather`. All ranks must decide to change generation together. If only one rank receives a fresh tensor wrapper while others retain their objects, it may enter all-gather while other ranks hit cache and skip it, causing a hang. The public static contract does not explicitly guarantee coherent object identity changes across ranks. Do not introduce a per-call global collective merely to resolve this without evaluating its cost and authorization/sandbox support. A synchronized judge-provided boundary would be the clean solution, but is unavailable in inspected evidence.
4. **Same-stream asynchronous lifetime:** v12 uses current-stream launches/torch operations. Under ordinary PyTorch same-stream allocation/use, dropping old cache references does not synchronously destroy executing GPU work: the allocator orders reuse on that stream. This is an assumption about the existing integration, not verified judge trace evidence. Different-stream raw Triton use can require `record_stream` or explicit synchronization to prevent allocator reuse hazards; neither API allowance nor need has been established here. Do not add unconditional synchronize/events or retain unbounded generations. Keep the old state live through cache invalidation; source inputs supplied to the current call naturally remain referenced.
5. **Peak memory:** strong references do not copy source tensor payload, but may extend its lifetime until the next generation. Clear all obsolete derived caches before building new ones, and retain only one generation. Even then caching allocator delayed reuse/current in-flight work can temporarily overlap old/new derived allocations. In the replicated ordinary path, full BF16 GU+Down uses approximately `6*E*I*H` bytes, FP8 GU+Down another `3*E*I*H`, and interleaved FP8 GU another `2*E*I*H`, excluding scales, original local inputs, all-gather temporary lists, dynamic activations and workspace. For c4 that subtotal is about704MiB; c6 about2.406GiB. c9/c10 use their lowmem path and should not be estimated with this ordinary-path subtotal. Keeping two generations would approach doubling their retained derived portion, so avoid it. CUDA reserved memory may remain high after clear; report allocated/peak evidence only if a legal diagnostic is available.

## Review recommendation

Proceed only as a separate correctness candidate with a clearly stated guarantee: new static objects/topk or observable metadata transitions invalidate same-shape caches. Preserve v12 math, JIT source bodies, launch configuration, phase gating and communication buffer lifetimes. CPU-only acceptance can cover signature identity stability, A→same-shape B→A invalidation, topk-only/metadata transitions, strong-reference id-reuse protection, exact 14-cache clearing and untouched communication caches. GPU/OJ must establish sandbox acceptance, collective safety, runtime budget and normal performance; a single complete AC does not resolve same-object mutation or fresh-view/rank-coherence limits.
