# Native weight-cache contract

Date: 2026-09-12. Source/history audit only; no platform probe or implementation change.

- The problem guarantees only that `gate_weight`, the three expert weights, and `topk` keep the same values from warmup through timed runs **within one testcase**; they may change after switching testcase (`1-full.md:225`). Inputs are read-only to the submission (`1-full.md:193-199`). It does not guarantee stable Python `Tensor` object identity.
- A minimal object-replacement-safe entry can retain strong references to every source weight and accept a hit only when every live argument `is` its saved reference, together with matching shape, stride/layout, dtype, device, rank/world, `topk`, and quantization granularity. Strong references prevent Python `id` reuse from making a stale entry look live.
- Replacing a tensor object therefore causes a safe miss. Mutating the same object in place is invisible to `is`; the within-testcase static/read-only contract excludes contestant-side mutation there, but no source proves that a harness could not reuse one object and overwrite it while changing testcase. That cross-testcase case remains unsupported by identity alone.
- SID 141451/141457 logs contain no object-identity trace, so their observed rebuild/OOM behavior cannot distinguish wrapper proxy/clone replacement from another cause. No evidence establishes stable identity across repeated calls.
- Performance risk: if the harness presents a fresh proxy/tensor object for each timed call, the safe identity guard rebuilds native quantization every call. The scheme is correctness-conservative but has no proven warmup-hit performance contract.

Decision: defer cache integration and any control probe unless the packed native benchmark first clears its fixed speed threshold.
