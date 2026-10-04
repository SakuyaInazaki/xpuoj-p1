# P1 Triton 3.4 compile-parameter cross-audit

Date: 2026-09-05. Scope is deliberately limited to the four named candidates and their deltas from the current `p1/kernel.py`. This is a static/source audit; there is no local GPU, so it does not claim Triton JIT, PTXAS, or runtime validation. No candidate source was changed.

## Results by file

| file | SHA-256 | `.cg` + `evict_last` | new inline asm | 3.6-only surface found | disposition |
|---|---|---:|---:|---:|---|
| `p1/codex_int6_c9_g64_bk64.py` | `6f1af0da8056ee6cf69fd6e8a4258a9c14887a224f96b2241deeac97336f6bd3` | none | none | none | No instance of the fusion compile failure. The two identical retries, SID 140230 and 140248, both ended TLE 501 s with `compile.success=true` and no schema/SQNR; archive as lacking GPU evidence, not as a precision or PTXAS failure. |
| `p1/codex_int6_c9_g64_bk128_draft.py` | `9070a0bd2f0de953ba0545b24c6ac61f6e0b84f97afa3e89de8d16e9520fbba0` | none | two `pack=4` decode blocks | none identified | PTX constraint classes and instruction syntax are internally consistent, but the assumed four-lane packing order is not guaranteed by the Triton API. Keep frozen until a GPU sentinel validates layout and compilation. |
| `p1/codex_bulk_c9_comm_probe.py` | `36e094b5f17426de2329dede5148564aa2d8caf6c5877985587c495d02922b40` | none | none | none identified | New surface is NVSHMEM communication plus `tl.pointer_type`; the exact API families were already Accepted in V701 SID 138845 under the P1 environment. Static review cannot prove this larger probe compiles or runs without deadlock. |
| `p1/codex_profile_c5_c7_md_repeat.py` | `fa6923403d48bd278d3ffea0006a943073699d42034c326c2b3cd427aaa8bf4f` | none | none | none | The delta is only a host Boolean and a second launch of the existing md kernel. SID 140251 Accepted, giving direct compile/runtime evidence. |

An AST enumeration found **zero `tl.load` calls with both `cache_modifier` and `eviction_policy` in all four files**. It also found no `cache_modifier` keyword at all. Their `evict_first`/`evict_last` uses therefore cannot emit the fusion branch's unsupported `.cg.L1::evict_last` combination.

Existing `tl.range(..., flatten=...)`, launch `maxnreg=168`, `eviction_policy`, and inline-asm `pack=1/2` uses are inherited from current `kernel.py`, so they are not new Triton 3.6 dependencies in these candidates. The bulk probe adds `putmem_nbi_block`, `fence`, `putmem_signal_nbi_block`, `signal_wait_until`, `NVSHMEM_SIGNAL_SET`, and `tl.pointer_type(tl.uint64)`. V701 SID 138845 exercised these interface families successfully; this is platform evidence for availability, not a proof of the probe's capacity or synchronization correctness.

## Inline assembly audit

BK64, bulk, and md-repeat add no inline assembly. Their calls match baseline constraints:

- FP32 scalar tanh: `"=f,f"`, `dtype=tl.float32`, `pack=1`.
- Packed FP16x2 tanh: `"=r,r"`, `dtype=tl.float16`, `pack=2`.

BK128 adds two identical int6 decode blocks. Each invocation has one packed uint8 result and two packed uint8 operands, so `"=r,r,r"` is the right operand count and 32-bit register class for `dtype=tl.uint8, pack=4`. The PTX body uses ordinary `and.b32`, `or.b32`, `shl.b32`, and `shr.u32` instructions. It consumes both input operands before the final write to `$0`, so possible output/input register aliasing from `=r` is harmless; an early-clobber output constraint is not needed. `is_pure=True` matches a deterministic register-only expression.

The unresolved problem is the **logical lane mapping**, not PTX syntax. Triton's official inline-asm contract says each invocation processes `pack` elements, values smaller than four bytes are packed into four-byte registers, and the exact set of elements assigned to an invocation is unspecified ([Triton documentation](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.inline_asm_elementwise.html), [Triton IR definition](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonOps.td)). BK128 assumes each register contains four consecutive K lanes in little-endian byte positions after `join` and `reshape`: high-nibble inputs `[h0,h0,h1,h1]` and low-bit inputs `[q2,q2,q2,q2]`. That assumption is stronger than the API guarantee and may change with layout propagation.

`check_bk128_draft()` proves the mask/shift algebra only under that assumed register construction. It exhausts all `h0,h1` byte pairs but samples only 8 of 256 `q2` bytes. Testing all 256 is cheap and would strengthen the scalar algebra check, but even exhaustive CPU testing cannot establish Triton's input grouping, byte order, output unpack mapping, TTGIR layout conversions, register pressure, or PTXAS acceptance.

## Minimal gate before reviving BK128

Use a single small Triton 3.4 GPU sentinel containing the exact `join -> reshape -> inline_asm_elementwise(pack=4)` sequence. Feed lane-distinct q4/q2 patterns, store the decoded uint8 tensor, compare every lane with the scalar unpack reference, and retain TTIR/TTGIR/PTX plus PTXAS outcome. Include rows and K positions crossing 4-, 32-, 64-, and 128-element boundaries. Only after this passes should the packed decode be placed beside `tl.dot`; that second kernel is needed to expose dot-layout propagation and resource use.

Given two opaque BK64 TLEs and no BK128 GPU sentinel, neither int6 candidate currently has evidence of a precision failure. They also do not have evidence that the specialized path JITs or completes on P1. The compile-parameter audit supports redirecting current effort to bulk/fusion structure while keeping BK128 frozen.
