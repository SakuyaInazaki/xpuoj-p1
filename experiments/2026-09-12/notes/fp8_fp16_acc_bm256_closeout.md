# FP8 / FP16-accumulator BM256 closeout

Date: 2026-09-12. Official-source and bounded history audit only; no implementation or platform action.

- Triton 3.4 and 3.6 both accept FP8 `tl.dot(..., out_dtype=tl.float16)` as an FP16 result ([3.4 semantic.py:1323-1371](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/language/semantic.py#L1323-L1371), [3.6 semantic.py:1336-1398](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/semantic.py#L1336-L1398)).
- Their Hopper lowering selects `WGMMAEltType::f16` from the result type and packs two FP16 accumulator elements into each 32-bit register ([3.4 WGMMA.cpp:35-45,212-286](https://github.com/triton-lang/triton/blob/v3.4.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/WGMMA.cpp#L35-L45), [3.6 WGMMA.cpp:36-45,66-130](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/WGMMA.cpp#L36-L45)). NVIDIA PTX likewise permits FP8 WGMMA `.dtype={.f16,.f32}`.
- This corrects the old suspicion that Triton used FP32 WGMMA plus repeated conversion: official lowering preserves a physical half accumulator, so BM256×physical-N256 on eight warps has about the same 32-bit accumulator-register count per thread as the current BM128×N256 FP32 shape.
- The H800 `bench19a` history nevertheless measured FP16-acc BM128 at 0.80× the FP32 baseline throughput and FP16-acc BM256/BN128/s3 at 0.91×; the desired larger tile remained about 9% slower (`archive/handoffs/handoff-session-20260821/session-notes.md:2569-2580`). Existing full-FP16 accumulator source is `p1/kernel_v618a.py:4860-4866,4921-4926`.
- No exact historical `scale=L2/128` experiment was found. That scale addresses overflow but does not remove the already measured native-half WGMMA throughput loss; FP8 rounding also means the pre-rounding Cauchy value 16384 is not by itself a strict post-quantization bound.

Decision: close the L2-normalized FP8 / FP16-accumulator BM256 route; do not implement or sweep it.
