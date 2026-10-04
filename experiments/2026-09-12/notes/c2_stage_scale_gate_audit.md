# c2 stage diagnostic scale-gate audit

CID `5cf47dd4-dd6b-49b5-b8c7-757640849c90` executed the frozen c2 stage
diagnostic through both GEMMs and its oracle.  `status=0`; MD and DN SQNR were
40.43047 and 35.39511 dB, both above the diagnostic's 35 dB requirement.  It
stopped before timing and resource reporting only because `dn_scale_rel` was
0.0021165172 while the script required at most 0.001.  There are therefore no
MD-only, DN-only, consecutive MD+DN, register, spill, or shared-memory results
from this run.

`dn_scale_rel` is
`sqrt(sum((s_kernel-s_scalar)^2) / sum(s_scalar^2))`.  Each of the three
selected 256-column chunk scales is accumulated once per 128 checked output
columns, so this is equivalent to an equally weighted relative L2 over three
scales.  It is not a maximum per-scale error.  The 0.001 cutoff was introduced
by this diagnostic; the problem statement requires final-output SQNR of at
least 22 dB and contains no 0.1% intermediate-scale contract.

The scale algebra and indexing match the active DN kernel: FP8 dot, multiply
each output column by its DN weight scale, take the full 256-column absolute
maximum, form `s=max(act_rowscl*max/448,1e-12)`, then quantize
`z*act_rowscl/s`.  No formula or chunk-address error was found.

Official Triton 3.6.0 gives [`tl.dot` a default
`max_num_imprecise_acc=None`](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L1845-L1897).
For FP8 x FP8, the [semantic layer replaces
`None`](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/semantic.py#L1409-L1419)
with the backend default, and the [CUDA SM90 backend sets that default to
`2**30`](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/compiler.py#L164-L185).
The active DN and diagnostic GEMM omit this argument, whereas the independent
oracle first converts both FP8 inputs to FP32 and reduces `a*b` with `tl.sum`.
Those are different accumulation contracts.  This is a supported possible
source of the observed 0.21165% scale difference, not proof that it is the only
cause.

Root has paused any gate repair or rerun while reviewing the user's new
material in `/Users/sakimi/Desktop/xpuoj-add-info`.  The frozen candidate and
production kernel remain unchanged.
