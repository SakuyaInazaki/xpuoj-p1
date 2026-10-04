# c5/c7 current down-projection replay probe

Date: 2026-09-05

- Baseline: `p1/kernel.py`, SHA-256 `dd46bdebb7be2eed2f1ebe1106be258789bde35e4ee6c9421d5756b163f426b9`
- Candidate: `p1/codex_profile_c5_c7_dn_repeat.py`, SHA-256 `7419e8a267e524df2db5c5d824ee6be0f6dde683be4e1d2af9566654c17f5a78`
- Static verifier: `p1/codex_profile_c5_c7_dn_repeat_verify.py`, SHA-256 `a2106712335220c3ae9ba01422495d8ef1f58825ce019aa26b0796e95ed22c90`
- Scope: local implementation and static checks only; not uploaded and not GPU/JIT tested.

## Exact path measured

Both cases are current known-shape replicated paths. On the existing third-and-later call path, `_q8_act` selects `_fgs_tma1_intq_host`, which returns `act_q8` plus per-row `act_rowscl`. The pre-existing `act_q8 is not None and _FL[0]` branch then calls `_dn_tma2_f8_host`, whose only compute launch is `_dn_tma2_f8_kernel`. The probe adds no call-number condition: its Boolean depends only on the complete public shape, and is passed only at that existing branch.

| case | public shape `(T,H,E,I,k)` | down GEMM `(M,N,K)` | input | weights | outputs | launch |
|---|---|---|---|---|---|---|
| c5 | `(8192,3584,64,2560,8)` | `(65536,3584,2560)` | `act_q`: FP8 e4m3fn `[65536,2560]`; `act_s`: FP32 row scale | `dn_q`: FP8 e4m3fn `[64,3584,2560]`; `dn_s`: FP32 row scale | `down`: FP8 e4m3fn `[65536,3584]`; `_dscl`: FP32 `[65536,14]` | grid 132, BM128/BN256/BK128, GM32, FLAT=true, 8 warps, 4 stages |
| c7 | `(16384,4096,96,2048,3)` | `(49152,4096,2048)` | `act_q`: FP8 e4m3fn `[49152,2048]`; `act_s`: FP32 row scale | `dn_q`: FP8 e4m3fn `[96,4096,2048]`; `dn_s`: FP32 row scale | `down`: FP8 e4m3fn `[49152,4096]`; `_dscl`: FP32 `[49152,16]` | grid 132, BM128/BN256/BK128, GM32, FLAT=true, 8 warps, 4 stages |

`M=T*k`, `N=H`, and `K=I`. The candidate repeats the exact launch with the same descriptors, metadata, inputs, launch constants, `down`, and `_dscl` buffers immediately after the original launch on the same stream.

## Idempotence and precision

`_dn_tma2_f8_kernel` contains no atomic operation. Its tile mapping assigns every logical `(row, 256-column chunk)` to one tile id. That tile performs the same ordered FP8 dot accumulation and stores:

1. one FP32 scale to `CSCL[row, N-chunk]`; and
2. one FP8 e4m3fn output tile to `C[row, columns]`.

The second launch overwrites those locations with the same values. It does not feed the first output back as input, accumulate into it, modify metadata, or change the subsequent fixed-order gather/combine. Thus the returned result and precision path remain the baseline result. No allocation, CUDA Event, explicit synchronization, NVSHMEM operation, call counter, cache, or timing API was added.

## Static verification

`python3 p1/codex_profile_c5_c7_dn_repeat_verify.py` passes and checks:

- the two `_dn_tma2_f8_kernel` launch ASTs are identical;
- the repeated kernel has no atomics and retains exactly its two stores;
- the repeat guard contains exactly the c5/c7 public shapes and no `_CALLN` reference;
- deleting the added parameter, guard assignment, keyword, and repeated launch yields an AST exactly equal to current `p1/kernel.py`;
- counts of Event, synchronization, barrier, and symmetric-allocation markers are unchanged.

Both files also pass `python3 -m py_compile`. This cannot establish Triton 3.4 JIT, PTXAS, GPU determinism, or measured timing.

## Interpretation of a platform delta

The paired `tk(repeat_dn) - tk(anchor)` is the harness-visible incremental cost of replaying this kernel in its real stream context. It includes any launch scheduling, overlap changes, cache warming, and measurement quantization. It must not be reported as an isolated physical kernel wall-clock time. Together with the measured md replay effective increments (c5 about `1.033 ms`, c7 about `0.981 ms`), it can bound how much of current `tk` remains in down projection versus routing, metadata, quantization, and final gather/combine, but subtraction should use a same-window exact-code anchor.
