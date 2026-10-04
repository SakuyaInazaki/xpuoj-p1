# 2026-09-12 public24 custom diagnostic

- CID: `19cd9142-96e7-430c-90c0-8eeca55e8d14`
- Submitted source SHA-256: `8ccf9f3766b6b13fd6f459cd2149bf1bec05718cd0b1b65698591bd02b38e367`
- Request: one `problemId=24`, `GeneratedWorkload`, `triton-h800` custom test; no retry
- Terminal result: `Finished / WrongAnswer`
- Outer compile: success
- Generated-workload result: Triton JIT module compilation failed at the native INT8 dot expression `acc_g = tl.dot(a, bg.T, acc_g)`
- Runtime diagnostics: no `P1MD ` line was emitted; math check, timings, resources, device name, and Triton version have no execution evidence
- P1 production kernel: unchanged

## Triton 3.6 source cause audit

- **Confirmed 3.6 frontend contract:** [`tl.dot`](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L1997) defaults `out_dtype` to `float32`; its [semantic check](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/semantic.py#L1531-L1558) derives an `int32` result for integer inputs, but when `acc` is supplied it requires `acc.element_ty == out_dtype`. The submitted three-argument call supplied an `int32` accumulator without `out_dtype=tl.int32`, which explains the caret-level compilation failure.
- **Minimal successor:** retain the existing three-argument dot and add `out_dtype=tl.int32` to both native G/U calls. This preserves signed INT8 x INT8, INT32 accumulation and the existing loop recurrence.
- The descriptor path preserves the `torch.int8` base dtype in [JIT specialization](https://github.com/triton-lang/triton/blob/v3.6.0/python/test/unit/runtime/test_specialize.py#L73-L76), including signedness in its [descriptor IR type](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L1330-L1341). The Hopper backend has an explicit [s8 x s8 to s32 WGMMA case](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp#L443-L461). The observed failure is therefore a frontend accumulator/output-dtype mismatch, not evidence of a TMA or Hopper integer-WGMMA absence.
- These findings apply to official Triton 3.6.0 source. P1's remote Triton version remains unknown; this custom run supplied no runtime version evidence.

Artifacts:

- `public24_19cd9142-96e7-430c-90c0-8eeca55e8d14_raw.json` — complete API result, SHA-256 `61f8fc1b80393f24e6c20b30ff86e178c8123118d55d4bc27c47213f6766f043`
- `public24_19cd9142-96e7-430c-90c0-8eeca55e8d14_parsed.json` — structured interpretation, SHA-256 `7cc57df21f32da9e77dcee19accee85762424ba46fcb73615931d9db0fad630f`

## INT32 `out_dtype` successor

- CID: `04511ff8-38e2-4c6c-bd5d-027cce65a245`
- Submitted source SHA-256: `0336cd9bec4d40c28c1e6b69ab759c817b0ce1c87da21e672d1b0dc673058728`
- Request: one `problemId=24`, `GeneratedWorkload`, `triton-h800` custom test; no retry
- Terminal result: `Finished / WrongAnswer`
- Outer compile: success
- The two native three-argument `tl.dot` calls with `out_dtype=tl.int32` compiled past the earlier failure point.
- Generated-workload result: Triton JIT compilation failed in `native_math_check_kernel` when `branch_row` read the module global `M`; the frontend requires such a global to be instantiated as `triton.language.constexpr(...)`.
- Runtime diagnostics: no `P1MD ` line was emitted; math check, timings, resources, device name, and Triton version have no execution evidence.
- P1 production kernel: unchanged

Artifacts:

- `public24_04511ff8-38e2-4c6c-bd5d-027cce65a245_raw.json` — complete API result, SHA-256 `fd310da3a26d523e4e91c397a72f1fd259881c1efbc00a088ef93b9d6f104ae7`
- `public24_04511ff8-38e2-4c6c-bd5d-027cce65a245_parsed.json` — structured interpretation, SHA-256 `4981a6727ba1b6b6a8dbdaf7e6e7acb8bec7f3893791aca1af1c252eabe62b12`
- `public24_04511ff8-38e2-4c6c-bd5d-027cce65a245_user_error.txt` — exact returned `userError`, SHA-256 `cbd0db32a871410bd11cff96cdf001cf97df3bf03ae08b5a0a7d1aceb1fd2bf2`

## Explicit check-kernel shape arguments

- CID: `19fd1bbe-71a0-4b9c-b52b-34a11748f78e`
- Submitted source SHA-256: `6c011210268763f8a6ec2cea36821f1a5b1ea8f132c77582e33a2a739cd1fc3e`
- Request: one `problemId=24`, `GeneratedWorkload`, `triton-h800` custom test; no retry
- Terminal result: `Finished / WrongAnswer`; outer compile succeeded; total occupied time was 5.561 seconds
- The check kernel compiled past the earlier ordinary-global access, but LLVM lowering rejected its FP8 comparison: `actual_q != expected_q` became `llvm.fcmp` on `i8` operands.
- Runtime diagnostics: no `P1MD ` line was emitted. The INT8 MD kernel passed its earlier frontend compilation point, but this run supplies no math-check pass or valid timing.
- The API returned the `userError` as a truncated object with `omittedLength=3234`; the raw and error artifacts preserve exactly what the detail endpoint returned.
- P1 production kernel: unchanged

Artifacts:

- `public24_19fd1bbe-71a0-4b9c-b52b-34a11748f78e_raw.json` — complete API detail response, SHA-256 `7e541ea1c09e9b39d9818e90ea3b6564c3409247e9ddea873d9eb7556ceb7a99`
- `public24_19fd1bbe-71a0-4b9c-b52b-34a11748f78e_parsed.json` — structured interpretation, SHA-256 `217bc606bb487ae21b3b49374d2f6ced02ad96b56d799d7c446be3bbfaec5a9e`
- `public24_19fd1bbe-71a0-4b9c-b52b-34a11748f78e_user_error.json` — exact returned `userError` object, SHA-256 `68f44064f7eece1fe1bfb8462e6cd49c063df400640ce7671fd077d356fb5eff`

## FP8 bit-pattern math check

- CID: `ac6c2e16-d9d5-476d-bd94-b0618dc5b91f`
- Submitted source SHA-256: `5217c4bc77fbce18fd9cca511ac1d679343e85305e32fcec3dc62df35466d0a2`
- Request: one `problemId=24`, `GeneratedWorkload`, `triton-h800` custom test; no retry
- Terminal result: `Finished / WrongAnswer`; the executed diagnostic deliberately raised the `P1MD done` marker after printing its result
- Fixed workload: P1 case c10, `(T,H,E,I,topk,M)=(4096,4096,256,1536,8,32768)`
- Math check: passed all 384 bitwise comparisons across three selected rows and three corresponding N128 segments
- FP8 times: 1.231154, 1.234601, 1.253760 ms; native INT8/INT6 times: 1.310126, 1.310144, 1.358475 ms
- Native/FP8 ratios: 1.064144, 1.061188, 1.083520; mean 1.069618 and median 1.064144
- Resources: FP8 170 registers, native 173 registers; both had zero spills and 196640 bytes shared memory
- Scope: MD kernel only, single GPU, synthetic 112/144 alternating route, unpacked signed INT6 stored in INT8; no P1 integration
- Runtime Triton version and device name remain unavailable because the diagnostic sandbox blocks the attempted access paths
- P1 production kernel: unchanged

Artifacts:

- `public24_ac6c2e16-d9d5-476d-bd94-b0618dc5b91f_raw.json` — complete API detail response, SHA-256 `86a03ce036a7a40847b9dac431a3eac1b91d4dcabbefedd3f8589d6cac3d7837`
- `public24_ac6c2e16-d9d5-476d-bd94-b0618dc5b91f_parsed.json` — structured execution evidence, SHA-256 `e71a28ad1fa2207c96a5bc212dc7aef0a9b77fc337a44d5562da6955fb81f820`
- `public24_ac6c2e16-d9d5-476d-bd94-b0618dc5b91f_user_error.txt` — exact returned diagnostic and terminal marker, SHA-256 `1d82deccd9c12c253367f53ba4017637dbf308c80b71de0a84fcf6f84ebb5132`

## Packed signed-INT6 comparison

- CID: `256c7f3d-28e1-487b-9649-2ebd22e89f09`; source SHA-256 `4077d317ac20b4446b26645c744b33a34dfef1fb6cbf1b4bd8aa10ec36107d57`; one request, no retry
- Fixed workload: P1 c10, `(T,H,E,I,topk,M)=(4096,4096,256,1536,8,32768)`
- Both unpacked and packed native paths passed 384 bitwise comparisons each
- Three packed times were 7.096384, 7.137216, and 7.121504 ms, averaging 5.758890x FP8 and 5.426048x unpacked native
- Packed resources were 255 registers and 78 spills with 90112 bytes shared memory; FP8/unpacked remained 170/173 registers, zero spills, and 196640 bytes shared
- Scope: single-GPU MD kernel only, packing and setup excluded, analytic nonrandom route, no P1 integration
- Terminal `WrongAnswer` was the deliberate post-diagnostic marker; the P1MD data came from actual execution

Artifacts:

- `public24_256c7f3d-28e1-487b-9649-2ebd22e89f09_raw.json` — SHA-256 `dddf970962b990cb932bc121db3a0d2feee2a461f61b8ebbfa88978ec44a0355`
- `public24_256c7f3d-28e1-487b-9649-2ebd22e89f09_parsed.json` — SHA-256 `f6d39e81d3b2159d506c8318452a49b55b43f1a78be203e4421bde6b99fdd013`
- `public24_256c7f3d-28e1-487b-9649-2ebd22e89f09_user_error.txt` — SHA-256 `ef432d5aefa2b65d11aee9cced998f36a78a1037fe837d9bd97e44f6b92debe9`

## L2 FP8 scaling and FP16 accumulation

- CID: `f7325752-8bbf-4c76-bc6f-c0a2a817d8c4`; source SHA-256 `3c2d4f61badbb108220e8a84d24969fdf41c3b1991e1a38ba8f30343d211cec5`; one request, no retry
- Platform elapsed: 7.641 seconds total occupied time; 7.520771 seconds case time
- Fixed workload: P1 c10, single-GPU MD kernel only; setup excluded, no DN or P1 integration
- All 384 checks passed. SQNR was 40.8001 dB for L2/F32 accumulation and 37.1515 dB for L2/FP16 accumulation; threshold 35 dB
- L2/F32 averaged 1.019424x baseline; L2/FP16 averaged 1.321173x baseline
- Resources: baseline 170 registers/0 spills; L2/F32 171/0; L2/FP16 255/64
- Terminal `WrongAnswer` was the deliberate post-diagnostic marker; the P1MD data came from actual execution

Artifacts:

- `public24_f7325752-8bbf-4c76-bc6f-c0a2a817d8c4_raw.json` — SHA-256 `a0d54bccb33734374aa4a2352697c91786f7f236e14d61e8de917e25a576333f`
- `public24_f7325752-8bbf-4c76-bc6f-c0a2a817d8c4_parsed.json` — SHA-256 `729eaca8081d7421deb985ff4349ca96379c12a6d5f2d8a8dc17710b010ef952`
- `public24_f7325752-8bbf-4c76-bc6f-c0a2a817d8c4_user_error.txt` — SHA-256 `a17778fd0b92a696b79297e53243b82b801889436c7c84dafd176e71b0ba7832`

## N64 split FP16 epilogue

- CID: `909d9826-68c6-4495-aab4-08204a77937e`; source SHA-256 `ed6cef502d03e96fa06b8b822e8f79804fcb9104ce9fd8bd2723a5f2a938d7f6`; one request, no retry
- Platform elapsed: 8.686 seconds total occupied time; 8.568360 seconds case time
- Fixed workload: P1 c10, single-GPU MD kernel only; setup excluded, no DN or P1 integration
- All 384 checks passed. Split output matched original FP16 accumulation bitwise, including row scales
- Splitting the epilogue into N64 stripes reduced resources from 255 registers/64 spills to 217/0
- Split averaged 0.904832x original FP16 accumulation, but remained 1.178810x baseline
- Terminal `WrongAnswer` was the deliberate post-diagnostic marker; the P1MD data came from actual execution

Artifacts:

- `public24_909d9826-68c6-4495-aab4-08204a77937e_raw.json` — SHA-256 `ca277a423c7f4d1ccc525c817f75cc767ee1e45cd5ca2d41ade7bf3368ff36eb`
- `public24_909d9826-68c6-4495-aab4-08204a77937e_parsed.json` — SHA-256 `0cf4fcbb604e77b69ae95ee720091cfccd5418b5b33587afbbac652b40755f2e`
- `public24_909d9826-68c6-4495-aab4-08204a77937e_user_error.txt` — SHA-256 `9a2914097ff94ff258d4f2f763c22151d820ce17301205d672284254fb8684c9`

## Sorted-A dual-TMA comparison

- CID: `c2595d22-83d7-4e86-a3d0-0da158315025`; source SHA-256 `3db5de30dee6fec470807d88ea602eb541ff5c1e6dcb45d2a015cb10cd3dff56`; one request, no retry
- Platform elapsed: 8.997 seconds total occupied time; 8.877704 seconds case time
- Fixed workload: P1 c10, single-GPU; decision metric is prep plus MD; no DN or P1 integration
- All 384 checks passed. Sorted-A and base output matched bitwise, including row scales; inverse order was exact
- MD alone averaged 1.000768x base, while sorted prep averaged 2.612749x base prep
- Prep plus MD averaged 1.030713x base and was slower in all three timing groups
- Resources: base/sorted prep used 31/32 registers and base/sorted MD used 172/177 registers; all had zero spills
- Terminal `WrongAnswer` was the deliberate post-diagnostic marker; the P1MD data came from actual execution

Artifacts:

- `public24_c2595d22-83d7-4e86-a3d0-0da158315025_raw.json` — SHA-256 `bfa5dd4091df3b4926165e75087a2a7f3c58e7183722bc2c7bb6d51a1eefe864`
- `public24_c2595d22-83d7-4e86-a3d0-0da158315025_parsed.json` — SHA-256 `e2cef6b696025b7673d655e1ff300a6b9c72eb5807e4e0d21624031b3d76f64f`
- `public24_c2595d22-83d7-4e86-a3d0-0da158315025_user_error.txt` — SHA-256 `5893e3ae7197aff04fe374630550cacb2bc9cd86f458ca2e2ef639f4857ec4fd`

## BM64 MD comparison

- CID: `d9cfb5b8-9c1a-4ba9-8f83-04c071d70adc`; source SHA-256 `c02ea135ac93e60be6a8a02d8e0d0aab5a2c9aa0ead44492f2e4fb448b2ec1b7`; one request, no retry
- Platform elapsed: 6.189 seconds total occupied time; 6.055852 seconds case time
- Fixed workload: P1 c10, single-GPU MD kernel only; metadata, quantization, BNORM, allocations and DN excluded
- All 384 boundary checks passed. BM64 and BM128 output matched bitwise, including row scales
- BM64 averaged 1.090611x BM128 and was slower in all three timing groups
- BM64 reduced resources from 172 registers and 196640 bytes shared memory to 128 registers and 163872 bytes; both had zero spills
- Terminal `WrongAnswer` was the deliberate post-diagnostic marker; the P1MD data came from actual execution

Artifacts:

- `public24_d9cfb5b8-9c1a-4ba9-8f83-04c071d70adc_raw.json` — SHA-256 `bb5db671a1ae1264e74c80aaacb4f79d3318487d58d3b4c198e5c6b6052c7e61`
- `public24_d9cfb5b8-9c1a-4ba9-8f83-04c071d70adc_parsed.json` — SHA-256 `956e3ea20c655ebe05cfd77c3c1e8d13822ea07a94273eeeeec6a8cb46439359`
- `public24_d9cfb5b8-9c1a-4ba9-8f83-04c071d70adc_user_error.txt` — SHA-256 `c17ebc3d2cb3de5c090e8170729c6b1b81f7c66a7013f5cdd14736f94b47a301`

## c2 MD and DN stage diagnostic

- CID: `5cf47dd4-dd6b-49b5-b8c7-757640849c90`; source SHA-256 `9a341ca7ccd524917b3314b3300ada7757b443563fb8a94b7ae387699dd3fa9d`; one request, no retry
- Platform elapsed: 6.056 seconds total occupied time; 5.937788 seconds case time
- Fixed workload: P1 c2, `(T,H,E,I,topk,M)=(16384,4096,8,14336,2,32768)`
- The executed math gate failed only because DN row-scale relative error was `0.0021165172`, above the configured `0.001` threshold
- MD and DN SQNR separately exceeded the 35 dB threshold: `40.43047` and `35.39511` dB; status was zero. MD/DN relative errors were `0.00951648` and `0.01699200`
- The diagnostic stopped before benchmarking and resource extraction, so there are no MD, DN, combined timing groups or resource figures
- Scope: single-GPU c2 stage diagnostic; allocation, routing/metadata, quantization, BNORM/descriptors, weight cache/collectives and final gather were outside the intended timing scope; no P1 integration
- Terminal `WrongAnswer` records the explicit pre-timing math-gate failure, rather than the intended post-diagnostic marker

Artifacts:

- `public24_5cf47dd4-dd6b-49b5-b8c7-757640849c90_raw.json` — SHA-256 `43af785aa2178086bfd53b6c7899cca8f7ded914139e7ed1c71f7e67db10f058`
- `public24_5cf47dd4-dd6b-49b5-b8c7-757640849c90_parsed.json` — SHA-256 `27c6a04e389b1e65bd22e5fe20d021e8714404280d4810797816dc2a44195dfa`
- `public24_5cf47dd4-dd6b-49b5-b8c7-757640849c90_user_error.txt` — SHA-256 `41fa44faa89f88cd57ba7f569504f4f7858efd1dfeefe59c64fa2d005a12d67d`

## c2 scale-report successor

- CID: `18f7c2a8-6203-4d01-b4e3-0978906c54ae`; source SHA-256 `8179a279ed7037125bc1546eca151401e0458deb73f5ab3827bcd1bf085947e1`; one request, no retry
- Platform elapsed: 6.093 seconds total occupied time; 5.978918 seconds case time
- Fixed workload: P1 c2, `(T,H,E,I,topk,M)=(16384,4096,8,14336,2,32768)`
- The scale-relative-error field was diagnostic-only in this successor. The retained gate failed because DN SQNR was `34.46203` dB, below 35 dB; MD SQNR was `50.06300` dB, status was zero, and all values were finite
- The diagnostic stopped before benchmarking and resource extraction, so there are no MD, DN, combined timing groups or resource figures
- Scope: single-GPU stage diagnostic, with setup, collectives and final gather excluded; no P1 integration

Artifacts:

- `public24_18f7c2a8-6203-4d01-b4e3-0978906c54ae_raw.json` — SHA-256 `e51fa22fed346b6a6da8852a55bd21d6793d29ced4f5e7c20715ff4a096e9e12`
- `public24_18f7c2a8-6203-4d01-b4e3-0978906c54ae_parsed.json` — SHA-256 `f68d2e93758434b879f47a806b3779ea61db9fbc11f15b2b8d9ee58060fa8fd7`
- `public24_18f7c2a8-6203-4d01-b4e3-0978906c54ae_user_error.txt` — SHA-256 `8c2bfa563339a36658402c72e70101605f04fb32c1b1f7c5712fe2669ffd19e3`

## c2 existing-stage profile report

- CID: `ea5a3bbe-8f08-4245-8a3e-62d18a3163f6`; source SHA-256 `885868a7570212385471e0b4dbe1c9da7b02e68e196aac25bfbc47625675722a`; one request, no retry
- Platform elapsed: 6.683 seconds total occupied time; 6.495260 seconds case time
- Fixed workload: P1 c2, `(T,H,E,I,topk,M)=(16384,4096,8,14336,2,32768)`
- Stage reference reported `status=0`, finite values and threshold passed. MD/DN SQNR was `49.74050/39.96737` dB; DN scale relative error `0.00258795` remained diagnostic-only
- Group 0 MD/DN/separate-sum/combined times were `5.431488/2.651840/8.083328/8.832992` ms; combined/sum `1.092742`
- Group 1 times were `6.139136/2.745472/8.884608/8.963424` ms; combined/sum `1.008871`
- Group 2 times were `6.173920/2.747573/8.921493/8.965184` ms; combined/sum `1.004897`
- Resources: MD 168 registers, zero spills, 196640 bytes shared; DN 186 registers, zero spills, 147480 bytes shared
- Scope: profiling of existing production-equivalent MD and DN stage ASTs on one GPU. Allocation, route/metadata, quantization, BNORM/descriptors, weight cache/collectives and final gather were excluded. This is neither P1 end-to-end timing nor new-candidate correctness validation
- Terminal `WrongAnswer` was the deliberate post-report marker; all figures above came from actual execution

Artifacts:

- `public24_ea5a3bbe-8f08-4245-8a3e-62d18a3163f6_raw.json` — SHA-256 `c6ca2e59f02937741c3cd977d046657fd734146d3a72af6358175b5d2cc89663`
- `public24_ea5a3bbe-8f08-4245-8a3e-62d18a3163f6_parsed.json` — SHA-256 `ff624b6cb2e08c5e0b50909dd5052db1c02686b4ba95c99da98fba10481bcc63`
- `public24_ea5a3bbe-8f08-4245-8a3e-62d18a3163f6_user_error.txt` — SHA-256 `ef7d3757a5d251e2639ed3c6a4347d40897de1c1be670d4283f2f41b99048e58`

## c2 MD `num_ctas=1` versus `num_ctas=2`

- CID: `dd535374-7eba-488f-bb1b-c1afc5fb86d9`; source SHA-256 `3e9eb215ba0f7b9223389b1ce19964ee946211a8f34201412d5a15da0ffb3012`; one request, no retry
- Terminal result: `Finished / WrongAnswer`; outer compile succeeded; total occupied time was 4.780 seconds and case time was 4.670467 seconds
- Generated-workload compilation aborted in Triton NVIDIA GPU CTA planning while compiling `active_c2_md_kernel`: `PlanCTA.cpp:212` asserted `CTA tiling is already determined`
- No `P1MD` diagnostic was emitted. The full 469,762,048-value comparison, SQNR/scale/finite gates, both timing paths, and resource extraction were not reached, so this run provides no `num_ctas=1` versus `num_ctas=2` performance or correctness result
- The returned `userError` is a truncated object with `omittedLength=7370`; the exact returned object is preserved. The module fragment records `ttg.maxnreg = 168`, which is launch configuration evidence rather than a compiled resource measurement

Artifacts:

- `public24_dd535374-7eba-488f-bb1b-c1afc5fb86d9_raw.json` — complete API detail response, SHA-256 `66e0e6ad6cb52c272c39e6897a3c0ec26905471284732687f39126e4c60762d0`
- `public24_dd535374-7eba-488f-bb1b-c1afc5fb86d9_parsed.json` — structured interpretation, SHA-256 `183007f74fcb596bbd0863a10246cbfaa551773c2ea6b96e45f904f581d43823`
- `public24_dd535374-7eba-488f-bb1b-c1afc5fb86d9_user_error.json` — exact returned `userError` object, SHA-256 `2a1988da19b9132bba82d9341a95dfb5a218512b3812e8983df812794fbec947`

## c2 MD BM64 / two-stage / maxnreg 128 occupancy configuration

- CID: `7e57a7d9-1d75-4b56-8f96-62d950c0a221`; source SHA-256 `ee83ef2ba38980ca011208b68b54fc008183604d8593fa2be490573e8a04afc0`; one request, no retry
- Terminal result: `Finished / WrongAnswer`; outer compile succeeded; total occupied time was 5.581 seconds and case time was 5.463484 seconds
- Generated-workload compilation failed in `full_compare_partial_kernel`: `tl.atomic_or(STATUS, 1, mask=mask & (value_finite == 0))` was rejected because the `tt.atomic_rmw` mask type did not match the scalar value type
- No `P1MD` diagnostic was emitted. The 469,762,048-value strict comparison, SQNR/scale/finite gates, three timing groups and compiled resource extraction were not reached, so this run provides no BM64 performance or correctness result
- The returned `userOutput` is a truncated compiler-IR object with `omittedLength=37933`; the raw response preserves exactly what the detail endpoint returned. Configuration caps (`maxnreg=168/128`) are not reported as measured resources

Artifacts:

- `public24_7e57a7d9-1d75-4b56-8f96-62d950c0a221_raw.json` — complete API detail response, SHA-256 `e1e3bff7ae3a2292d362d4a6ed534d36a54e108e7452b4e3587a48361c25e240`
- `public24_7e57a7d9-1d75-4b56-8f96-62d950c0a221_parsed.json` — structured interpretation, SHA-256 `9ff872269ac73bf672172baf8db686fb870ac986073866d059ea92b83f96d42b`
- `public24_7e57a7d9-1d75-4b56-8f96-62d950c0a221_user_error.txt` — exact returned `userError`, SHA-256 `77ad60edf5b6ddf7742eb22bb02f0555d93ef7ec8d346a4434c47550d5db8c87`

## c2 MD BM64 occupancy configuration atomic-mask successor

- CID: `cda307aa-966a-447c-840f-a535695c0bf9`; source SHA-256 `8dddacd28ccd7b4bc392a0c19da3fe2d8dc747faa47a8f1ce2b91c93055f4925`; one request, no retry
- Terminal `Finished / WrongAnswer` was the deliberate `P1MD done` marker; outer compile succeeded; total occupied time was 6.863 seconds and case time was 6.760783 seconds
- The 469,762,048-value strict GPU comparison passed: all values were finite, candidate/baseline SQNR was `119.999992 dB`, maximum value relative error was zero, FP8 exact fraction was 1.0, and maximum row-scale relative error was zero
- Three candidate/base MD timing groups were `14.658016/5.411040 = 2.708909x`, `14.682656/5.446112 = 2.695989x`, and `14.671712/6.168704 = 2.378411x`; the BM64 candidate was slower in every group
- Measured resources: baseline `168 registers / 0 spills / 196640 bytes shared`; candidate `122 / 0 / 81936`. These are compiled measurements, distinct from the configured register caps
- Scope: single-GPU c2 MD only with analytic route. Allocation, route/metadata, quantization, BNORM/descriptors, validation, DN and final gather were excluded

Artifacts:

- `public24_cda307aa-966a-447c-840f-a535695c0bf9_raw.json` — complete API detail response, SHA-256 `f8a4f7956566995267bc2269b3f34045c494c3f8044c438ac87280f896a5be9d`
- `public24_cda307aa-966a-447c-840f-a535695c0bf9_parsed.json` — structured execution evidence, SHA-256 `3946f4391930e7eaceded1e3a2cc0a68dfddd69e0e39bbedbe9745c380e38a93`
- `public24_cda307aa-966a-447c-840f-a535695c0bf9_user_error.txt` — exact returned diagnostic and terminal marker, SHA-256 `e6f33d95bcb30376ad87cc1d20b413688a3f93fe920d3aef87c3fe1fe7878607`
