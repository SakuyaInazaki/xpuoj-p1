# 2026-09-13 public24 custom diagnostics

Earlier custom-test records are indexed in the [2026-09-12 results](../../2026-09-12/results/INDEX.md). Current project status is in [STATE](../../../docs/STATE.md).

## Legacy warp-MMA throughput probe initial compile

- CID: `6d48a6dd-65e8-472b-bc86-848a855e36f5`; source SHA-256 `61adf9b7d0383c428879e22c554aa43cacafb15af9b7867e7df56ab8596676c9`; one request, no retry
- Terminal result: `Finished / WrongAnswer`; outer compile succeeded; total occupied time was 3.156 seconds and case time was 3.046714 seconds
- Generated-workload JIT compilation stopped in `mma_s4_kernel` at `tl.arange(0, LANES_PER_CTA)`: current Triton rejects access to an ordinary module global from a JIT kernel and requires a `tl.constexpr` instance or explicit constexpr argument
- No `P1MD` diagnostic was emitted. The s4/s8/FP8 gates, three timing groups, effective TOPS, ratios and compiled resources were not reached; this run contains no instruction-throughput evidence
- Per the experiment gate, the asm was not changed and the failed source was not resubmitted

Artifacts:

- `public24_6d48a6dd-65e8-472b-bc86-848a855e36f5_raw.json` — complete API detail response, SHA-256 `c03e76f199b85a856f3c23bc024b2372c9d2304796def8fc9c5ff3917de255d1`
- `public24_6d48a6dd-65e8-472b-bc86-848a855e36f5_parsed.json` — structured interpretation, SHA-256 `50277f7da412fd611edcb6a57893d45ece6ce0a6a994eb5173283ee4290b95f7`
- `public24_6d48a6dd-65e8-472b-bc86-848a855e36f5_user_error.txt` — exact returned `userError`, SHA-256 `77048f91d4e442970b8dba67ebabda721556d8145e211adaf6c254956e171512`

## Legacy warp-MMA constexpr successor

- CID: `96aa7f26-4e82-4b06-a159-303d97d40437`; source SHA-256 `81c7dc52ce4857face8346b3d7189792e0041a117ff516c45be8e1de79413eeb`; one request, no retry
- Terminal `Finished / WrongAnswer` was the deliberate `P1MD done` marker; outer compile succeeded; total occupied time was 4.097 seconds and case time was 3.980887 seconds
- All 540672 outputs passed exact GPU validation: s4 expected `131072`, s8 `65536`, and FP8 e4m3 `65536.0`
- Three s4/s8/FP8 timing groups were `0.240751/0.060865/0.069455`, `0.240822/0.060961/0.069482`, and `0.241078/0.060614/0.069670 ms`; means were `0.240884/0.060813/0.069536 ms`
- Effective throughput was `588.391 TOPS` for s4, `1165.320` for s8, and `1019.145` for FP8. Ratios were s4/s8 `0.504918`, s4/FP8 `0.577337`, and s8/FP8 `1.143429`
- Measured resources were s4 `16 registers / 0 spills / 0 shared`, s8 `18/0/0`, and FP8 `18/0/0`
- Scope: one single-GPU, register-resident legacy warp-MMA geometry with constants. HBM, TMA, quantization, unpacking and scales were excluded; this is an instruction upper bound and does not predict P1 end-to-end throughput

Artifacts:

- `public24_96aa7f26-4e82-4b06-a159-303d97d40437_raw.json` — complete API detail response, SHA-256 `e836a85a8a5c2f1c7076b41176483de2f2b2ff950b4c138e0cb4821e30b90889`
- `public24_96aa7f26-4e82-4b06-a159-303d97d40437_parsed.json` — structured execution evidence, SHA-256 `916cccee820574551ee3078a8f38c220ce16e135cb30d7784e138df23bd3570f`
- `public24_96aa7f26-4e82-4b06-a159-303d97d40437_user_error.txt` — exact returned diagnostic and terminal marker, SHA-256 `194eff647a5663d4a5197dff63acbda5fd78e7d8f415ad9e61aed80b23555be5`

## c4 DN full-tile TMA descriptor-store probe

- CID: `de45e5a9-16ed-475f-ab49-b733daa621d0`; source SHA-256 `5d69abb6c04b715cb0479cc93338cabde1cb52b63a7e417f841a52de602756aa`; 18,913 bytes; one request, no retry
- Terminal result: `Finished / WrongAnswer`; outer compile succeeded; total occupied time was 4.792 seconds and case time was 4.621762 seconds
- `candidate_dn_kernel` JIT stopped in MLIR verification: the flattened `tl.range` containing the full-tile descriptor-store conditional lowered to an `scf.if` whose region/result arity did not match (`source has 1 operands, but target successor needs 5`)
- This was not a shared-memory limit failure. No `P1MD` line was emitted, so the FP8/scale/final-output checks, resource fields and all three DN and DN+gather timing groups were not reached. It gives no TMA-store performance result and was not modified or retried
- Audit boundary: the unreached zero-row check used FP8 bit-pattern zero rather than FP32 numeric zero, so it could falsely reject `-0`. The planned baseline/candidate full-output bitwise comparison is unaffected, but `zero_rows_exact` cannot be claimed from this source

Artifacts:

- `public24_de45e5a9-16ed-475f-ab49-b733daa621d0_raw.json` — terminal API-detail projection with source referenced by path/SHA and the server-returned truncated compiler diagnostic, SHA-256 `15245ecd9ff19af27542db503edacfd35c3459b9b703e582636756ed549afe90`
- `public24_de45e5a9-16ed-475f-ab49-b733daa621d0_parsed.json` — structured interpretation and evidence gates, SHA-256 `0e70675ccea100f33c2cf241e06650fd6bb313443773391d47d5211264e85e1a`
- `public24_de45e5a9-16ed-475f-ab49-b733daa621d0_user_error.txt` — returned compiler error header and platform truncation boundary, SHA-256 `ca997ae9e448aac1f2b3ffdb6634d47521ed4786e743f6effd5b971d49285f74`

## c4 DN TMA descriptor-store + candidate noflat successor

- CID: `aa4ff079-1264-4b84-bff5-73398e324b8b`; source SHA-256 `2d3956980def52da84c106fda030f03362c8e21800249def6af198dc42beb342`; 19,071 bytes; one request, no retry
- Terminal `Finished / WrongAnswer` was the deliberate `P1MD done` marker; outer compile succeeded; total occupied time was 6.245 seconds and case time was 6.115150 seconds
- All hard gates passed: full FP8 output bitwise exact, full scale exact, final BF16 output exact, FP32 numeric-zero rows exact; FP8/scale/final mismatch counts were all zero
- Baseline/candidate/gather resources were respectively `170/0/229408`, `231/0/229408`, and `43/0/8192` for registers/spills/shared bytes
- AB/BA/AB DN baseline→candidate times were `0.244938→0.290673`, `0.246133→0.291547`, and `0.247740→0.294373 ms`, ratios `1.18672/1.18451/1.18823×`
- Corresponding consecutive DN+same-gather times were `0.319670→0.365551`, `0.321578→0.367768`, and `0.321979→0.369001 ms`, ratios `1.14353/1.14364/1.14604×`
- Mean slowdowns were **18.65% for DN** and **14.44% for DN+gather**, with 61 additional candidate registers and unchanged shared memory. The combined output-TMA + candidate noflat implementation is closed; this is single-GPU mechanism evidence, not P1 end-to-end timing

Artifacts:

- `public24_aa4ff079-1264-4b84-bff5-73398e324b8b_raw.json` — terminal API-detail projection with immutable source reference, SHA-256 `6346908b74b1b94960e7da8938379a0466c6ac821afce809508d91d0d690a335`
- `public24_aa4ff079-1264-4b84-bff5-73398e324b8b_parsed.json` — structured hard gates, resources, timing groups and interpretation, SHA-256 `5202a1d08beb52a58e0c7cd0a5f5079ed1544eabbebbf4542da2fa06ddbe6921`
- `public24_aa4ff079-1264-4b84-bff5-73398e324b8b_user_error.txt` — exact returned P1MD diagnostic and terminal marker, SHA-256 `27ff1a7de5a9d825b9ea60745a3971e004628d4108628f320db25fa839fef5e8`

## c4 padded output-TMA + taxed rowmap successor

- CID: `edea0c78-e34a-4bdb-b31d-82636ed70f0c`; source SHA-256 `b09efe6476054104502f3a42fc19d8b70bf6f45d13e3a3a79e8d8bad75b85d65`; 23,955 bytes; one request, no retry
- Terminal `Finished / WrongAnswer` was the deliberate `P1MD done` marker. Full FP8/scale/final BF16/numeric-zero and analytic PAD_ROW gates passed with all mismatch counts zero
- Resources baseline/candidate/rowmap were `170/0/229408`, `172/0/230432`, and `16/0/0` for registers/spills/shared; baseline and padded gather were both `43/0/8192`
- DN-only AB/BA/AB ratios were `0.97199/0.96821/0.97111×` (2.80–3.18% faster)
- The main comparison includes candidate rowmap→DN→padded gather on every measured call. Its three ratios were `0.99847/0.99182/0.97859×`, or 0.15–2.14% faster; ratio of means was `0.98952×`
- This is a single-GPU c4 mechanism result, not P1 end-to-end evidence. Performance disposition belongs to root; this record does not authorize expansion or parameter sweeps

Artifacts:

- `public24_edea0c78-e34a-4bdb-b31d-82636ed70f0c_raw.json` — terminal API-detail projection with immutable source reference, SHA-256 `229988f26155e96dcc42b895db573e713c1e46094e0ea5fec7f0b8b0b2647bbc`
- `public24_edea0c78-e34a-4bdb-b31d-82636ed70f0c_parsed.json` — structured hard gates, resources and three timing groups, SHA-256 `f47189dffeb2ab31b1ba1554551c49eb7d63b3fa5e08d3cf1f463c69b04c6c7e`
- `public24_edea0c78-e34a-4bdb-b31d-82636ed70f0c_user_error.txt` — exact returned P1MD diagnostic and terminal marker, SHA-256 `43ba9b3f2c509800c648ca21672358e7f606c2d7dbf844685bce92baa015aec2`

## c4 padded output-TMA six-group confirmation

- CID: `f5c14b6e-30cf-4e52-8c94-cf7bdb1a5595`; source SHA-256 `e3f0db278e452bf1058a2bf04cc5ed15176dccc929d6683245f4f105da00fd59`; 24,008 bytes; one request, no retry
- All hard gates and resources reproduced exactly. `WrongAnswer` was the deliberate `P1MD done` marker
- Six balanced AB/BA/BA/AB/AB/BA DN-only ratios were `0.97476/0.94526/0.95818/0.97276/0.94478/0.96707×`; DN-only was faster in all groups
- The taxed main rowmap→DN→padded-gather ratios were `0.99323/1.00855/1.01244/1.01600/0.98521/0.99015×`: three faster and three slower. Ratio of means was `1.00088×`, so the earlier tentative main-path gain did not reproduce as a stable benefit
- This confirmation does not expand case scope or authorize another measurement; disposition belongs to root

Artifacts:

- `public24_f5c14b6e-30cf-4e52-8c94-cf7bdb1a5595_raw.json` — terminal API-detail projection with immutable source reference, SHA-256 `8a5cffda2088ac3e85b98221778447b2f8cb3f4e8115b68b7510016a7b47177a`
- `public24_f5c14b6e-30cf-4e52-8c94-cf7bdb1a5595_parsed.json` — six raw timing groups, hard gates, resources and means, SHA-256 `2d69846ef34eb86153c5e944b823be4fa53af84bc7d34ea1117ecdef528564b1`
- `public24_f5c14b6e-30cf-4e52-8c94-cf7bdb1a5595_user_error.txt` — exact returned P1MD diagnostic and terminal marker, SHA-256 `190504f5c181a0d221bfb92e08189afc287b7da87777c95c3c5acb4ed6b6ba47`
