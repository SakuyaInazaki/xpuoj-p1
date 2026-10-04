# L2-normalized FP8 MD with FP16 accumulation: bounded CPU screen

This screen covers only `K=4096`, `I in {1536, 2048}`, `M=8`, `O=256`, eight experts, and two fixed seeds per shape. Inputs and weights are Gaussian values rounded to BF16. The reference uses BF16 inputs and weights with float32 matrix multiplication. Route weights are the fixed normalized positive sequence `[1, ..., 8]`.

The L2 path computes each A/GU scale from the original BF16 row as `max(row_l2 / 128, 1e-12)`, rounds the scaled row to E4M3FN, and dequantizes after MD. Its ACT bound continues to use the original BF16 A row `amax / 448`, independently of the new A dequantization scale. BNORM is measured from the actually dequantized GU for each path. FP8 ACT, FP8 DN, the per-row FP8 terminal quantization for the one 256-column output block, expert branch sum, and final BF16 rounding match the earlier bounded screen.

The FP16 result is a CPU rounding proxy: each K=32 partial dot is computed in float32, rounded to float16, and added to a float16 running accumulator. The final accumulator is converted to float32 before dequantization and SiLU.

| I | Seed | Baseline FP8 SQNR (dB) | L2 + F32 acc (dB) | L2 + K32 F16 proxy (dB) | F16 margin over 22 dB | All finite | F16 running max | Quant A row L2 max | Quant GU row L2 max |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1536 | 2026091201 | 23.1635 | 23.2687 | 23.1958 | 1.1958 | yes | 1179 | 128.0930 | 128.5530 |
| 1536 | 2026091202 | 22.8859 | 23.1379 | 23.0069 | 1.0069 | yes | 1140 | 128.1304 | 128.6115 |
| 2048 | 2026091201 | 22.9858 | 23.3833 | 23.4147 | 1.4147 | yes | 1179 | 128.0930 | 128.5530 |
| 2048 | 2026091202 | 23.0169 | 23.0877 | 23.0757 | 1.0757 | yes | 1140 | 128.1304 | 128.6115 |

All four L2 + F32-control cases and all four L2 + K32-FP16-proxy cases exceed 22 dB. The minimum proxy SQNR is **23.0069 dB**, a **1.0069 dB** margin. The minimum L2 + F32-control SQNR is **23.0877 dB**. All references, outputs, and FP16 running accumulators are finite.

Across the four cases, the current FP8 baseline's raw MD accumulator maximum is 4,665,583.5. L2 normalization reduces the L2 + F32 accumulator maximum to 1,130.46 and the L2 + FP16 final/running maxima to 1,130/1,179. The rounded FP8 rows remain near the target L2: at most 128.1304 for A and 128.6115 for GU.

The source SHA-256 is `093694156a4a9575d189190ce1c9ebc8fb4a70fe4d3542eec48f69b95e140a8c`. Full-precision values and per-expert timings are in [fp16acc_l2_precision_screen_results.json](fp16acc_l2_precision_screen_results.json). This CPU screen does not prove GPU compilation, exact GPU accumulation order, platform precision, register use, BM256 feasibility, or speed.
