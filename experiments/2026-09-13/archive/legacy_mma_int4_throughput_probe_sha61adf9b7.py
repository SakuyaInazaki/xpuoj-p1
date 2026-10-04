"""GeneratedWorkload-only H800 legacy warp-MMA throughput probe.

This is a register-resident instruction upper-bound diagnostic, not a P1
candidate.  Packed operands are constants in registers.  Allocation, checks,
HBM traffic, TMA, quantization, unpacking, and scale application are excluded.
"""

import torch
import triton
import triton.language as tl


GRID = 1056
WARPS = 4
LANES_PER_CTA = 128
VALUES_PER_LANE = 4
ITERATIONS = 2048
OUTPUT_VALUES = 540672
CHECK_BLOCK = 1024


@triton.jit
def mma_s4_kernel(OUT, ITER: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, LANES_PER_CTA)
    one = lane * 0 + 0x11111111
    c0 = lane * 0
    c1 = lane * 0
    c2 = lane * 0
    c3 = lane * 0
    for _ in tl.range(0, ITER, loop_unroll_factor=1):
        c0, c1, c2, c3 = tl.inline_asm_elementwise(
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
            [one, one, one, one, one, one, c0, c1, c2, c3],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
            is_pure=True,
            pack=1,
        )
    base = pid * (LANES_PER_CTA * VALUES_PER_LANE) + lane * VALUES_PER_LANE
    tl.store(OUT + base + 0, c0)
    tl.store(OUT + base + 1, c1)
    tl.store(OUT + base + 2, c2)
    tl.store(OUT + base + 3, c3)


@triton.jit
def mma_s8_kernel(OUT, ITER: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, LANES_PER_CTA)
    one = lane * 0 + 0x01010101
    c0 = lane * 0
    c1 = lane * 0
    c2 = lane * 0
    c3 = lane * 0
    for _ in tl.range(0, ITER, loop_unroll_factor=1):
        c0, c1, c2, c3 = tl.inline_asm_elementwise(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
            [one, one, one, one, one, one, c0, c1, c2, c3],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
            is_pure=True,
            pack=1,
        )
    base = pid * (LANES_PER_CTA * VALUES_PER_LANE) + lane * VALUES_PER_LANE
    tl.store(OUT + base + 0, c0)
    tl.store(OUT + base + 1, c1)
    tl.store(OUT + base + 2, c2)
    tl.store(OUT + base + 3, c3)


@triton.jit
def mma_fp8_kernel(OUT, ITER: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, LANES_PER_CTA)
    one = lane * 0 + 0x38383838
    c0 = (lane * 0).to(tl.float32)
    c1 = (lane * 0).to(tl.float32)
    c2 = (lane * 0).to(tl.float32)
    c3 = (lane * 0).to(tl.float32)
    for _ in tl.range(0, ITER, loop_unroll_factor=1):
        c0, c1, c2, c3 = tl.inline_asm_elementwise(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            [one, one, one, one, one, one, c0, c1, c2, c3],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )
    base = pid * (LANES_PER_CTA * VALUES_PER_LANE) + lane * VALUES_PER_LANE
    tl.store(OUT + base + 0, c0)
    tl.store(OUT + base + 1, c1)
    tl.store(OUT + base + 2, c2)
    tl.store(OUT + base + 3, c3)


@triton.jit
def check_i32_kernel(OUT, STATUS, EXPECTED: tl.constexpr, N: tl.constexpr):
    offsets = tl.program_id(0) * CHECK_BLOCK + tl.arange(0, CHECK_BLOCK)
    mask = offsets < N
    value = tl.load(OUT + offsets, mask=mask, other=EXPECTED)
    bad = tl.sum((value != EXPECTED).to(tl.int32), axis=0)
    tl.atomic_or(STATUS, (bad != 0).to(tl.int32))


@triton.jit
def check_f32_kernel(OUT, STATUS, EXPECTED: tl.constexpr, N: tl.constexpr):
    offsets = tl.program_id(0) * CHECK_BLOCK + tl.arange(0, CHECK_BLOCK)
    mask = offsets < N
    value = tl.load(OUT + offsets, mask=mask, other=EXPECTED)
    bad = tl.sum((value != EXPECTED).to(tl.int32), axis=0)
    tl.atomic_or(STATUS, (bad != 0).to(tl.int32))


def run_diagnostic():
    device = "cuda"
    out_s4 = torch.empty((OUTPUT_VALUES,), dtype=torch.int32, device=device)
    out_s8 = torch.empty((OUTPUT_VALUES,), dtype=torch.int32, device=device)
    out_fp8 = torch.empty((OUTPUT_VALUES,), dtype=torch.float32, device=device)

    def call_s4():
        return mma_s4_kernel[(GRID,)](
            out_s4, ITER=ITERATIONS, num_warps=WARPS, num_stages=1
        )

    def call_s8():
        return mma_s8_kernel[(GRID,)](
            out_s8, ITER=ITERATIONS, num_warps=WARPS, num_stages=1
        )

    def call_fp8():
        return mma_fp8_kernel[(GRID,)](
            out_fp8, ITER=ITERATIONS, num_warps=WARPS, num_stages=1
        )

    compiled_s4 = call_s4()
    compiled_s8 = call_s8()
    compiled_fp8 = call_fp8()

    status_s4 = torch.zeros((1,), dtype=torch.int32, device=device)
    status_s8 = torch.zeros((1,), dtype=torch.int32, device=device)
    status_fp8 = torch.zeros((1,), dtype=torch.int32, device=device)
    check_grid = (triton.cdiv(OUTPUT_VALUES, CHECK_BLOCK),)
    check_i32_kernel[check_grid](
        out_s4, status_s4, EXPECTED=64 * ITERATIONS, N=OUTPUT_VALUES,
        num_warps=4, num_stages=1,
    )
    check_i32_kernel[check_grid](
        out_s8, status_s8, EXPECTED=32 * ITERATIONS, N=OUTPUT_VALUES,
        num_warps=4, num_stages=1,
    )
    check_f32_kernel[check_grid](
        out_fp8, status_fp8, EXPECTED=32.0 * ITERATIONS, N=OUTPUT_VALUES,
        num_warps=4, num_stages=1,
    )
    s4_status = int(status_s4[0].item())
    s8_status = int(status_s8[0].item())
    fp8_status = int(status_fp8[0].item())
    if s4_status != 0 or s8_status != 0 or fp8_status != 0:
        print(
            'P1MD {"kind":"legacy_mma_int4_throughput_probe",'
            '"validation_passed":false,"status":{"s4":' + str(s4_status)
            + ',"s8":' + str(s8_status) + ',"fp8":' + str(fp8_status) + "}}"
        )
        raise RuntimeError("P1MD legacy MMA result validation failed")

    calls = [call_s4, call_s8, call_fp8]
    names = ["s4", "s8", "fp8"]
    orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    groups_text = []
    sums = [0.0, 0.0, 0.0]
    for group in range(3):
        measured = [0.0, 0.0, 0.0]
        for index in orders[group]:
            measured[index] = float(
                triton.testing.do_bench(calls[index], warmup=5, rep=10)
            )
            sums[index] += measured[index]
        groups_text.append(
            '{"group":' + str(group) + ',"order":"'
            + names[orders[group][0]] + "_" + names[orders[group][1]] + "_"
            + names[orders[group][2]] + '","s4_ms":' + str(measured[0])
            + ',"s8_ms":' + str(measured[1]) + ',"fp8_ms":' + str(measured[2]) + "}"
        )

    mean_s4 = sums[0] / 3.0
    mean_s8 = sums[1] / 3.0
    mean_fp8 = sums[2] / 3.0
    ops_s4 = 2 * 16 * 8 * 64 * WARPS * GRID * ITERATIONS
    ops_s8 = 2 * 16 * 8 * 32 * WARPS * GRID * ITERATIONS
    ops_fp8 = 2 * 16 * 8 * 32 * WARPS * GRID * ITERATIONS
    tops_s4 = ops_s4 / (mean_s4 * 1.0e9)
    tops_s8 = ops_s8 / (mean_s8 * 1.0e9)
    tops_fp8 = ops_fp8 / (mean_fp8 * 1.0e9)
    def resource_text(compiled):
        return ('{"n_regs":' + str(compiled.n_regs) + ',"n_spills":'
                + str(compiled.n_spills) + ',"shared":'
                + str(compiled.metadata.shared) + "}")

    report = (
        '{"kind":"legacy_mma_int4_throughput_probe",'
        '"purpose":"single_GPU_register_resident_legacy_warp_MMA_upper_bound_not_P1_candidate",'
        '"geometry":{"grid":1056,"warps":4,"lanes_per_CTA":128,'
        '"iterations":2048,"outputs_per_lane":4},'
        '"operands":{"prepacked_register_constants":true,"s4":"0x11111111",'
        '"s8":"0x01010101","e4m3":"0x38383838","unpacking":false},'
        '"validation":{"passed":true,"scope":"all_540672_GPU_outputs",'
        '"GPU_independent_reduction":true,"s4_expected":131072,'
        '"s8_expected":65536,"fp8_expected":65536.0},'
        '"resources":{"s4":' + resource_text(compiled_s4)
        + ',"s8":' + resource_text(compiled_s8)
        + ',"fp8":' + resource_text(compiled_fp8) + '},'
        '"timing":{"method":"do_bench_warmup5_rep10_kernel_only",'
        '"orders":"ABC_BCA_CAB","groups":[' + ",".join(groups_text)
        + '],"mean_ms":{"s4":' + str(mean_s4) + ',"s8":' + str(mean_s8)
        + ',"fp8":' + str(mean_fp8) + '},"effective_TOPS":{"s4":'
        + str(tops_s4) + ',"s8":' + str(tops_s8) + ',"fp8":' + str(tops_fp8)
        + '},"speedup":{"s4_over_s8_TOPS":' + str(tops_s4 / tops_s8)
        + ',"s4_over_fp8_TOPS":' + str(tops_s4 / tops_fp8)
        + ',"s8_over_fp8_TOPS":' + str(tops_s8 / tops_fp8) + '}},'
        '"excluded":{"allocation":true,"validation":true,"HBM":true,'
        '"TMA":true,"quantization":true,"unpacking":true,"scales":true},'
        '"limitations":"one_geometry_no_parameter_sweep;register_resident_instruction_upper_bound;does_not_predict_end_to_end_or_P1_throughput"}'
    )
    print("P1MD " + report)
    raise RuntimeError("P1MD done")


def run_kernel(x):
    return x


run_diagnostic()
