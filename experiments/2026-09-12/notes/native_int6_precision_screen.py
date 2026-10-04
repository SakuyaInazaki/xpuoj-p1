"""Bounded CPU precision screen for native INT8-A / signed-6-bit GU MD.

This intentionally covers only K=4096, I in {1536, 2048}, M=8, O=256,
eight experts, and two fixed seeds per shape.  It is a mathematical screen,
not a reproduction of GPU accumulation, compilation, layout, or timing.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch


K = 4096
INTERMEDIATE_SIZES = (1536, 2048)
M = 8
O = 256
EXPERTS = 8
SEEDS = (2026091201, 2026091202)
SQNR_TARGET_DB = 22.0
FP8_MAX = 448.0
INT8_MAX = 127.0
INT6_MAX = 31.0
TINY_SCALE = 1.0e-12

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FP8_HELPER_PATH = REPO / "experiments/2026-09-08/notes/fp6_e4m3_cpu_verify.py"
RESULT_PATH = HERE / "native_int6_precision_screen_results.json"


def _load_fp8_helper():
    spec = importlib.util.spec_from_file_location("fp6_e4m3_cpu_verify", FP8_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FP8 helper: {FP8_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FP8_HELPER = _load_fp8_helper()
FP8 = FP8_HELPER.FP8


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round finite float32 values to BF16, returning float32 containers."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def gaussian_bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return bf16_round(rng.standard_normal(shape, dtype=np.float32))


def fp8_row_quant(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the existing CPU helper for current per-row E4M3FN quantization."""
    q, scale = FP8_HELPER.current_row_fp8(torch.from_numpy(np.ascontiguousarray(x)))
    q_f32 = q.float().numpy()
    scale_np = scale.numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(q_f32), scale_np[:, 0].copy()


def fp8_quant_with_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = np.ascontiguousarray(x / scale[:, None], dtype=np.float32)
    q = torch.from_numpy(scaled).to(FP8)
    return np.ascontiguousarray(q.float().numpy())


def native_int_row_quant(
    x: np.ndarray, qmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amax = np.max(np.abs(x), axis=1)
    scale = np.maximum(amax / np.float32(qmax), np.float32(TINY_SCALE)).astype(np.float32)
    q = np.clip(np.rint(x / scale[:, None]), -qmax, qmax).astype(np.int8)
    deq = q.astype(np.float32) * scale[:, None]
    return q, scale, deq


def silu(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = x[positive] / (
        np.float32(1.0) + np.exp(-x[positive], dtype=np.float32)
    )
    exp_x = np.exp(x[~positive], dtype=np.float32)
    out[~positive] = x[~positive] * exp_x / (np.float32(1.0) + exp_x)
    return out


def act_power2_scale(
    effective_a_scale: np.ndarray, gu_bnorm: float, route_weight: float
) -> np.ndarray:
    """Mirror _fgs_tma1_kernel_gq's exponent-only ACT scale formula."""
    bound = (
        effective_a_scale.astype(np.float32) ** np.float32(2.0)
        * np.float32(gu_bnorm) ** np.float32(2.0)
        * np.float32(abs(route_weight))
    )
    bound = np.maximum(bound, np.float32(1.0e-30))
    # The kernel extracts the float32 exponent bits rather than calling log2.
    biased_exponent = ((bound.view(np.uint32) >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    exponent = biased_exponent - np.int32(127) + np.int32(10)
    return np.ldexp(np.ones_like(bound, dtype=np.float32), exponent).astype(np.float32)


def terminal_fp8_qdq(down: np.ndarray) -> np.ndarray:
    """One O=256 block: per-row row_max/448 quantization before branch sum."""
    row_max = np.max(np.abs(down), axis=1)
    scale = np.maximum(row_max / np.float32(FP8_MAX), np.float32(TINY_SCALE))
    q = fp8_quant_with_scale(down, scale.astype(np.float32))
    return q * scale[:, None]


def sqnr(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    ref64 = reference.astype(np.float64)
    err64 = candidate.astype(np.float64) - ref64
    signal_rms = float(np.sqrt(np.mean(ref64 * ref64)))
    error_rms = float(np.sqrt(np.mean(err64 * err64)))
    sqnr_db = float(20.0 * math.log10(signal_rms / error_rms))
    return {
        "signal_rms": signal_rms,
        "error_rms": error_rms,
        "relative_rms": error_rms / signal_rms,
        "sqnr_db": sqnr_db,
        "passes_22db": sqnr_db >= SQNR_TARGET_DB,
        "margin_over_22db": sqnr_db - SQNR_TARGET_DB,
    }


def rss_mib() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0**2))
    except ImportError:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux reports KiB.
        return peak / (1024.0**2) if platform.system() == "Darwin" else peak / 1024.0


def float_matmul(a: np.ndarray, b_t: np.ndarray) -> np.ndarray:
    return np.asarray(a @ b_t, dtype=np.float32)


def exact_int_matmul(q_a: np.ndarray, q_b_t: np.ndarray) -> np.ndarray:
    # float32 represents every possible K*127*31=16,125,952 integer exactly.
    return np.asarray(q_a.astype(np.float32) @ q_b_t.astype(np.float32), dtype=np.float32)


def run_case(intermediate: int, seed: int) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    route_weights = np.arange(1, EXPERTS + 1, dtype=np.float32)
    route_weights /= np.sum(route_weights, dtype=np.float32)

    x = gaussian_bf16(rng, (M, K))
    x_fp8_q, x_fp8_scale = fp8_row_quant(x)
    x_int8, x_int8_scale, _ = native_int_row_quant(x, INT8_MAX)

    reference_sum = np.zeros((M, O), dtype=np.float32)
    baseline_sum = np.zeros((M, O), dtype=np.float32)
    native_sum = np.zeros((M, O), dtype=np.float32)
    sampled_peak_rss = rss_mib()
    expert_seconds: list[float] = []

    for expert, route_weight in enumerate(route_weights):
        expert_started = time.perf_counter()
        gu = gaussian_bf16(rng, (2 * intermediate, K))
        dn = gaussian_bf16(rng, (O, intermediate))
        sampled_peak_rss = max(sampled_peak_rss, rss_mib())

        gu_fp8_q, gu_fp8_scale = fp8_row_quant(gu)
        gu_int6, gu_int6_scale, gu_int6_deq = native_int_row_quant(gu, INT6_MAX)
        dn_fp8_q, dn_fp8_scale = fp8_row_quant(dn)
        baseline_bnorm = float(
            np.max(np.linalg.norm(gu_fp8_q * gu_fp8_scale[:, None], axis=1))
        )
        native_bnorm = float(np.max(np.linalg.norm(gu_int6_deq, axis=1)))

        ref_gu = float_matmul(x, gu.T)
        baseline_gu = float_matmul(x_fp8_q, gu_fp8_q.T)
        baseline_gu *= x_fp8_scale[:, None] * gu_fp8_scale[None, :]
        native_gu = exact_int_matmul(x_int8, gu_int6.T)
        native_gu *= x_int8_scale[:, None] * gu_int6_scale[None, :]

        ref_g, ref_u = ref_gu[:, :intermediate], ref_gu[:, intermediate:]
        base_g, base_u = baseline_gu[:, :intermediate], baseline_gu[:, intermediate:]
        native_g, native_u = native_gu[:, :intermediate], native_gu[:, intermediate:]
        ref_act = silu(ref_g) * ref_u * route_weight
        baseline_act = silu(base_g) * base_u * route_weight
        native_act = silu(native_g) * native_u * route_weight

        baseline_act_scale = act_power2_scale(x_fp8_scale, baseline_bnorm, float(route_weight))
        # Preserve the kernel's equivalent amax/448 convention only for ACT bound.
        native_bound_a_scale = x_int8_scale * np.float32(INT8_MAX / FP8_MAX)
        native_act_scale = act_power2_scale(native_bound_a_scale, native_bnorm, float(route_weight))
        baseline_act_fp8_q = fp8_quant_with_scale(baseline_act, baseline_act_scale)
        native_act_fp8_q = fp8_quant_with_scale(native_act, native_act_scale)

        ref_down = float_matmul(ref_act, dn.T)
        baseline_down = float_matmul(baseline_act_fp8_q, dn_fp8_q.T)
        baseline_down *= baseline_act_scale[:, None] * dn_fp8_scale[None, :]
        native_down = float_matmul(native_act_fp8_q, dn_fp8_q.T)
        native_down *= native_act_scale[:, None] * dn_fp8_scale[None, :]
        reference_sum += ref_down
        baseline_sum += terminal_fp8_qdq(baseline_down)
        native_sum += terminal_fp8_qdq(native_down)

        sampled_peak_rss = max(sampled_peak_rss, rss_mib())
        expert_seconds.append(time.perf_counter() - expert_started)
        del (
            gu,
            dn,
            gu_fp8_q,
            gu_fp8_scale,
            gu_int6,
            gu_int6_scale,
            gu_int6_deq,
            dn_fp8_q,
            dn_fp8_scale,
            ref_gu,
            baseline_gu,
            native_gu,
            ref_g,
            ref_u,
            base_g,
            base_u,
            native_g,
            native_u,
            ref_act,
            baseline_act,
            native_act,
            baseline_act_fp8_q,
            native_act_fp8_q,
            ref_down,
            baseline_down,
            native_down,
        )
        gc.collect()

    # _gather_branch_sum_f8 stores the accumulated result as BF16.
    baseline_sum = bf16_round(baseline_sum)
    native_sum = bf16_round(native_sum)
    baseline_metrics = sqnr(reference_sum, baseline_sum)
    native_metrics = sqnr(reference_sum, native_sum)
    return {
        "shape": {"M": M, "K": K, "I": intermediate, "O": O, "experts": EXPERTS},
        "seed": seed,
        "route_weights": route_weights.tolist(),
        "baseline_fp8_A_GU_ACT_DN": baseline_metrics,
        "native_int8_A_int6_GU_fp8_ACT_DN": native_metrics,
        "native_sqnr_delta_vs_baseline_db": native_metrics["sqnr_db"] - baseline_metrics["sqnr_db"],
        "timing_seconds": {
            "total": time.perf_counter() - started,
            "experts": expert_seconds,
        },
        "sampled_peak_rss_mib": sampled_peak_rss,
    }


def main() -> None:
    # Sanity-check the local NumPy BF16 emulation against torch before the screen.
    sanity_rng = np.random.default_rng(7)
    sanity = sanity_rng.standard_normal(4096, dtype=np.float32)
    torch_bf16 = torch.from_numpy(sanity.copy()).to(torch.bfloat16).float().numpy()
    if not np.array_equal(bf16_round(sanity), torch_bf16):
        raise AssertionError("NumPy BF16 round-to-nearest-even disagrees with torch")

    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results = []
    for intermediate in INTERMEDIATE_SIZES:
        for seed in SEEDS:
            print(f"running K={K} I={intermediate} seed={seed}", flush=True)
            case = run_case(intermediate, seed)
            results.append(case)
            base = case["baseline_fp8_A_GU_ACT_DN"]
            native = case["native_int8_A_int6_GU_fp8_ACT_DN"]
            print(
                f"  baseline={base['sqnr_db']:.4f} dB "
                f"native={native['sqnr_db']:.4f} dB "
                f"native_margin={native['margin_over_22db']:.4f} dB "
                f"seconds={case['timing_seconds']['total']:.2f}",
                flush=True,
            )

    native_sqnr = [case["native_int8_A_int6_GU_fp8_ACT_DN"]["sqnr_db"] for case in results]
    baseline_sqnr = [case["baseline_fp8_A_GU_ACT_DN"]["sqnr_db"] for case in results]
    document = {
        "experiment": "native INT8 A + signed-6-bit GU precision screen",
        "code_sha256": code_sha256,
        "fp8_helper": str(FP8_HELPER_PATH.relative_to(REPO)),
        "constants": {
            "K": K,
            "I": list(INTERMEDIATE_SIZES),
            "M": M,
            "O": O,
            "experts": EXPERTS,
            "seeds_per_shape": list(SEEDS),
            "sqnr_target_db": SQNR_TARGET_DB,
        },
        "method": {
            "source": "NumPy Gaussian values rounded to BF16; float32 forward reference",
            "baseline": "per-row FP8 A/GU, power-of-two-bound FP8 ACT, per-row FP8 DN",
            "native": "per-row INT8 A and signed-6-bit GU stored in int8; same FP8 ACT/DN",
            "native_act_bound": "native_A_scale * (127/448) before squaring",
            "terminal": "each expert DN output quantized per row per one 256-column block to FP8, dequantized, then branch-summed",
            "md_int_accumulation": "float32 matmul of integer values; K*127*31=16125952 < 2^24",
            "route": "fixed positive weights [1..8] normalized to sum to one; applied before ACT quantization",
        },
        "summary": {
            "minimum_baseline_sqnr_db": min(baseline_sqnr),
            "minimum_native_sqnr_db": min(native_sqnr),
            "minimum_native_margin_over_22db": min(native_sqnr) - SQNR_TARGET_DB,
            "all_native_pass_22db": all(value >= SQNR_TARGET_DB for value in native_sqnr),
        },
        "cases": results,
        "limitations": [
            "CPU mathematical simulation only",
            "does not prove GPU compilation",
            "does not prove platform/judge precision",
            "does not measure or prove GPU speed",
            "does not reproduce GPU instruction-level rounding or accumulation order",
        ],
    }
    RESULT_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
