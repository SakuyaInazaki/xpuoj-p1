"""Bounded CPU precision screen for L2-normalized FP8 MD with FP16 accumulation.

This intentionally covers only K=4096, I in {1536, 2048}, M=8, O=256,
eight experts, and two fixed seeds per shape.  It is a mathematical screen,
not a reproduction of GPU compilation, instruction scheduling, or timing.
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
L2_TARGET = 128.0
FP16_PARTIAL_K = 32
TINY_SCALE = 1.0e-12

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FP8_HELPER_PATH = REPO / "experiments/2026-09-08/notes/fp6_e4m3_cpu_verify.py"
RESULT_PATH = HERE / "fp16acc_l2_precision_screen_results.json"


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


def fp8_l2_row_quant(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize original BF16 row L2 to 128, then round to E4M3FN."""
    row_l2 = np.linalg.norm(x, axis=1).astype(np.float32)
    scale = np.maximum(
        row_l2 / np.float32(L2_TARGET), np.float32(TINY_SCALE)
    ).astype(np.float32)
    q = fp8_quant_with_scale(x, scale)
    deq = q * scale[:, None]
    quant_row_l2 = np.linalg.norm(q, axis=1).astype(np.float32)
    return q, scale, deq, quant_row_l2


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
        "finite": bool(np.isfinite(candidate).all()),
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


def fp16_accum_k32_matmul(
    a: np.ndarray, b_t: np.ndarray
) -> tuple[np.ndarray, bool, float]:
    """Approximate FP16 hardware accumulation with ordered K32 partials.

    Each K32 partial dot is computed in float32, rounded to float16, then added
    to a float16 running accumulator.  This is only a CPU rounding proxy.
    """
    accumulator = np.zeros((a.shape[0], b_t.shape[1]), dtype=np.float16)
    finite = True
    max_abs = 0.0
    for k0 in range(0, a.shape[1], FP16_PARTIAL_K):
        partial = np.asarray(
            a[:, k0 : k0 + FP16_PARTIAL_K]
            @ b_t[k0 : k0 + FP16_PARTIAL_K, :],
            dtype=np.float32,
        ).astype(np.float16)
        accumulator = np.add(accumulator, partial, dtype=np.float16)
        finite = finite and bool(np.isfinite(accumulator).all())
        max_abs = max(max_abs, float(np.max(np.abs(accumulator.astype(np.float32)))))
    return accumulator.astype(np.float32), finite, max_abs


def run_case(intermediate: int, seed: int) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    route_weights = np.arange(1, EXPERTS + 1, dtype=np.float32)
    route_weights /= np.sum(route_weights, dtype=np.float32)

    x = gaussian_bf16(rng, (M, K))
    x_fp8_q, x_fp8_scale = fp8_row_quant(x)
    x_l2_q, x_l2_scale, _, x_l2_quant_norm = fp8_l2_row_quant(x)
    # ACT bound deliberately keeps the original amax/448 convention even
    # though L2 dequantization uses x_l2_scale.
    x_bound_scale = np.maximum(
        np.max(np.abs(x), axis=1) / np.float32(FP8_MAX),
        np.float32(TINY_SCALE),
    ).astype(np.float32)

    reference_sum = np.zeros((M, O), dtype=np.float32)
    baseline_sum = np.zeros((M, O), dtype=np.float32)
    l2_f32_sum = np.zeros((M, O), dtype=np.float32)
    l2_f16_sum = np.zeros((M, O), dtype=np.float32)
    max_abs_acc = {
        "baseline_fp8_f32_final": 0.0,
        "l2_fp8_f32_final": 0.0,
        "l2_fp8_f16_final": 0.0,
        "l2_fp8_f16_running": 0.0,
    }
    all_f16_acc_finite = True
    gu_l2_quant_norm_max = 0.0
    sampled_peak_rss = rss_mib()
    expert_seconds: list[float] = []

    for route_weight in route_weights:
        expert_started = time.perf_counter()
        gu = gaussian_bf16(rng, (2 * intermediate, K))
        dn = gaussian_bf16(rng, (O, intermediate))
        sampled_peak_rss = max(sampled_peak_rss, rss_mib())

        gu_fp8_q, gu_fp8_scale = fp8_row_quant(gu)
        gu_l2_q, gu_l2_scale, gu_l2_deq, gu_l2_quant_norm = fp8_l2_row_quant(gu)
        gu_l2_quant_norm_max = max(
            gu_l2_quant_norm_max, float(np.max(gu_l2_quant_norm))
        )
        dn_fp8_q, dn_fp8_scale = fp8_row_quant(dn)
        baseline_bnorm = float(
            np.max(np.linalg.norm(gu_fp8_q * gu_fp8_scale[:, None], axis=1))
        )
        l2_bnorm = float(np.max(np.linalg.norm(gu_l2_deq, axis=1)))

        ref_gu = float_matmul(x, gu.T)
        baseline_acc = float_matmul(x_fp8_q, gu_fp8_q.T)
        l2_f32_acc = float_matmul(x_l2_q, gu_l2_q.T)
        l2_f16_acc, f16_acc_finite, f16_running_max = fp16_accum_k32_matmul(
            x_l2_q, gu_l2_q.T
        )
        all_f16_acc_finite = all_f16_acc_finite and f16_acc_finite
        max_abs_acc["baseline_fp8_f32_final"] = max(
            max_abs_acc["baseline_fp8_f32_final"],
            float(np.max(np.abs(baseline_acc))),
        )
        max_abs_acc["l2_fp8_f32_final"] = max(
            max_abs_acc["l2_fp8_f32_final"],
            float(np.max(np.abs(l2_f32_acc))),
        )
        max_abs_acc["l2_fp8_f16_final"] = max(
            max_abs_acc["l2_fp8_f16_final"],
            float(np.max(np.abs(l2_f16_acc))),
        )
        max_abs_acc["l2_fp8_f16_running"] = max(
            max_abs_acc["l2_fp8_f16_running"], f16_running_max
        )

        baseline_gu = baseline_acc * (
            x_fp8_scale[:, None] * gu_fp8_scale[None, :]
        )
        l2_f32_gu = l2_f32_acc * (
            x_l2_scale[:, None] * gu_l2_scale[None, :]
        )
        l2_f16_gu = l2_f16_acc * (
            x_l2_scale[:, None] * gu_l2_scale[None, :]
        )

        ref_g, ref_u = ref_gu[:, :intermediate], ref_gu[:, intermediate:]
        base_g, base_u = baseline_gu[:, :intermediate], baseline_gu[:, intermediate:]
        l2_f32_g = l2_f32_gu[:, :intermediate]
        l2_f32_u = l2_f32_gu[:, intermediate:]
        l2_f16_g = l2_f16_gu[:, :intermediate]
        l2_f16_u = l2_f16_gu[:, intermediate:]
        ref_act = silu(ref_g) * ref_u * route_weight
        baseline_act = silu(base_g) * base_u * route_weight
        l2_f32_act = silu(l2_f32_g) * l2_f32_u * route_weight
        l2_f16_act = silu(l2_f16_g) * l2_f16_u * route_weight

        baseline_act_scale = act_power2_scale(x_fp8_scale, baseline_bnorm, float(route_weight))
        l2_act_scale = act_power2_scale(x_bound_scale, l2_bnorm, float(route_weight))
        baseline_act_fp8_q = fp8_quant_with_scale(baseline_act, baseline_act_scale)
        l2_f32_act_fp8_q = fp8_quant_with_scale(l2_f32_act, l2_act_scale)
        l2_f16_act_fp8_q = fp8_quant_with_scale(l2_f16_act, l2_act_scale)

        ref_down = float_matmul(ref_act, dn.T)
        baseline_down = float_matmul(baseline_act_fp8_q, dn_fp8_q.T)
        baseline_down *= baseline_act_scale[:, None] * dn_fp8_scale[None, :]
        l2_f32_down = float_matmul(l2_f32_act_fp8_q, dn_fp8_q.T)
        l2_f32_down *= l2_act_scale[:, None] * dn_fp8_scale[None, :]
        l2_f16_down = float_matmul(l2_f16_act_fp8_q, dn_fp8_q.T)
        l2_f16_down *= l2_act_scale[:, None] * dn_fp8_scale[None, :]
        reference_sum += ref_down
        baseline_sum += terminal_fp8_qdq(baseline_down)
        l2_f32_sum += terminal_fp8_qdq(l2_f32_down)
        l2_f16_sum += terminal_fp8_qdq(l2_f16_down)

        sampled_peak_rss = max(sampled_peak_rss, rss_mib())
        expert_seconds.append(time.perf_counter() - expert_started)
        del (
            gu,
            dn,
            gu_fp8_q,
            gu_fp8_scale,
            gu_l2_q,
            gu_l2_scale,
            gu_l2_deq,
            gu_l2_quant_norm,
            dn_fp8_q,
            dn_fp8_scale,
            ref_gu,
            baseline_acc,
            l2_f32_acc,
            l2_f16_acc,
            baseline_gu,
            l2_f32_gu,
            l2_f16_gu,
            ref_g,
            ref_u,
            base_g,
            base_u,
            l2_f32_g,
            l2_f32_u,
            l2_f16_g,
            l2_f16_u,
            ref_act,
            baseline_act,
            l2_f32_act,
            l2_f16_act,
            baseline_act_fp8_q,
            l2_f32_act_fp8_q,
            l2_f16_act_fp8_q,
            ref_down,
            baseline_down,
            l2_f32_down,
            l2_f16_down,
        )
        gc.collect()

    # _gather_branch_sum_f8 stores the accumulated result as BF16.
    baseline_sum = bf16_round(baseline_sum)
    l2_f32_sum = bf16_round(l2_f32_sum)
    l2_f16_sum = bf16_round(l2_f16_sum)
    baseline_metrics = sqnr(reference_sum, baseline_sum)
    l2_f32_metrics = sqnr(reference_sum, l2_f32_sum)
    l2_f16_metrics = sqnr(reference_sum, l2_f16_sum)
    return {
        "shape": {"M": M, "K": K, "I": intermediate, "O": O, "experts": EXPERTS},
        "seed": seed,
        "route_weights": route_weights.tolist(),
        "baseline_fp8_A_GU_ACT_DN": baseline_metrics,
        "l2_fp8_A_GU_f32acc_fp8_ACT_DN": l2_f32_metrics,
        "l2_fp8_A_GU_f16acc_proxy_fp8_ACT_DN": l2_f16_metrics,
        "sqnr_delta_vs_baseline_db": {
            "l2_f32acc": l2_f32_metrics["sqnr_db"] - baseline_metrics["sqnr_db"],
            "l2_f16acc_proxy": l2_f16_metrics["sqnr_db"] - baseline_metrics["sqnr_db"],
        },
        "finite": {
            "reference": bool(np.isfinite(reference_sum).all()),
            "baseline": bool(np.isfinite(baseline_sum).all()),
            "l2_f32acc": bool(np.isfinite(l2_f32_sum).all()),
            "l2_f16acc_proxy": bool(np.isfinite(l2_f16_sum).all()),
            "l2_f16acc_running": all_f16_acc_finite,
        },
        "max_abs_accumulator": max_abs_acc,
        "normalized_quant_row_l2_max": {
            "A": float(np.max(x_l2_quant_norm)),
            "GU_across_experts": gu_l2_quant_norm_max,
        },
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
            l2_f32 = case["l2_fp8_A_GU_f32acc_fp8_ACT_DN"]
            l2_f16 = case["l2_fp8_A_GU_f16acc_proxy_fp8_ACT_DN"]
            print(
                f"  baseline={base['sqnr_db']:.4f} dB "
                f"l2_f32={l2_f32['sqnr_db']:.4f} dB "
                f"l2_f16={l2_f16['sqnr_db']:.4f} dB "
                f"l2_f16_margin={l2_f16['margin_over_22db']:.4f} dB "
                f"finite={case['finite']} "
                f"seconds={case['timing_seconds']['total']:.2f}",
                flush=True,
            )

    baseline_sqnr = [case["baseline_fp8_A_GU_ACT_DN"]["sqnr_db"] for case in results]
    l2_f32_sqnr = [
        case["l2_fp8_A_GU_f32acc_fp8_ACT_DN"]["sqnr_db"] for case in results
    ]
    l2_f16_sqnr = [
        case["l2_fp8_A_GU_f16acc_proxy_fp8_ACT_DN"]["sqnr_db"]
        for case in results
    ]
    all_finite = all(all(case["finite"].values()) for case in results)
    document = {
        "experiment": "L2-normalized FP8 A/GU with FP16 accumulation proxy",
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
            "l2_target": L2_TARGET,
            "fp16_partial_k": FP16_PARTIAL_K,
        },
        "method": {
            "source": "NumPy Gaussian values rounded to BF16; float32 forward reference",
            "baseline": "per-row FP8 A/GU, power-of-two-bound FP8 ACT, per-row FP8 DN",
            "l2_quantization": "for original BF16 A/GU rows, scale=max(row_l2/128, 1e-12), then E4M3FN(x/scale)",
            "l2_f32_control": "float32 dot accumulation of L2-normalized FP8 A/GU, then float32 dequantization and SiLU",
            "l2_f16_accumulation_proxy": "ordered K=32 float32 partial dots, each rounded to float16 and added to a float16 running accumulator; final accumulator converted to float32 before dequantization and SiLU",
            "act_bound": "uses original BF16 A row amax/448 independently of the L2 A dequantization scale; BNORM is measured from each path's actually dequantized GU",
            "terminal": "each expert DN output quantized per row per one 256-column block to FP8, dequantized, then branch-summed",
            "route": "fixed positive weights [1..8] normalized to sum to one; applied before ACT quantization",
        },
        "summary": {
            "minimum_baseline_sqnr_db": min(baseline_sqnr),
            "minimum_l2_f32acc_sqnr_db": min(l2_f32_sqnr),
            "minimum_l2_f16acc_proxy_sqnr_db": min(l2_f16_sqnr),
            "minimum_l2_f32acc_margin_over_22db": min(l2_f32_sqnr)
            - SQNR_TARGET_DB,
            "minimum_l2_f16acc_proxy_margin_over_22db": min(l2_f16_sqnr)
            - SQNR_TARGET_DB,
            "all_l2_f32acc_pass_22db": all(
                value >= SQNR_TARGET_DB for value in l2_f32_sqnr
            ),
            "all_l2_f16acc_proxy_pass_22db": all(
                value >= SQNR_TARGET_DB for value in l2_f16_sqnr
            ),
            "all_paths_finite": all_finite,
        },
        "cases": results,
        "limitations": [
            "CPU mathematical simulation only",
            "does not prove GPU compilation",
            "does not prove platform/judge precision",
            "does not measure or prove GPU speed",
            "K32 partial-dot float16 accumulation is a CPU rounding proxy and does not prove exact GPU instruction-level accumulation or scheduling",
        ],
    }
    RESULT_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
