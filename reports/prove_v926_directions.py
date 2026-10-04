#!/usr/bin/env python3
"""CPU-only index/arithmetic proofs; no GPU correctness or speed claim."""
import json
import math
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


def scalars(a_scale, bnorm, weight):
    # Match the existing left-associated FP32 expression, not a reassociation.
    bound = f32(f32(f32(f32(a_scale * a_scale) * bnorm) * bnorm) * abs(weight))
    exponent = (bits(max(bound, f32(1e-30))) >> 23) & 255
    scale = from_bits((exponent + 10) << 23)
    inverse = from_bits((244 - exponent) << 23)
    half_a = f32(a_scale * 0.5)
    wi = f32(f32(weight * a_scale) * inverse)
    return tuple(bits(x) for x in (scale, inverse, half_a, wi))


def prove_scalars(rng):
    checks = 0
    scenarios = 0
    for experts in (8, 32, 64, 96, 256):
        for topk in (2, 3, 4, 8):
            tokens = 73
            ids = [rng.sample(range(experts), topk) for _ in range(tokens)]
            flat = [e for row in ids for e in row]
            order = sorted(range(len(flat)), key=lambda i: flat[i])
            inverse = [None] * len(order)
            for pos, branch in enumerate(order):
                inverse[branch] = pos
            a = [f32(2 ** rng.uniform(-25, -4)) for _ in range(tokens)]
            norms = [f32(2 ** rng.uniform(-6, 4)) for _ in range(experts)]
            weights = [f32(rng.random()) for _ in flat]
            weights[0] = 0.0
            weights[1] = f32(1e-20)
            producer = [None] * len(flat)
            for branch, expert in enumerate(flat):
                producer[inverse[branch]] = scalars(a[branch // topk], norms[expert], weights[branch])
            for pos, branch in enumerate(order):
                reference = scalars(a[branch // topk], norms[flat[branch]], weights[branch])
                assert reference == producer[pos]
                checks += 1
            scenarios += 1
    return {"scenarios": scenarios, "branch_scalar_tuples_bit_equal": checks,
            "scope": "FP32 CPU arithmetic and sorted-row ownership only; excludes GPU FMA/lowering"}


def prove_interleave(rng):
    checks = 0
    for I in (1536, 2048):
        H, E, BN, BK = 4096, 256, 128, 128
        R, KT = I // BN, H // BK
        for _ in range(40000):
            e, n, h, which = rng.randrange(E), rng.randrange(I), rng.randrange(H), rng.randrange(2)
            r, ni, kt, ki = n // BN, n % BN, h // BK, h % BK
            flat = (((((e * R + r) * KT + kt) * BN + ni) * 2 + which) * BK + ki)
            col = flat % BK
            row = flat // BK
            branch = row % 2
            n_in_tile = (row // 2) % BN
            tile = row // (2 * BN)
            k_tile = tile % KT
            n_tile = (tile // KT) % R
            expert = tile // (KT * R)
            assert (expert, n_tile * BN + n_in_tile, k_tile * BK + col, branch) == (e, n, h, which)
            start = ((e * R + r) * KT + kt) * (2 * BN)
            assert row == start + 2 * ni + which
            assert flat < E * 2 * I * H
            checks += 1
    return {"coordinate_checks": checks, "shapes": ["E256/I1536/H4096", "E256/I2048/H4096"],
            "scope": "Bijective packed coordinates and descriptor tile extent; excludes TMA and MMA"}


def main():
    rng = random.Random(930926)
    # The clamp counterexample concerns current static DN dequantization,
    # independent of whether typical official inputs exercise this branch.
    acc_times_bscale, C, a_scale = 2.0, 4.0, 1e-14
    q = acc_times_bscale / C  # 0.5 is exactly representable in E4M3.
    stored_scale = max(a_scale * C, 1e-12)
    expected = acc_times_bscale * a_scale
    reconstructed = q * stored_scale
    assert math.isclose(reconstructed / expected, 25.0)
    result = {
        "cpu_only": True,
        "row_scalar_hoist": prove_scalars(rng),
        "tile_interleaved_gu": prove_interleave(rng),
        "static_dn_clamp_counterexample": {
            "acc_times_bscale": acc_times_bscale, "C": C, "a_scale": a_scale,
            "q": q, "stored_scale": stored_scale, "expected": expected,
            "reconstructed": reconstructed, "ratio": reconstructed / expected,
            "claim": "Exposes an algebraic clamp mismatch, not a measured OJ failure"},
    }
    out = ROOT / "reports/2026-09-30-v926-direction-proofs.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
