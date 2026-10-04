"""CPU checks for the proposed sign+5-bit coding of finite E4M3FN values."""

import math

import torch


FP8 = torch.float8_e4m3fn


def encode_fp6(q):
    raw = q.contiguous().view(torch.uint8)
    sign = raw >> 7
    mag = raw & 0x7F
    if bool((mag == 0x7F).any()):
        raise ValueError("NaN has no FP6 code")
    value = q.abs().float()
    small = torch.round(value / 8.0).to(torch.uint8)
    large = mag - 96
    abs_code = torch.where(mag >= 104, large, small)
    if bool((abs_code > 30).any()):
        raise AssertionError("reserved abs code 31 produced")
    return (sign << 5) | abs_code


def decode_fp6(code):
    code = code.to(torch.uint8)
    sign = code >> 5
    abs_code = code & 31
    if bool((abs_code == 31).any()):
        raise ValueError("reserved FP6 code 31")
    low_values = (torch.arange(8, dtype=torch.float32) * 8).to(FP8)
    low_raw = low_values.view(torch.uint8)
    low_index = torch.clamp(abs_code.long(), max=7)
    mag_raw = torch.where(abs_code < 8, low_raw[low_index], abs_code + 96)
    return (mag_raw | (sign << 7)).contiguous().view(FP8)


def verify_all_finite_codes():
    raw_all = torch.arange(256, dtype=torch.uint8)
    finite = (raw_all & 0x7F) != 0x7F
    raw = raw_all[finite].contiguous()
    q = raw.view(FP8)
    code = encode_fp6(q)
    qr = decode_fp6(code)
    rr = qr.view(torch.uint8)

    mag_all = raw_all & 0x7F
    low = torch.where(mag_all <= 72, 0,
          torch.where(mag_all < 84, 80,
          torch.where(mag_all <= 90, 88,
          torch.where(mag_all < 94, 92,
          torch.where(mag_all <= 97, 96,
          torch.where(mag_all < 99, 98,
          torch.where(mag_all <= 101, 100,
          torch.where(mag_all < 103, 102, 104)))))))).to(torch.uint8)
    kernel_rr = (raw_all & 0x80) | torch.where(mag_all >= 104, mag_all, low)
    assert torch.equal(kernel_rr[finite], rr)
    assert torch.equal(kernel_rr[~finite], raw_all[~finite])

    mag = raw & 0x7F
    sign = raw >> 7
    expected = torch.round(q.abs().float() / 8.0) * 8.0
    exact_large = mag >= 104
    assert torch.equal(rr[exact_large], raw[exact_large])
    assert torch.equal(qr.abs().float()[~exact_large], expected[~exact_large])
    assert torch.equal(rr >> 7, sign)
    assert int(code.max()) == 62 and not bool(((code & 31) == 31).any())

    valid_codes = torch.cat((torch.arange(31), torch.arange(32, 63))).to(torch.uint8)
    assert torch.equal(encode_fp6(decode_fp6(valid_codes)), valid_codes)

    checks = {
        0x67: (8, 64.0),    # +60 rounds ties-to-even to code 8 / +64
        0x68: (8, 64.0),    # +64 is exact
        0x7E: (30, 448.0),  # +448 is exact
        0x80: (32, -0.0),   # negative zero retains its sign bit
    }
    for raw_in, (code_expected, value_expected) in checks.items():
        x = torch.tensor([raw_in], dtype=torch.uint8).view(FP8)
        c = encode_fp6(x)
        y = decode_fp6(c)
        assert int(c[0]) == code_expected
        assert int(y.view(torch.uint8)[0]) == (raw_in if raw_in != 0x67 else 0x68)
        assert float(y.float()[0]) == value_expected
    return int(finite.sum()), int(exact_large.sum())


def current_row_fp8(weights):
    wf = weights.float()
    amax = wf.abs().amax(dim=1, keepdim=True)
    scale = torch.maximum(amax / 448.0, torch.full_like(amax, 1e-12))
    return (wf / scale).to(FP8), scale


def uniform6_g64(q):
    rows, k = q.shape
    w = q.float().view(rows, k // 64, 64)
    scale = w.abs().amax(dim=2, keepdim=True) / 31.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qi = torch.clamp(torch.round(w / scale), -31, 31)
    return (qi * scale).view(rows, k).to(FP8)


def metrics(reference_q, candidate_q, row_scale):
    reference = reference_q.float() * row_scale
    candidate = candidate_q.float() * row_scale
    error = candidate - reference
    signal_rms = float(torch.sqrt(torch.mean(reference.square())))
    error_rms = float(torch.sqrt(torch.mean(error.square())))
    return {
        "rms_error": error_rms,
        "relative_rms": error_rms / signal_rms,
        "sqnr_db_vs_fp8": 20.0 * math.log10(signal_rms / error_rms),
        "exact_fraction": float(torch.mean((candidate_q.view(torch.uint8) == reference_q.view(torch.uint8)).float())),
    }


def gaussian_case(name, rows, k, seed):
    gen = torch.Generator().manual_seed(seed)
    # P1 source weights are BF16; scale makes the chosen Gaussian sigma immaterial.
    weights = torch.randn((rows, k), generator=gen).to(torch.bfloat16)
    q, scale = current_row_fp8(weights)
    fp6 = decode_fp6(encode_fp6(q))
    uniform = uniform6_g64(q)
    max_maps_448 = float(torch.mean((q.abs().float().amax(dim=1) == 448.0).float()))
    return name, max_maps_448, metrics(q, fp6, scale), metrics(q, uniform, scale)


def main():
    finite_count, exact_large_count = verify_all_finite_codes()
    print(f"finite_e4m3fn_codes={finite_count} exact_large_signed_codes={exact_large_count}")
    for result in (
        gaussian_case("c9_gate_up_row_K4096", 2048, 4096, 20260909),
        gaussian_case("c9_down_row_K2048", 2048, 2048, 20260910),
    ):
        name, max_maps_448, fp6, uniform = result
        print(name, f"rows_mapping_max_to_448={max_maps_448:.6f}")
        print("  proposed_fp6", fp6)
        print("  uniform6_g64", uniform)


if __name__ == "__main__":
    main()
